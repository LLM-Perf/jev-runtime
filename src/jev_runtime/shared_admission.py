"""Single-host admission shared by API workers through the durable lease registry."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from jev_runtime.admission import Admission
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry


class SharedAdmission:
    """Reserve expanded work until its request lease is safely released.

    The SQLite transaction is the admission linearization point. A worker never
    dispatches before its ticket is durably ADMITTED. Exiting the context does not
    release that reservation: normal completion, confirmed cancellation or explicit
    recovery deletes the lease and its ticket atomically. Uncertain aborts and dead
    workers keep their reservations. There is no timeout-based capacity reclamation.
    """

    def __init__(self, registry: Registry, backend: str, **limits):
        self.registry, self.backend = registry, backend
        self._limits = Admission(**limits).limits()
        self._policy = json.dumps(self._limits, sort_keys=True)

    def start(self, registry: Registry, backend: str) -> None:
        if registry is not self.registry or backend != self.backend:
            raise ValueError("Admission must share the runtime's registry and engine identity")
        registry.start_worker(backend, self._limits)

    def snapshot(self) -> dict:
        with self.registry._connection() as db:
            row = db.execute(
                "SELECT COALESCE(SUM(state='ADMITTED'),0) AS requests, "
                "COALESCE(SUM(CASE WHEN state='ADMITTED' THEN tokens ELSE 0 END),0) "
                "AS expanded_tokens, "
                "COALESCE(SUM(CASE WHEN state='ADMITTED' THEN branches ELSE 0 END),0) "
                "AS expanded_branches, "
                "COALESCE(SUM(state='QUEUED'),0) AS queued_requests "
                "FROM admission_tickets WHERE backend=?",
                (self.backend,),
            ).fetchone()
        return {"scope": "shared_registry_engine", "limits": dict(self._limits), **dict(row)}

    def _check_worker(self, db) -> None:
        row = db.execute(
            "SELECT w.state,a.policy FROM workers w JOIN worker_admission a ON a.owner=w.owner "
            "WHERE w.owner=? AND w.backend=?",
            (self.registry.owner, self.backend),
        ).fetchone()
        policy = db.execute(
            "SELECT policy FROM admission_policies WHERE backend=?",
            (self.backend,),
        ).fetchone()
        if (
            not row
            or row["state"] != "SERVING"
            or row["policy"] != self._policy
            or not policy
            or policy["policy"] != self._policy
        ):
            raise JevError(
                "admission_unavailable", "Admission worker or policy is not serving", 503
            )

    def _enqueue(
        self, lease_id: str, tenant: str, tokens: int, branches: int, branch_ids: list[str] | None
    ) -> bool:
        with self.registry._transaction() as db:
            self._check_worker(db)
            lease = db.execute(
                "SELECT l.owner,b.backend,COALESCE(t.tenant,'default') AS tenant "
                "FROM leases l JOIN bundles b ON b.ref=l.ref "
                "LEFT JOIN lease_tenants t ON t.lease_id=l.id WHERE l.id=?",
                (lease_id,),
            ).fetchone()
            if (
                not lease
                or lease["owner"] != self.registry.owner
                or lease["backend"] != self.backend
                or lease["tenant"] != tenant
            ):
                raise JevError(
                    "admission_lease_mismatch",
                    "Admission requires the owned tenant/engine lease",
                    409,
                )
            if db.execute(
                "SELECT 1 FROM admission_tickets WHERE lease_id=?", (lease_id,)
            ).fetchone():
                raise JevError("duplicate_admission", "Lease already has an admission ticket", 409)
            queued = db.execute(
                "SELECT COUNT(*) AS total,COALESCE(SUM(tenant=?),0) AS tenant_total "
                "FROM admission_tickets WHERE backend=? AND state='QUEUED'",
                (tenant, self.backend),
            ).fetchone()
            if (
                queued["total"] >= self._limits["max_queue"]
                or queued["tenant_total"] >= self._limits["max_tenant_queue"]
            ):
                raise JevError("queue_full", "Shared admission queue is full", 429)
            # Remove idle tenant scheduling entries. Authenticated tenant names
            # are configured, but long-lived deployments need not accumulate old ones.
            db.execute(
                "DELETE FROM admission_tenants WHERE backend=? AND tenant NOT IN "
                "(SELECT tenant FROM admission_tickets WHERE backend=?)",
                (self.backend, self.backend),
            )
            db.execute(
                "INSERT OR IGNORE INTO admission_tenants SELECT ?,?,COALESCE(MAX(turn),0)+1 "
                "FROM admission_tenants WHERE backend=?",
                (self.backend, tenant, self.backend),
            )
            db.execute(
                "INSERT INTO admission_tickets(lease_id,backend,tenant,tokens,branches,state) "
                "VALUES(?,?,?,?,?,'QUEUED')",
                (lease_id, self.backend, tenant, tokens, branches),
            )
            if branch_ids is not None:
                # Recovery IDs and the reservation become durable together. Neither
                # an admitted caller nor a queued waiter can dispatch before commit.
                self.registry._record_branches(db, lease_id, branch_ids)
            return self._try_admit(db, lease_id)

    def _try_admit(self, db, lease_id: str) -> bool:
        usage = db.execute(
            "SELECT tenant,COUNT(*) AS requests,SUM(tokens) AS tokens,SUM(branches) AS branches "
            "FROM admission_tickets WHERE backend=? AND state='ADMITTED' GROUP BY tenant",
            (self.backend,),
        ).fetchall()
        total = {key: sum(row[key] for row in usage) for key in ("requests", "tokens", "branches")}
        if total["requests"] >= self._limits["max_requests"]:
            return False
        tenants = {row["tenant"]: dict(row) for row in usage}
        heads = db.execute(
            "SELECT t.*,o.identity FROM admission_tickets t "
            "JOIN leases l ON l.id=t.lease_id JOIN owners o ON o.owner=l.owner "
            "JOIN admission_tenants a ON a.backend=t.backend AND a.tenant=t.tenant "
            "WHERE t.id IN (SELECT MIN(id) FROM admission_tickets WHERE backend=? "
            "AND state='QUEUED' GROUP BY tenant) ORDER BY a.turn,t.id",
            (self.backend,),
        ).fetchall()
        for head in heads:
            used = tenants.get(head["tenant"], {"requests": 0, "tokens": 0, "branches": 0})
            eligible = (
                used["requests"] < self._limits["max_tenant_requests"]
                and total["tokens"] + head["tokens"] <= self._limits["max_tokens"]
                and used["tokens"] + head["tokens"] <= self._limits["max_tenant_tokens"]
                and total["branches"] + head["branches"] <= self._limits["max_branches"]
                and used["branches"] + head["branches"] <= self._limits["max_tenant_branches"]
            )
            if not eligible or self.registry._owner_status(json.loads(head["identity"])) == "dead":
                continue
            if head["lease_id"] != lease_id:
                return False
            db.execute(
                "UPDATE admission_tickets SET state='ADMITTED' WHERE lease_id=?", (lease_id,)
            )
            db.execute(
                "UPDATE admission_tenants SET turn=(SELECT COALESCE(MAX(turn),0)+1 "
                "FROM admission_tenants WHERE backend=?) WHERE backend=? AND tenant=?",
                (self.backend, self.backend, head["tenant"]),
            )
            return True
        return False

    def _poll(self, lease_id: str) -> bool:
        with self.registry._transaction() as db:
            self._check_worker(db)
            ticket = db.execute(
                "SELECT t.state FROM admission_tickets t JOIN leases l ON l.id=t.lease_id "
                "WHERE t.lease_id=? AND t.backend=? AND l.owner=?",
                (lease_id, self.backend, self.registry.owner),
            ).fetchone()
            if ticket is None:
                raise JevError("admission_lease_mismatch", "Admission lease no longer exists", 409)
            return ticket["state"] == "ADMITTED" or self._try_admit(db, lease_id)

    @asynccontextmanager
    async def acquire(
        self,
        tokens: int,
        tenant: str = "default",
        *,
        branches: int = 1,
        lease_id: str | None = None,
        branch_ids: list[str] | None = None,
    ):
        if lease_id is None:
            raise JevError(
                "admission_lease_required", "Shared admission requires a durable lease", 409
            )
        if tokens <= 0:
            raise JevError("invalid_token_budget", "Token demand must be positive", 400)
        if tokens > min(self._limits["max_tokens"], self._limits["max_tenant_tokens"]):
            raise JevError(
                "engine_token_budget", "Request exceeds shared admission token budget", 413
            )
        if branches <= 0 or branches > min(
            self._limits["max_branches"], self._limits["max_tenant_branches"]
        ):
            raise JevError(
                "engine_branch_budget", "Request exceeds shared admission branch budget", 413
            )
        if branch_ids is not None and (
            len(branch_ids) != branches
            or any(not isinstance(branch, str) or not branch for branch in branch_ids)
            or len(set(branch_ids)) != branches
        ):
            raise JevError(
                "admission_branch_mismatch", "Expected one distinct engine ID per branch", 409
            )
        queued = False
        try:
            admitted = self._enqueue(lease_id, tenant, tokens, branches, branch_ids)
            queued = not admitted
            while not admitted:
                await asyncio.sleep(0.02)
                admitted = self._poll(lease_id)
                queued = not admitted
            yield
        finally:
            if queued:
                # Only never-dispatched waiters can be removed independently of
                # the lease. An ADMITTED ticket survives even a failed abort.
                with self.registry._transaction() as db:
                    db.execute(
                        "DELETE FROM admission_tickets WHERE lease_id=? AND state='QUEUED' "
                        "AND lease_id IN (SELECT id FROM leases WHERE owner=?)",
                        (lease_id, self.registry.owner),
                    )
