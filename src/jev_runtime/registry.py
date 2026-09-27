from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jev_runtime.adapters import AdapterArtifact, AdapterBinding
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle

REGISTRY_SCHEMA_SQL = """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS bundles (
                    ref TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE,
                    manifest TEXT NOT NULL, state TEXT NOT NULL,
                    backend TEXT, error TEXT, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS routes (
                    alias TEXT PRIMARY KEY, ref TEXT REFERENCES bundles(ref),
                    generation INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                    id TEXT PRIMARY KEY, ref TEXT NOT NULL REFERENCES bundles(ref),
                    owner TEXT NOT NULL, request_id TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS leases_by_ref ON leases(ref);
                CREATE UNIQUE INDEX IF NOT EXISTS unique_active_request ON leases(request_id);
                CREATE TABLE IF NOT EXISTS owners (
                    owner TEXT PRIMARY KEY, identity TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers (
                    owner TEXT PRIMARY KEY REFERENCES owners(owner),
                    backend TEXT NOT NULL, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_bundles (
                    owner TEXT NOT NULL REFERENCES workers(owner),
                    ref TEXT NOT NULL REFERENCES bundles(ref), digest TEXT NOT NULL,
                    PRIMARY KEY(owner, ref)
                );
                CREATE TABLE IF NOT EXISTS worker_deployments (
                    owner TEXT PRIMARY KEY REFERENCES owners(owner),
                    deployment TEXT NOT NULL, release TEXT NOT NULL,
                    expected_workers INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lease_work (
                    lease_id TEXT PRIMARY KEY REFERENCES leases(id) ON DELETE CASCADE,
                    branches TEXT NOT NULL, phase TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lease_tenants (
                    lease_id TEXT PRIMARY KEY REFERENCES leases(id) ON DELETE CASCADE,
                    tenant TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS admission_policies (
                    backend TEXT PRIMARY KEY, policy TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_admission (
                    owner TEXT PRIMARY KEY REFERENCES workers(owner), policy TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS admission_tenants (
                    backend TEXT NOT NULL, tenant TEXT NOT NULL, turn INTEGER NOT NULL,
                    PRIMARY KEY(backend,tenant)
                );
                CREATE TABLE IF NOT EXISTS admission_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lease_id TEXT NOT NULL UNIQUE REFERENCES leases(id) ON DELETE CASCADE,
                    backend TEXT NOT NULL, tenant TEXT NOT NULL,
                    tokens INTEGER NOT NULL CHECK(tokens>0),
                    branches INTEGER NOT NULL CHECK(branches>0),
                    state TEXT NOT NULL CHECK(state IN ('QUEUED','ADMITTED'))
                );
                CREATE INDEX IF NOT EXISTS admission_by_backend
                    ON admission_tickets(backend,state,tenant);
                CREATE TABLE IF NOT EXISTS cancel_commands (
                    id TEXT PRIMARY KEY, lease_id TEXT NOT NULL UNIQUE,
                    owner TEXT NOT NULL, request_id TEXT NOT NULL,
                    tenant TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cancel_commands_by_owner
                    ON cancel_commands(owner,state);
                CREATE TABLE IF NOT EXISTS adapters (
                    ref TEXT PRIMARY KEY, manifest TEXT NOT NULL, digest TEXT NOT NULL,
                    binding TEXT NOT NULL, backend TEXT NOT NULL, state TEXT NOT NULL,
                    operation TEXT, owner TEXT, error TEXT, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bundle_adapters (
                    ref TEXT PRIMARY KEY REFERENCES bundles(ref),
                    adapter_ref TEXT NOT NULL REFERENCES adapters(ref)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS adapter_engine_ids
                    ON adapters(backend,json_extract(binding,'$.engine_id'));
                CREATE UNIQUE INDEX IF NOT EXISTS adapter_engine_names
                    ON adapters(backend,json_extract(binding,'$.engine_name'));
                CREATE INDEX IF NOT EXISTS bundles_by_adapter ON bundle_adapters(adapter_ref);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                    action TEXT NOT NULL, details TEXT NOT NULL
                );
            """


@dataclass(frozen=True)
class Snapshot:
    bundle: Bundle
    generation: int
    lease_id: str


class Registry:
    """Transactional bundle/route registry for a single-node deployment.

    Multiple API processes can use the same local SQLite database. Leases are
    durable and never expire merely because an observer timed out: a failed
    worker can leave a lease requiring confirmed cancellation before recovery.
    Network filesystems and multi-node SQLite are not supported deployments.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner = uuid.uuid4().hex
        self._pid = os.getpid()
        self._connection_lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        with self._connection() as db:
            db.executescript(REGISTRY_SCHEMA_SQL)
            db.execute(
                "INSERT INTO owners VALUES(?,?,?)",
                (self.owner, json.dumps(self._process_identity()), time.time()),
            )

    @staticmethod
    def _process_identity() -> dict:
        identity = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "boot_id": None,
            "start_ticks": None,
            "pid_namespace": None,
        }
        try:
            identity["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            identity["pid_namespace"] = os.readlink("/proc/self/ns/pid")
            identity["start_ticks"] = (
                Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, IndexError):
            pass
        return identity

    @staticmethod
    def _owner_status(identity: dict) -> str:
        """Fail closed without a matching host and verifiable Linux process identity."""
        if (
            identity.get("host") != socket.gethostname()
            or not identity.get("boot_id")
            or not identity.get("start_ticks")
            or not identity.get("pid_namespace")
        ):
            return "unknown"
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if (
                boot != identity["boot_id"]
                or os.readlink("/proc/self/ns/pid") != identity["pid_namespace"]
            ):
                return "unknown"
            try:
                fields = (
                    Path(f"/proc/{int(identity['pid'])}/stat").read_text().rsplit(")", 1)[1].split()
                )
            except FileNotFoundError:
                return "dead"
            if fields[0] == "Z" or fields[19] != identity["start_ticks"]:
                return "dead"
            return "alive"
        except (OSError, ValueError, KeyError, IndexError):
            return "unknown"

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if os.getpid() != self._pid:
            raise RuntimeError("Create a new Registry in each worker; never reuse one after fork")
        # Keep one connection per registry owner. Closing the last connection
        # after every transaction forces WAL cleanup/checkpoint work onto every
        # request. Serialize access across threads without weakening durability.
        with self._connection_lock:
            if self._db is None:
                self._db = sqlite3.connect(
                    self.path, timeout=5, isolation_level=None, check_same_thread=False
                )
                self._db.row_factory = sqlite3.Row
                self._db.execute("PRAGMA foreign_keys=ON")
                self._db.execute("PRAGMA synchronous=FULL")
            yield self._db

    def close(self) -> None:
        with self._connection_lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    @staticmethod
    def _event(db: sqlite3.Connection, action: str, **details) -> None:
        db.execute(
            "INSERT INTO events(ts,action,details) VALUES(?,?,?)",
            (
                time.time(),
                action,
                json.dumps(details, sort_keys=True),
            ),
        )

    def upload(self, bundle: Bundle) -> dict:
        with self._transaction() as db:
            adapter_ref = None
            if bundle.model.adapter_id:
                adapter_ref = f"{bundle.model.adapter_id}@{bundle.model.adapter_revision}"
                adapter = db.execute(
                    "SELECT * FROM adapters WHERE ref=?", (adapter_ref,)
                ).fetchone()
                if adapter is None:
                    raise JevError(
                        "adapter_not_found", "Register the immutable adapter before its bundle", 409
                    )
                artifact = AdapterArtifact.model_validate_json(adapter["manifest"])
                if (artifact.base_model_id, artifact.base_model_revision) != (
                    bundle.model.id,
                    bundle.model.revision,
                ):
                    raise JevError(
                        "adapter_base_mismatch",
                        "Adapter belongs to a different base model revision",
                        409,
                    )
            previous = db.execute(
                "SELECT * FROM bundles WHERE ref=?", (bundle.reference,)
            ).fetchone()
            if previous:
                if previous["digest"] != bundle.digest:
                    raise JevError(
                        "immutable_version", "A bundle version cannot be overwritten", 409
                    )
            else:
                db.execute(
                    "INSERT INTO bundles VALUES(?,?,?,?,?,?,?)",
                    (
                        bundle.reference,
                        bundle.digest,
                        bundle.model_dump_json(),
                        "VALIDATED",
                        None,
                        None,
                        time.time(),
                    ),
                )
                self._event(db, "upload", ref=bundle.reference, digest=bundle.digest)
                if adapter_ref is not None:
                    db.execute(
                        "INSERT INTO bundle_adapters VALUES(?,?)", (bundle.reference, adapter_ref)
                    )
        return self.inspect(bundle.reference)

    def begin_prepare(self, reference: str, backend: str, request_id: str) -> tuple[Bundle, str]:
        with self._transaction() as db:
            row = self._get(db, reference)
            self._adapter_ready(db, reference, backend)
            if row["state"] not in ("VALIDATED", "FAILED", "RETIRED"):
                raise JevError("invalid_state", f"Cannot prepare a {row['state']} bundle", 409)
            if db.execute("SELECT 1 FROM leases WHERE ref=?", (reference,)).fetchone():
                raise JevError(
                    "bundle_in_use",
                    "Outstanding inference must be recovered before preparation",
                    409,
                )
            db.execute(
                "UPDATE bundles SET state='PREPARING',backend=?,error=NULL WHERE ref=?",
                (
                    backend,
                    reference,
                ),
            )
            lease = uuid.uuid4().hex
            db.execute(
                "INSERT INTO leases VALUES(?,?,?,?,?)",
                (lease, reference, self.owner, request_id, time.time()),
            )
            db.execute("INSERT INTO lease_work VALUES(?,?,'inflight')", (lease, "[]"))
            self._event(db, "prepare", ref=reference, backend=backend)
            return Bundle.model_validate_json(row["manifest"]), lease

    def finish_prepare(
        self,
        reference: str,
        lease_id: str,
        error: str | None = None,
        unconfirmed: set[str] | None = None,
    ) -> None:
        """Publish preparation and settle its lease in one crash-safe transaction."""
        with self._transaction() as db:
            row = self._get(db, reference)
            if row["state"] != "PREPARING":
                raise JevError("invalid_state", "Prepare was superseded", 409)
            if not db.execute(
                "SELECT 1 FROM leases WHERE id=? AND ref=? AND owner=?",
                (lease_id, reference, self.owner),
            ).fetchone():
                raise JevError("lease_not_owned", "Preparation lease is not owned", 409)
            if unconfirmed:
                error = error or "cancellation unconfirmed"
                db.execute(
                    "UPDATE lease_work SET branches=?,phase='abort_pending' WHERE lease_id=?",
                    (json.dumps(sorted(unconfirmed)), lease_id),
                )
                self._event(
                    db, "abort_pending", lease_id=lease_id, engine_request_ids=sorted(unconfirmed)
                )
            else:
                db.execute("DELETE FROM leases WHERE id=?", (lease_id,))
            state = "FAILED" if error else "READY"
            db.execute("UPDATE bundles SET state=?,error=? WHERE ref=?", (state, error, reference))
            self._event(db, "prepare_finished", ref=reference, state=state)

    @staticmethod
    def _get(db: sqlite3.Connection, reference: str):
        row = db.execute("SELECT * FROM bundles WHERE ref=?", (reference,)).fetchone()
        if not row:
            raise JevError("bundle_not_found", f"Unknown bundle {reference}", 404)
        return row

    def start_worker(self, backend: str, admission_limits: dict | None = None) -> None:
        with self._transaction() as db:
            policy = json.dumps(admission_limits, sort_keys=True)
            current = db.execute(
                "SELECT policy FROM admission_policies WHERE backend=?", (backend,)
            ).fetchone()
            if (current is None or current["policy"] != policy) and db.execute(
                "SELECT 1 FROM leases l JOIN bundles b ON b.ref=l.ref WHERE b.backend=?",
                (backend,),
            ).fetchone():
                raise JevError(
                    "admission_policy_conflict",
                    "Drain admission leases before changing limits",
                    409,
                )
            for worker in db.execute(
                "SELECT w.owner,o.identity,a.policy FROM workers w "
                "JOIN owners o ON o.owner=w.owner "
                "LEFT JOIN worker_admission a ON a.owner=w.owner "
                "WHERE w.backend=? AND w.state!='STOPPED'",
                (backend,),
            ).fetchall():
                if (
                    worker["policy"] != policy
                    and self._owner_status(json.loads(worker["identity"])) != "dead"
                ):
                    raise JevError(
                        "admission_policy_conflict",
                        "All workers must share admission limits; "
                        "stop legacy workers before migration",
                        409,
                    )
            db.execute(
                "INSERT INTO admission_policies VALUES(?,?) "
                "ON CONFLICT(backend) DO UPDATE SET policy=excluded.policy",
                (backend, policy),
            )
            db.execute(
                "INSERT INTO workers VALUES(?,?,'STARTING') ON CONFLICT(owner) DO UPDATE SET "
                "backend=excluded.backend,state='STARTING'",
                (self.owner, backend),
            )
            db.execute("DELETE FROM worker_bundles WHERE owner=?", (self.owner,))
            db.execute(
                "INSERT INTO worker_admission VALUES(?,?) "
                "ON CONFLICT(owner) DO UPDATE SET policy=excluded.policy",
                (self.owner, policy),
            )

    def record_worker_prepared(self, reference: str) -> None:
        with self._transaction() as db:
            bundle = self._get(db, reference)
            worker = db.execute("SELECT * FROM workers WHERE owner=?", (self.owner,)).fetchone()
            if not worker or worker["backend"] != bundle["backend"]:
                raise JevError("worker_mismatch", "Worker is not registered for this engine", 409)
            db.execute(
                "INSERT INTO worker_bundles VALUES(?,?,?) ON CONFLICT(owner,ref) DO UPDATE "
                "SET digest=excluded.digest",
                (self.owner, reference, bundle["digest"]),
            )

    def forget_worker_prepared(self, reference: str) -> None:
        with self._transaction() as db:
            db.execute(
                "DELETE FROM worker_bundles WHERE owner=? AND ref=?", (self.owner, reference)
            )

    def serve_worker(self, backend: str) -> bool:
        """Join traffic only if every current route was validated by this worker."""
        with self._transaction() as db:
            missing = db.execute(
                "SELECT 1 FROM routes r JOIN bundles b ON b.ref=r.ref "
                "LEFT JOIN worker_bundles w ON w.ref=b.ref AND w.owner=? AND w.digest=b.digest "
                "WHERE b.backend=? AND w.owner IS NULL LIMIT 1",
                (self.owner, backend),
            ).fetchone()
            if missing:
                return False
            db.execute("UPDATE workers SET state='SERVING' WHERE owner=?", (self.owner,))
            return True

    def stop_worker(self) -> None:
        with self._transaction() as db:
            db.execute("UPDATE workers SET state='STOPPED' WHERE owner=?", (self.owner,))

    def drain_worker(self) -> None:
        with self._transaction() as db:
            db.execute("UPDATE workers SET state='DRAINING' WHERE owner=?", (self.owner,))

    def worker_status(self) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT w.*,o.identity FROM workers w JOIN owners o ON o.owner=w.owner"
            ).fetchall()
            prepared = db.execute("SELECT owner,ref FROM worker_bundles").fetchall()
        return [
            {
                "worker_id": row["owner"],
                "backend": row["backend"],
                "state": row["state"],
                "owner_status": self._owner_status(json.loads(row["identity"])),
                "prepared": sorted(
                    item["ref"] for item in prepared if item["owner"] == row["owner"]
                ),
            }
            for row in rows
        ]

    def tag_deployment(self, deployment: str, release: str, expected_workers: int) -> None:
        """Bind the process to one rollout slot incarnation before it starts serving."""
        with self._transaction() as db:
            db.execute(
                "INSERT INTO worker_deployments VALUES(?,?,?,?)",
                (self.owner, deployment, release, expected_workers),
            )

    def deployment_profile(self) -> dict | None:
        with self._connection() as db:
            own = db.execute(
                "SELECT * FROM worker_deployments WHERE owner=?", (self.owner,)
            ).fetchone()
            if own is None:
                return None
            rows = db.execute(
                "SELECT d.*,o.identity,w.state,w.backend FROM worker_deployments d "
                "JOIN owners o ON o.owner=d.owner LEFT JOIN workers w ON w.owner=d.owner "
                "WHERE d.deployment=? ORDER BY d.owner",
                (own["deployment"],),
            ).fetchall()
        stat = self.path.stat()
        return {
            "protocol": 1,
            "id": own["deployment"],
            "release": own["release"],
            "expected_workers": own["expected_workers"],
            "registry": {"host": socket.gethostname(), "device": stat.st_dev, "inode": stat.st_ino},
            "workers": [
                {
                    "id": row["owner"],
                    "release": row["release"],
                    "expected_workers": row["expected_workers"],
                    "state": row["state"],
                    "backend": row["backend"],
                    "owner_status": self._owner_status(json.loads(row["identity"])),
                }
                for row in rows
            ],
        }

    def activate(self, alias: str, reference: str, expected_generation: int) -> dict:
        with self._transaction() as db:
            bundle = self._get(db, reference)
            if bundle["state"] not in ("READY", "ACTIVE", "DRAINING"):
                raise JevError("not_ready", "Only prepared bundles can receive traffic", 409)
            self._adapter_ready(db, reference, bundle["backend"])
            route = db.execute("SELECT * FROM routes WHERE alias=?", (alias,)).fetchone()
            generation = route["generation"] if route else 0
            if expected_generation != generation:
                raise JevError(
                    "generation_conflict", "Active generation changed; refresh before retry", 409
                )
            workers = db.execute(
                "SELECT w.owner,o.identity,p.digest FROM workers w "
                "JOIN owners o ON o.owner=w.owner "
                "LEFT JOIN worker_bundles p ON p.owner=w.owner AND p.ref=? "
                "WHERE w.backend=? AND w.state='SERVING'",
                (reference, bundle["backend"]),
            ).fetchall()
            unprepared = [
                row["owner"]
                for row in workers
                if row["digest"] != bundle["digest"]
                and self._owner_status(json.loads(row["identity"])) != "dead"
            ]
            if unprepared:
                raise JevError(
                    "replicas_not_ready",
                    f"Prepare this version on {len(unprepared)} remaining serving worker(s)",
                    409,
                )
            if route and route["ref"] == reference:
                return {"alias": alias, "bundle": reference, "generation": generation}
            old = route["ref"] if route else None
            generation += 1
            db.execute(
                "INSERT INTO routes VALUES(?,?,?) ON CONFLICT(alias) DO UPDATE SET "
                "ref=excluded.ref,generation=excluded.generation",
                (alias, reference, generation),
            )
            db.execute("UPDATE bundles SET state='ACTIVE' WHERE ref=?", (reference,))
            if old and not db.execute("SELECT 1 FROM routes WHERE ref=?", (old,)).fetchone():
                db.execute("UPDATE bundles SET state='DRAINING' WHERE ref=?", (old,))
            self._event(db, "activate", alias=alias, ref=reference, old=old, generation=generation)
            return {"alias": alias, "bundle": reference, "generation": generation}

    def disable(self, alias: str, expected_generation: int) -> dict:
        with self._transaction() as db:
            route = db.execute("SELECT * FROM routes WHERE alias=?", (alias,)).fetchone()
            if not route:
                raise JevError("route_not_found", "Unknown model alias", 404)
            if expected_generation != route["generation"]:
                raise JevError("generation_conflict", "Active generation changed", 409)
            if route["ref"] is None:
                return {"alias": alias, "bundle": None, "generation": route["generation"]}
            generation = route["generation"] + 1
            db.execute("UPDATE routes SET ref=NULL,generation=? WHERE alias=?", (generation, alias))
            if not db.execute("SELECT 1 FROM routes WHERE ref=?", (route["ref"],)).fetchone():
                db.execute("UPDATE bundles SET state='DRAINING' WHERE ref=?", (route["ref"],))
            self._event(db, "disable", alias=alias, generation=generation)
            return {"alias": alias, "bundle": None, "generation": generation}

    def acquire(
        self,
        alias: str,
        request_id: str,
        reference: str | None,
        backend: str,
        tenant: str = "default",
    ) -> Snapshot:
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM leases WHERE request_id=?", (request_id,)).fetchone():
                raise JevError("duplicate_request", "Request ID is already in flight", 409)
            route = db.execute("SELECT * FROM routes WHERE alias=?", (alias,)).fetchone()
            if not route or not route["ref"]:
                raise JevError(
                    "route_unavailable", "Model alias has no active decision bundle", 503
                )
            if reference is not None and reference != route["ref"]:
                raise JevError(
                    "bundle_not_active", "Requested bundle is not active for this alias", 409
                )
            row = self._get(db, route["ref"])
            if row["state"] != "ACTIVE" or row["backend"] != backend:
                raise JevError("bundle_not_ready", "Bundle is not prepared for this backend", 503)
            self._adapter_ready(db, row["ref"], backend)
            lease = uuid.uuid4().hex
            db.execute(
                "INSERT INTO leases VALUES(?,?,?,?,?)",
                (
                    lease,
                    row["ref"],
                    self.owner,
                    request_id,
                    time.time(),
                ),
            )
            db.execute("INSERT INTO lease_work VALUES(?,?,'inflight')", (lease, "[]"))
            db.execute("INSERT INTO lease_tenants VALUES(?,?)", (lease, tenant))
            return Snapshot(Bundle.model_validate_json(row["manifest"]), route["generation"], lease)

    def release(self, lease_id: str) -> None:
        with self._transaction() as db:
            # A cancellation not yet claimed lost the race to normal request
            # completion. RUNNING commands are settled by their owning worker.
            db.execute(
                "UPDATE cancel_commands SET state='COMPLETED' "
                "WHERE lease_id=? AND owner=? AND state='PENDING'",
                (lease_id, self.owner),
            )
            db.execute("DELETE FROM leases WHERE id=? AND owner=?", (lease_id, self.owner))

    def request_cancel(self, request_id: str, tenant: str, backend: str) -> str | None:
        """Authorize against persisted ownership and address one exact lease."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT l.* FROM leases l JOIN lease_tenants t ON t.lease_id=l.id "
                "JOIN bundles b ON b.ref=l.ref "
                "WHERE l.request_id=? AND t.tenant=? AND b.backend=?",
                (request_id, tenant, backend),
            ).fetchone()
            if row is None:
                return None
            previous = db.execute(
                "SELECT id FROM cancel_commands WHERE lease_id=?", (row["id"],)
            ).fetchone()
            if previous:
                return previous["id"]
            command = uuid.uuid4().hex
            db.execute(
                "INSERT INTO cancel_commands VALUES(?,?,?,?,?,'PENDING',?)",
                (command, row["id"], row["owner"], request_id, tenant, time.time()),
            )
            # Retain pending/uncertain work indefinitely; terminal control
            # responses only need a bounded observation window.
            db.execute(
                "DELETE FROM cancel_commands WHERE state IN ('CONFIRMED','COMPLETED') "
                "AND created<?",
                (time.time() - 3600,),
            )
            return command

    def pending_cancel_commands(self) -> list[dict]:
        with self._connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM cancel_commands WHERE owner=? AND state='PENDING'",
                    (self.owner,),
                )
            ]

    def claim_cancel(self, command: str) -> bool:
        with self._transaction() as db:
            return (
                db.execute(
                    "UPDATE cancel_commands SET state='RUNNING' "
                    "WHERE id=? AND owner=? AND state='PENDING'",
                    (command, self.owner),
                ).rowcount
                == 1
            )

    def finish_cancel(self, command: str, state: str) -> None:
        if state not in {"CONFIRMED", "COMPLETED", "UNCONFIRMED"}:
            raise ValueError("Invalid terminal cancellation state")
        with self._transaction() as db:
            db.execute(
                "UPDATE cancel_commands SET state=? WHERE id=? AND owner=? AND state='RUNNING'",
                (state, command, self.owner),
            )

    def cancel_status(self, command: str) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT state FROM cancel_commands WHERE id=?", (command,)).fetchone()
            return row["state"] if row else None

    def has_lease(self, lease_id: str) -> bool:
        with self._connection() as db:
            return db.execute("SELECT 1 FROM leases WHERE id=?", (lease_id,)).fetchone() is not None

    def record_branches(self, lease_id: str, branches: list[str]) -> None:
        """Persist engine IDs before dispatch, including canaries and queued work."""
        with self._transaction() as db:
            self._record_branches(db, lease_id, branches)

    def _record_branches(self, db, lease_id: str, branches: list[str]) -> None:
        """Join the caller's transaction; the caller commits before dispatch."""
        row = db.execute(
            "SELECT w.* FROM lease_work w JOIN leases l ON l.id=w.lease_id "
            "WHERE l.id=? AND l.owner=?",
            (lease_id, self.owner),
        ).fetchone()
        if not row or row["phase"] != "inflight":
            raise JevError("lease_not_owned", "Cannot dispatch against an unowned lease", 409)
        combined = sorted(set(json.loads(row["branches"])) | set(branches))
        db.execute(
            "UPDATE lease_work SET branches=? WHERE lease_id=?",
            (json.dumps(combined), lease_id),
        )

    def pin_revalidation(self, reference: str, request_id: str, backend: str) -> tuple[Bundle, str]:
        """Probe an existing version without changing another worker's active route."""
        with self._transaction() as db:
            row = self._get(db, reference)
            self._adapter_ready(db, reference, backend)
            if row["state"] not in {"READY", "ACTIVE", "DRAINING"} or row["backend"] != backend:
                raise JevError("not_ready", "Bundle is not prepared for this engine target", 409)
            lease = uuid.uuid4().hex
            db.execute(
                "INSERT INTO leases VALUES(?,?,?,?,?)",
                (lease, reference, self.owner, request_id, time.time()),
            )
            db.execute("INSERT INTO lease_work VALUES(?,?,'inflight')", (lease, "[]"))
            return Bundle.model_validate_json(row["manifest"]), lease

    def mark_abort_pending(self, lease_id: str, branches: set[str]) -> None:
        with self._transaction() as db:
            if not db.execute(
                "SELECT 1 FROM leases WHERE id=? AND owner=?", (lease_id, self.owner)
            ).fetchone():
                raise JevError("lease_not_owned", "Cannot mark an unowned lease", 409)
            db.execute(
                "UPDATE lease_work SET branches=?,phase='abort_pending' WHERE lease_id=?",
                (json.dumps(sorted(branches)), lease_id),
            )
            self._event(db, "abort_pending", lease_id=lease_id, engine_request_ids=sorted(branches))

    def recovery_candidates(self) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT l.*,w.branches,w.phase,o.identity,b.backend FROM leases l "
                "LEFT JOIN lease_work w ON w.lease_id=l.id "
                "LEFT JOIN owners o ON o.owner=l.owner "
                "JOIN bundles b ON b.ref=l.ref"
            ).fetchall()
        result = []
        for row in rows:
            owner_status = (
                self._owner_status(json.loads(row["identity"])) if row["identity"] else "unknown"
            )
            result.append(
                {
                    "lease_id": row["id"],
                    "request_id": row["request_id"],
                    "owner": row["owner"],
                    "owner_status": owner_status,
                    "phase": row["phase"],
                    "backend": row["backend"],
                    "reference": row["ref"],
                    "engine_request_ids": json.loads(row["branches"] or "[]"),
                    "recoverable": row["phase"] == "abort_pending"
                    or (row["phase"] == "inflight" and owner_status == "dead"),
                }
            )
        return result

    def recovery_snapshot(self, request_id: str, backend: str) -> dict | None:
        row = next((r for r in self.recovery_candidates() if r["request_id"] == request_id), None)
        if row is None:
            return None
        if row["backend"] != backend:
            raise JevError(
                "recovery_backend_mismatch", "Recovery requires the original engine target", 409
            )
        if not row["recoverable"]:
            raise JevError(
                "recovery_not_confirmed",
                "Owner is alive, unverifiable, or lacks a dispatch journal",
                409,
            )
        return row

    def release_recovered(self, snapshot: dict) -> None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT l.*,w.phase,w.branches FROM leases l JOIN lease_work w ON w.lease_id=l.id "
                "WHERE l.id=?",
                (snapshot["lease_id"],),
            ).fetchone()
            if row is None:
                return
            if (
                row["owner"] != snapshot["owner"]
                or row["phase"] != snapshot["phase"]
                or json.loads(row["branches"]) != snapshot["engine_request_ids"]
            ):
                raise JevError("recovery_conflict", "Lease changed during recovery", 409)
            db.execute("DELETE FROM leases WHERE id=?", (row["id"],))
            db.execute(
                "UPDATE cancel_commands SET state='CONFIRMED' WHERE lease_id=?",
                (row["id"],),
            )
            db.execute(
                "UPDATE bundles SET state='FAILED',error='preparation owner exited' "
                "WHERE ref=? AND state='PREPARING'",
                (row["ref"],),
            )
            self._event(db, "recovered", lease_id=row["id"], request_id=row["request_id"])

    def retire(self, reference: str) -> dict:
        with self._transaction() as db:
            row = self._get(db, reference)
            if db.execute("SELECT 1 FROM routes WHERE ref=?", (reference,)).fetchone():
                raise JevError(
                    "bundle_active", "Disable or replace active routes before retiring", 409
                )
            if row["state"] == "PREPARING":
                raise JevError("bundle_preparing", "Cannot retire during preparation", 409)
            count = db.execute("SELECT COUNT(*) FROM leases WHERE ref=?", (reference,)).fetchone()[
                0
            ]
            if count:
                raise JevError(
                    "bundle_in_use", f"Bundle still has {count} in-flight request(s)", 409
                )
            db.execute("UPDATE bundles SET state='RETIRED' WHERE ref=?", (reference,))
            self._event(db, "retire", ref=reference)
        return self.inspect(reference)

    def inspect(self, reference: str) -> dict:
        with self._connection() as db:
            row = self._get(db, reference)
            count = db.execute("SELECT COUNT(*) FROM leases WHERE ref=?", (reference,)).fetchone()[
                0
            ]
            return {
                "reference": row["ref"],
                "digest": row["digest"],
                "state": row["state"],
                "backend": row["backend"],
                "error": row["error"],
                "inflight": count,
            }

    @staticmethod
    def _adapter_ready(db: sqlite3.Connection, reference: str, backend: str) -> None:
        row = db.execute(
            "SELECT a.state,a.backend FROM bundle_adapters b JOIN adapters a "
            "ON a.ref=b.adapter_ref WHERE b.ref=?",
            (reference,),
        ).fetchone()
        if row is not None and (row["state"] != "READY" or row["backend"] != backend):
            raise JevError("adapter_not_ready", "Bundle adapter is not loaded for this engine", 503)
        if row is None:
            bundle = db.execute("SELECT manifest FROM bundles WHERE ref=?", (reference,)).fetchone()
            if bundle and json.loads(bundle["manifest"])["model"].get("adapter_id"):
                raise JevError(
                    "adapter_not_registered", "Legacy adapter bundle needs registration", 409
                )

    def start_adapter_session(self, backend: str) -> list[str]:
        """Single frontend only: quarantine previous engine state on restart.

        GPU residency cannot be inferred from SQLite. Pause affected aliases,
        preserve leases, and require explicit unload/reload and new canaries.
        A verified dead or gracefully stopped coordinator is required.
        """
        with self._transaction() as db:
            for worker in db.execute(
                "SELECT w.*,o.identity FROM workers w JOIN owners o ON o.owner=w.owner "
                "WHERE backend=? AND w.owner!=? AND state!='STOPPED'",
                (backend, self.owner),
            ):
                if self._owner_status(json.loads(worker["identity"])) != "dead":
                    raise JevError(
                        "adapter_coordinator_active",
                        "Managed LoRA requires one exclusive API worker",
                        409,
                    )
            affected = [
                row["ref"]
                for row in db.execute(
                    "SELECT ref FROM adapters WHERE backend=? "
                    "AND state IN ('READY','LOADING','UNLOADING')",
                    (backend,),
                )
            ]
            for reference in affected:
                db.execute(
                    "UPDATE adapters SET state='UNKNOWN',operation=NULL,owner=NULL,"
                    "error='engine_session_changed' WHERE ref=?",
                    (reference,),
                )
                db.execute(
                    "UPDATE routes SET ref=NULL,generation=generation+1 WHERE ref IN "
                    "(SELECT ref FROM bundle_adapters WHERE adapter_ref=?)",
                    (reference,),
                )
                db.execute(
                    "DELETE FROM worker_bundles WHERE ref IN "
                    "(SELECT ref FROM bundle_adapters WHERE adapter_ref=?)",
                    (reference,),
                )
                db.execute(
                    "UPDATE bundles SET state='VALIDATED' WHERE state IN "
                    "('PREPARING','READY','ACTIVE','DRAINING') AND ref IN "
                    "(SELECT ref FROM bundle_adapters WHERE adapter_ref=?)",
                    (reference,),
                )
                self._event(db, "adapter_session_quarantine", ref=reference)
            return affected

    def register_adapter(self, artifact: AdapterArtifact, backend: str) -> dict:
        with self._transaction() as db:
            row = db.execute("SELECT * FROM adapters WHERE ref=?", (artifact.reference,)).fetchone()
            if row is not None:
                if row["manifest"] != artifact.model_dump_json() or row["backend"] != backend:
                    raise JevError(
                        "immutable_adapter",
                        "An adapter reference cannot be overwritten or rebound",
                        409,
                    )
            else:
                # vLLM's GPU request state stores LoRA IDs in an int32 array.
                # Allocate in its positive range and check collisions inside
                # the same write transaction; UUID truncation alone is not a
                # uniqueness guarantee. IDs are never recycled by this registry.
                for _ in range(16):
                    engine_id = (uuid.uuid4().int & ((1 << 31) - 1)) or 1
                    if not db.execute(
                        "SELECT 1 FROM adapters WHERE backend=? "
                        "AND json_extract(binding,'$.engine_id')=?",
                        (backend, engine_id),
                    ).fetchone():
                        break
                else:
                    raise JevError(
                        "adapter_id_exhausted",
                        "Could not allocate a distinct engine adapter ID",
                        503,
                    )
                binding = AdapterBinding(
                    artifact=artifact,
                    engine_name="jev-lora-" + uuid.uuid4().hex,
                    engine_id=engine_id,
                )
                db.execute(
                    "INSERT INTO adapters VALUES(?,?,?,?,?,'REGISTERED',NULL,NULL,NULL,?)",
                    (
                        artifact.reference,
                        artifact.model_dump_json(),
                        artifact.revision,
                        binding.model_dump_json(),
                        backend,
                        time.time(),
                    ),
                )
                self._event(db, "adapter_register", ref=artifact.reference, backend=backend)
        return self.inspect_adapter(artifact.reference)

    def inspect_adapter(self, reference: str) -> dict:
        with self._connection() as db:
            row = db.execute("SELECT * FROM adapters WHERE ref=?", (reference,)).fetchone()
            if row is None:
                raise JevError("adapter_not_found", "Unknown immutable adapter", 404)
            bundles = [
                item["ref"]
                for item in db.execute(
                    "SELECT ref FROM bundle_adapters WHERE adapter_ref=?", (reference,)
                )
            ]
            return {
                "reference": reference,
                "state": row["state"],
                "backend": row["backend"],
                "binding": json.loads(row["binding"]),
                "operation": row["operation"],
                "owner": row["owner"],
                "error": row["error"],
                "bundles": bundles,
            }

    def list_adapters(self) -> list[dict]:
        with self._connection() as db:
            references = [row["ref"] for row in db.execute("SELECT ref FROM adapters ORDER BY ref")]
        return [self.inspect_adapter(reference) for reference in references]

    def adapter_binding(self, reference: str, backend: str) -> AdapterBinding:
        row = self.inspect_adapter(reference)
        if row["state"] != "READY" or row["backend"] != backend:
            raise JevError("adapter_not_ready", "Adapter is not loaded for this engine", 503)
        return AdapterBinding.model_validate(row["binding"])

    def begin_adapter_operation(
        self,
        reference: str,
        backend: str,
        action: str,
        recover: bool = False,
    ) -> tuple[str, AdapterBinding]:
        if action not in {"load", "unload"} or (recover and action != "unload"):
            raise ValueError("Invalid adapter lifecycle operation")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM adapters WHERE ref=?", (reference,)).fetchone()
            if row is None:
                raise JevError("adapter_not_found", "Unknown immutable adapter", 404)
            if row["backend"] != backend:
                raise JevError("adapter_backend_mismatch", "Adapter belongs to another engine", 409)
            if row["state"] in {"LOADING", "UNLOADING"}:
                owner = db.execute(
                    "SELECT identity FROM owners WHERE owner=?", (row["owner"],)
                ).fetchone()
                if (
                    not recover
                    or not owner
                    or self._owner_status(json.loads(owner["identity"])) != "dead"
                ):
                    raise JevError(
                        "adapter_operation_active",
                        "An adapter operation has a live or unverified owner",
                        409,
                    )
            elif action == "load" and row["state"] not in {"REGISTERED", "UNLOADED"}:
                raise JevError(
                    "adapter_state", "Reconcile an uncertain adapter before loading it", 409
                )
            elif action == "unload" and row["state"] not in {
                "READY",
                "UNKNOWN",
                "REGISTERED",
                "UNLOADED",
            }:
                raise JevError(
                    "adapter_state", "Adapter cannot be unloaded from its current state", 409
                )
            if action == "unload":
                active = db.execute(
                    "SELECT 1 FROM routes r JOIN bundle_adapters b ON r.ref=b.ref "
                    "WHERE b.adapter_ref=? LIMIT 1",
                    (reference,),
                ).fetchone()
                leased = db.execute(
                    "SELECT 1 FROM leases l JOIN bundle_adapters b ON l.ref=b.ref "
                    "WHERE b.adapter_ref=? LIMIT 1",
                    (reference,),
                ).fetchone()
                if active or leased:
                    raise JevError(
                        "adapter_in_use",
                        "Disable every adapter route and drain all leases before unloading",
                        409,
                    )
            operation = uuid.uuid4().hex
            db.execute(
                "UPDATE adapters SET state=?,operation=?,owner=?,error=NULL WHERE ref=?",
                ("LOADING" if action == "load" else "UNLOADING", operation, self.owner, reference),
            )
            self._event(db, "adapter_" + action, ref=reference, operation=operation)
            return operation, AdapterBinding.model_validate_json(row["binding"])

    def finish_adapter_operation(
        self, reference: str, operation: str, error: str | None = None
    ) -> None:
        with self._transaction() as db:
            row = db.execute("SELECT * FROM adapters WHERE ref=?", (reference,)).fetchone()
            if not row or row["operation"] != operation or row["owner"] != self.owner:
                raise JevError(
                    "adapter_operation_conflict", "Adapter operation ownership changed", 409
                )
            if row["state"] not in {"LOADING", "UNLOADING"}:
                raise JevError(
                    "adapter_operation_conflict", "Adapter operation already settled", 409
                )
            state = "UNKNOWN" if error else "READY" if row["state"] == "LOADING" else "UNLOADED"
            db.execute(
                "UPDATE adapters SET state=?,error=?,operation=NULL,owner=NULL WHERE ref=?",
                (state, error, reference),
            )
            if state == "UNLOADED":
                db.execute(
                    "DELETE FROM worker_bundles WHERE ref IN "
                    "(SELECT ref FROM bundle_adapters WHERE adapter_ref=?)",
                    (reference,),
                )
                db.execute(
                    "UPDATE bundles SET state='VALIDATED' WHERE state IN ('READY','DRAINING') "
                    "AND ref IN (SELECT ref FROM bundle_adapters WHERE adapter_ref=?)",
                    (reference,),
                )
            self._event(db, "adapter_settled", ref=reference, operation=operation, state=state)

    def list(self) -> dict:
        with self._connection() as db:
            return {
                "bundles": [
                    dict(row)
                    for row in db.execute(
                        "SELECT ref,digest,state,backend,error,created FROM bundles ORDER BY ref"
                    )
                ],
                "routes": [dict(row) for row in db.execute("SELECT * FROM routes ORDER BY alias")],
                "leases": [
                    dict(row) for row in db.execute("SELECT * FROM leases ORDER BY created")
                ],
            }
