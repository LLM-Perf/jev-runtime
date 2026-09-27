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

from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle


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
            db.executescript("""
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
                CREATE TABLE IF NOT EXISTS lease_work (
                    lease_id TEXT PRIMARY KEY REFERENCES leases(id) ON DELETE CASCADE,
                    branches TEXT NOT NULL, phase TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lease_tenants (
                    lease_id TEXT PRIMARY KEY REFERENCES leases(id) ON DELETE CASCADE,
                    tenant TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cancel_commands (
                    id TEXT PRIMARY KEY, lease_id TEXT NOT NULL UNIQUE,
                    owner TEXT NOT NULL, request_id TEXT NOT NULL,
                    tenant TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cancel_commands_by_owner
                    ON cancel_commands(owner,state);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                    action TEXT NOT NULL, details TEXT NOT NULL
                );
            """)
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
        return self.inspect(bundle.reference)

    def begin_prepare(self, reference: str, backend: str, request_id: str) -> tuple[Bundle, str]:
        with self._transaction() as db:
            row = self._get(db, reference)
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

    def start_worker(self, backend: str) -> None:
        with self._transaction() as db:
            db.execute(
                "INSERT INTO workers VALUES(?,?,'STARTING') ON CONFLICT(owner) DO UPDATE SET "
                "backend=excluded.backend,state='STARTING'",
                (self.owner, backend),
            )
            db.execute("DELETE FROM worker_bundles WHERE owner=?", (self.owner,))

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

    def activate(self, alias: str, reference: str, expected_generation: int) -> dict:
        with self._transaction() as db:
            bundle = self._get(db, reference)
            if bundle["state"] not in ("READY", "ACTIVE", "DRAINING"):
                raise JevError("not_ready", "Only prepared bundles can receive traffic", 409)
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
