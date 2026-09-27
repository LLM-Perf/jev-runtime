from __future__ import annotations

import json
import sqlite3
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
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                    action TEXT NOT NULL, details TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

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

    def begin_prepare(self, reference: str, backend: str) -> Bundle:
        with self._transaction() as db:
            row = self._get(db, reference)
            if row["state"] not in ("VALIDATED", "FAILED", "RETIRED"):
                raise JevError("invalid_state", f"Cannot prepare a {row['state']} bundle", 409)
            db.execute(
                "UPDATE bundles SET state='PREPARING',backend=?,error=NULL WHERE ref=?",
                (
                    backend,
                    reference,
                ),
            )
            self._event(db, "prepare", ref=reference, backend=backend)
            return Bundle.model_validate_json(row["manifest"])

    def finish_prepare(self, reference: str, error: str | None = None) -> None:
        with self._transaction() as db:
            row = self._get(db, reference)
            if row["state"] != "PREPARING":
                raise JevError("invalid_state", "Prepare was superseded", 409)
            state = "FAILED" if error else "READY"
            db.execute("UPDATE bundles SET state=?,error=? WHERE ref=?", (state, error, reference))
            self._event(db, "prepare_finished", ref=reference, state=state)

    @staticmethod
    def _get(db: sqlite3.Connection, reference: str):
        row = db.execute("SELECT * FROM bundles WHERE ref=?", (reference,)).fetchone()
        if not row:
            raise JevError("bundle_not_found", f"Unknown bundle {reference}", 404)
        return row

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

    def acquire(self, alias: str, request_id: str, reference: str | None, backend: str) -> Snapshot:
        with self._transaction() as db:
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
            return Snapshot(Bundle.model_validate_json(row["manifest"]), route["generation"], lease)

    def release(self, lease_id: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM leases WHERE id=? AND owner=?", (lease_id, self.owner))

    def pin_preparation(self, reference: str, request_id: str, backend: str) -> str:
        """Canary inference must block retirement just like ordinary traffic."""
        with self._transaction() as db:
            row = self._get(db, reference)
            if row["state"] != "PREPARING" or row["backend"] != backend:
                raise JevError("invalid_state", "Preparation no longer owns this bundle", 409)
            lease = uuid.uuid4().hex
            db.execute(
                "INSERT INTO leases VALUES(?,?,?,?,?)",
                (lease, reference, self.owner, request_id, time.time()),
            )
            return lease

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
