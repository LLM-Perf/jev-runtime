"""Durable, single-host backend quiescence; routes are never rewritten to stop traffic."""

from __future__ import annotations

import json
import uuid

from jev_runtime.errors import JevError


class QuiescenceRegistry:
    """Registry mixin. Dispatch gates join the caller's existing writer transaction."""

    def _require_backend_open(self, db, backend, *, starting=False):
        row = db.execute(
            "SELECT (SELECT state FROM backend_controls WHERE backend=?) AS gate, "
            "(SELECT state FROM workers WHERE owner=?) AS worker",
            (backend, self.owner),
        ).fetchone()
        if (row["gate"] and row["gate"] != "OPEN") or (
            not starting and row["worker"] and row["worker"] not in {"STARTING", "SERVING"}
        ):
            raise JevError("backend_quiescing", "Backend is draining; new work is disabled", 503)

    def require_backend_open(self, backend):
        with self._connection() as db:
            self._require_backend_open(db, backend)

    def backend_control(self, backend):
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM backend_controls WHERE backend=?", (backend,)
            ).fetchone()
            return dict(row) if row else {"backend": backend, "generation": 0, "state": "OPEN"}

    def begin_quiesce(self, backend, expected_generation):
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM backend_controls WHERE backend=?", (backend,)
            ).fetchone()
            generation = row["generation"] if row else 0
            if generation != expected_generation:
                raise JevError("generation_conflict", "Refresh the backend control generation", 409)
            if row and row["state"] == "QUIESCING":
                return
            for worker in db.execute(
                "SELECT w.state,o.identity,q.protocol FROM workers w "
                "JOIN owners o ON o.owner=w.owner "
                "LEFT JOIN worker_quiescence q ON q.owner=w.owner WHERE w.backend=?",
                (backend,),
            ):
                if (
                    worker["state"] != "STOPPED"
                    and worker["protocol"] != 1
                    and self._owner_status(json.loads(worker["identity"])) != "dead"
                ):
                    raise JevError(
                        "quiescence_protocol_mismatch",
                        "Stop legacy API workers before using backend quiescence",
                        409,
                    )
            db.execute(
                "INSERT INTO backend_controls VALUES(?,?,'QUIESCING') "
                "ON CONFLICT(backend) DO UPDATE SET "
                "generation=excluded.generation,state='QUIESCING'",
                (backend, generation + 1),
            )
            db.execute(
                "UPDATE workers SET state='DRAINING' WHERE backend=? "
                "AND state IN ('STARTING','SERVING')",
                (backend,),
            )
            self._event(db, "backend_quiesce", backend=backend, generation=generation + 1)

    @staticmethod
    def _outstanding(db, backend):
        return {
            "leases": db.execute(
                "SELECT count(*) FROM leases l JOIN bundles b ON b.ref=l.ref WHERE b.backend=?",
                (backend,),
            ).fetchone()[0],
            "raw_work": db.execute(
                "SELECT count(*) FROM raw_work WHERE backend=?", (backend,)
            ).fetchone()[0],
            "adapter_operations": db.execute(
                "SELECT count(*) FROM adapters WHERE backend=? "
                "AND state IN ('LOADING','UNLOADING','UNKNOWN')",
                (backend,),
            ).fetchone()[0],
            "recovery_operations": db.execute(
                "SELECT count(*) FROM recovery_claims WHERE backend=?", (backend,)
            ).fetchone()[0],
        }

    def acknowledge_quiescence(self, backend):
        """Called only after this worker's local tasks and health loop have finished."""
        with self._transaction() as db:
            control = db.execute(
                "SELECT state FROM backend_controls WHERE backend=?", (backend,)
            ).fetchone()
            if not control or control["state"] != "QUIESCING":
                return
            if db.execute("SELECT 1 FROM leases WHERE owner=?", (self.owner,)).fetchone():
                return
            if db.execute("SELECT 1 FROM raw_work WHERE owner=?", (self.owner,)).fetchone():
                return
            if db.execute("SELECT 1 FROM recovery_claims WHERE owner=?", (self.owner,)).fetchone():
                return
            if db.execute(
                "SELECT 1 FROM adapters WHERE owner=? "
                "AND state IN ('LOADING','UNLOADING','UNKNOWN')",
                (self.owner,),
            ).fetchone():
                return
            db.execute(
                "UPDATE workers SET state='QUIESCED' "
                "WHERE owner=? AND backend=? AND state='DRAINING'",
                (self.owner, backend),
            )

    def quiescence_status(self, backend):
        with self._connection() as db:
            control = self.backend_control(backend)
            outstanding = self._outstanding(db, backend)
            workers = [row for row in self.worker_status() if row["backend"] == backend]
            waiting = [
                row["worker_id"]
                for row in workers
                if row["state"] not in {"QUIESCED", "STOPPED"} and row["owner_status"] != "dead"
            ]
            return {
                **control,
                "scope": "single_host_registry_backend_jev_work",
                "drained": control["state"] == "QUIESCING"
                and not waiting
                and not any(outstanding.values()),
                "outstanding": outstanding,
                "waiting_workers": waiting,
                "workers": workers,
            }

    def resume_backend(self, backend, expected_generation):
        """Offline only: require every previous worker stopped or verifiably dead."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM backend_controls WHERE backend=?", (backend,)
            ).fetchone()
            if not row or row["generation"] != expected_generation:
                raise JevError("generation_conflict", "Refresh the backend control generation", 409)
            if row["state"] != "QUIESCING":
                raise JevError("invalid_state", "Backend is not quiescing", 409)
            if any(self._outstanding(db, backend).values()):
                raise JevError("quiescence_pending", "Recover outstanding engine work first", 409)
            for worker in db.execute(
                "SELECT w.state,o.identity FROM workers w JOIN owners o ON o.owner=w.owner "
                "WHERE w.backend=?",
                (backend,),
            ):
                if (
                    worker["state"] != "STOPPED"
                    and self._owner_status(json.loads(worker["identity"])) != "dead"
                ):
                    raise JevError(
                        "workers_not_stopped", "Stop all previous API workers first", 409
                    )
            db.execute(
                "UPDATE backend_controls SET state='OPEN',generation=generation+1 WHERE backend=?",
                (backend,),
            )
            self._event(db, "backend_resume", backend=backend, generation=expected_generation + 1)

    def begin_raw_work(self, backend, request_id):
        with self._transaction() as db:
            self._require_backend_open(db, backend)
            if db.execute(
                "SELECT 1 FROM raw_work WHERE backend=? AND request_id=?", (backend, request_id)
            ).fetchone():
                raise JevError("duplicate_request", "Raw request ID is already in flight", 409)
            work_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO raw_work VALUES(?,?,?,?,'inflight')",
                (work_id, backend, self.owner, request_id),
            )
            return work_id

    def finish_raw_work(self, work_id, *, unconfirmed=False):
        with self._transaction() as db:
            if unconfirmed:
                db.execute(
                    "UPDATE raw_work SET phase='abort_pending' WHERE id=? AND owner=?",
                    (work_id, self.owner),
                )
            else:
                db.execute("DELETE FROM raw_work WHERE id=? AND owner=?", (work_id, self.owner))

    def raw_recovery_candidates(self, backend):
        with self._connection() as db:
            rows = db.execute(
                "SELECT r.*,o.identity FROM raw_work r JOIN owners o ON o.owner=r.owner "
                "WHERE r.backend=?",
                (backend,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            status = self._owner_status(json.loads(item.pop("identity")))
            result.append(
                {
                    **item,
                    "owner_status": status,
                    "recoverable": row["phase"] == "abort_pending" or status == "dead",
                }
            )
        return result

    def raw_recovery_snapshot(self, backend, work_id):
        row = next((r for r in self.raw_recovery_candidates(backend) if r["id"] == work_id), None)
        if row and not row["recoverable"]:
            raise JevError("recovery_not_confirmed", "Raw owner is live or unverifiable", 409)
        return row

    def release_raw_recovered(self, snapshot):
        with self._transaction() as db:
            db.execute(
                "DELETE FROM raw_work WHERE id=? AND owner=? AND backend=? AND request_id=? "
                "AND phase=?",
                tuple(snapshot[key] for key in ("id", "owner", "backend", "request_id", "phase")),
            )

    def claim_recovery(self, backend, resource):
        """Serialize aborts and keep resume blocked until the entire operation is finished."""
        with self._transaction() as db:
            row = db.execute(
                "SELECT o.identity FROM recovery_claims r JOIN owners o ON o.owner=r.owner "
                "WHERE r.resource=?",
                (resource,),
            ).fetchone()
            if row and self._owner_status(json.loads(row["identity"])) != "dead":
                raise JevError(
                    "recovery_in_progress", "A recovery operation already owns this work", 409
                )
            db.execute(
                "INSERT INTO recovery_claims VALUES(?,?,?) ON CONFLICT(resource) DO UPDATE SET "
                "backend=excluded.backend,owner=excluded.owner",
                (resource, backend, self.owner),
            )

    def finish_recovery(self, resource):
        with self._transaction() as db:
            db.execute(
                "DELETE FROM recovery_claims WHERE resource=? AND owner=?", (resource, self.owner)
            )

    def pending_recoveries(self, backend):
        with self._connection() as db:
            return [
                dict(row)
                for row in db.execute("SELECT * FROM recovery_claims WHERE backend=?", (backend,))
            ]
