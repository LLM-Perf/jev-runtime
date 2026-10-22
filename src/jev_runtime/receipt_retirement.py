"""Retire completed TokenSpeed receipts without reopening their request IDs.

Pending rows stay in place. Retirement runs only on a stopped registry: every
owner is dead or explicitly fenced, and no lease, journal or cancel command
remains. Completed IDs move into ``retired_receipts`` in the same SQLite
transaction as the registry audit event, so a crash cannot drop the idempotency
record without also dropping the audit. Retired IDs still reject ``reserve``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid

from jev_runtime.registry_backup import blockers, canonical, connect

RETIRE_ACK = "retire-completed-unreferenced-receipts"
_NAMESPACE = re.compile(r"^[\w.:-]{1,128}$")


def retire_completed_receipts(source, namespace: str, limit: int, acknowledgement: str) -> dict:
    if acknowledgement != RETIRE_ACK:
        raise ValueError(f"Receipt retirement requires acknowledgement {RETIRE_ACK}")
    if not _NAMESPACE.fullmatch(namespace):
        raise ValueError("Invalid receipt namespace")
    if type(limit) is not int or not 1 <= limit <= 1024:
        raise ValueError("Retirement limit must be an integer from 1 to 1024")
    receipts = source.with_suffix(".tokenspeed-receipts.db")
    if source.is_symlink() or receipts.is_symlink() or not receipts.is_file():
        raise ValueError("Registry and receipt database must be existing regular files")
    db = connect(source, writable=True)
    try:
        db.execute("ATTACH DATABASE ? AS native", (str(receipts.resolve()),))
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("UPDATE native.receipts SET state=state WHERE 0")
            outstanding = blockers(db)
            if outstanding:
                raise ValueError("Registry is not stopped/drained: " + json.dumps(outstanding))
            db.execute(
                "CREATE TABLE IF NOT EXISTS native.retired_receipts ("
                "namespace TEXT NOT NULL, request_id TEXT NOT NULL, "
                "operation_id TEXT NOT NULL, PRIMARY KEY(namespace, request_id))"
            )
            rows = db.execute(
                "SELECT request_id FROM native.receipts WHERE namespace=? AND state='completed' "
                "AND request_id NOT IN (SELECT request_id FROM leases) "
                "AND request_id NOT IN (SELECT request_id FROM raw_work) "
                "AND request_id NOT IN (SELECT request_id FROM cancel_commands) "
                "ORDER BY request_id LIMIT ?",
                (namespace, limit),
            ).fetchall()
            if not rows:
                raise ValueError("No completed unreferenced receipts for this namespace")
            operation = uuid.uuid4().hex
            ids = [row[0] for row in rows]
            db.executemany(
                "INSERT INTO native.retired_receipts VALUES(?,?,?)",
                [(namespace, request_id, operation) for request_id in ids],
            )
            deleted = db.execute(
                "DELETE FROM native.receipts WHERE namespace=? AND state='completed' "
                "AND request_id IN (" + ",".join("?" * len(ids)) + ")",
                (namespace, *ids),
            ).rowcount
            if deleted != len(ids):
                raise ValueError("Receipt retirement changed under the writer lock")
            remaining = db.execute(
                "SELECT count(*) FROM native.receipts WHERE namespace=? AND state='completed'",
                (namespace,),
            ).fetchone()[0]
            pending = db.execute(
                "SELECT count(*) FROM native.receipts WHERE namespace=? AND state='pending'",
                (namespace,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO events(ts,action,details) VALUES(?,?,?)",
                (
                    time.time(),
                    "receipts_retired",
                    json.dumps(
                        {
                            "operation_id": operation,
                            "namespace": namespace,
                            "retired": len(ids),
                            "ids_sha256": hashlib.sha256(canonical(ids)).hexdigest(),
                        },
                        sort_keys=True,
                    ),
                ),
            )
            db.commit()
            return {
                "operation_id": operation,
                "namespace": namespace,
                "retired": len(ids),
                "remaining_completed": remaining,
                "pending": pending,
            }
        except BaseException:
            db.rollback()
            raise
    finally:
        db.close()
