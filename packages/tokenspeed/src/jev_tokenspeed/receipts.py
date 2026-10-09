"""Durable native completion evidence; pending rows never imply cancellation."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from jev_runtime.errors import JevError


class CompletionReceipts:
    def __init__(self, path: Path, namespace: str):
        self.path = path.resolve()
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ValueError("Unsupported TokenSpeed receipt schema")
            if version == 0:
                tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if tables:
                    raise ValueError("Receipt path must be a new database or a receipt store")
            else:
                columns = db.execute("PRAGMA table_info(receipts)").fetchall()
                if [(row[1], row[2], row[3], row[5]) for row in columns] != [
                    ("namespace", "TEXT", 1, 1),
                    ("request_id", "TEXT", 1, 2),
                    ("state", "TEXT", 1, 0),
                ]:
                    raise ValueError("Receipt path is not a compatible receipt store")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS receipts ("
                "namespace TEXT NOT NULL, request_id TEXT NOT NULL, "
                "state TEXT NOT NULL CHECK(state IN ('pending','completed')), "
                "PRIMARY KEY(namespace, request_id))"
            )
            db.execute("PRAGMA user_version=1")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def reserve(self, request_id: str) -> None:
        try:
            with closing(self.connect()) as db, db:
                db.execute(
                    "INSERT INTO receipts VALUES (?, ?, 'pending')",
                    (self.namespace, request_id),
                )
        except sqlite3.IntegrityError as exc:
            raise JevError(
                "duplicate_request", "Native scoring ID was already dispatched", 409
            ) from exc

    def complete(self, request_id: str) -> None:
        with closing(self.connect()) as db, db:
            count = db.execute(
                "UPDATE receipts SET state='completed' WHERE namespace=? AND request_id=?",
                (self.namespace, request_id),
            ).rowcount
            if count != 1:
                raise ValueError("Cannot confirm an unreserved TokenSpeed request")

    def completed(self, request_id: str) -> bool:
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT state FROM receipts WHERE namespace=? AND request_id=?",
                (self.namespace, request_id),
            ).fetchone()
            return row == ("completed",)
