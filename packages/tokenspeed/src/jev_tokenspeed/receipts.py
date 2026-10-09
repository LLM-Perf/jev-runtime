"""Bounded terminal-completion receipts for the pinned TokenSpeed readout.

A receipt only asserts one thing: a verified terminal engine output for this
exact request ID was observed, so no in-flight work remains. Receipts never
prove that an unknown ID completed, and a missing receipt never confirms an
abort. The durable variant survives a plugin restart, which lets explicit
administrative recovery confirm pre-crash completions instead of retaining
every orphan journal forever.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

from jev_runtime.errors import JevError

BOUND = 4096


class MemoryReceipts:
    """In-process receipts for embedded hosts without a durability requirement."""

    def __init__(self, bound: int = BOUND):
        self._bound = bound
        self._completed: OrderedDict[str, None] = OrderedDict()

    async def record(self, request_id: str) -> None:
        self._completed[request_id] = None
        while len(self._completed) > self._bound:
            self._completed.popitem(last=False)

    async def contains(self, request_id: str) -> bool:
        return request_id in self._completed

    def close(self) -> None:
        pass


class DurableReceipts:
    """SQLite-backed receipts; one short FULL-synchronous commit per completion."""

    def __init__(self, path: str | Path, bound: int = BOUND):
        if bound < 1:
            raise ValueError("Receipt bound must be positive")
        self.path, self._bound = Path(path), bound
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            self.path, timeout=5, isolation_level=None, check_same_thread=False
        )
        try:
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS receipts("
                "request_id TEXT PRIMARY KEY, created REAL NOT NULL)"
            )
        except BaseException:
            self._db.close()
            raise

    def _record(self, request_id: str) -> None:
        with self._lock:
            self._require_open()
            with self._db:
                self._db.execute(
                    "INSERT OR IGNORE INTO receipts VALUES(?,?)", (request_id, time.time())
                )
                # Deterministic FIFO bound, identical to the in-memory store.
                self._db.execute(
                    "DELETE FROM receipts WHERE request_id NOT IN "
                    "(SELECT request_id FROM receipts ORDER BY created DESC, rowid DESC LIMIT ?)",
                    (self._bound,),
                )

    def _contains(self, request_id: str) -> bool:
        with self._lock:
            self._require_open()
            return (
                self._db.execute(
                    "SELECT 1 FROM receipts WHERE request_id=?", (request_id,)
                ).fetchone()
                is not None
            )

    def _require_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("Create new DurableReceipts in each process; never reuse after fork")
        if self._closed:
            raise JevError(
                "receipt_store_unavailable", "The completion receipt store is closed", 503
            )

    async def record(self, request_id: str) -> None:
        try:
            await asyncio.to_thread(self._record, request_id)
        except JevError:
            raise
        except Exception as exc:
            raise JevError(
                "receipt_store_unavailable", "Could not persist the completion receipt", 503
            ) from exc

    async def contains(self, request_id: str) -> bool:
        try:
            return await asyncio.to_thread(self._contains, request_id)
        except JevError:
            raise
        except Exception as exc:
            raise JevError(
                "receipt_store_unavailable", "Could not read the completion receipts", 503
            ) from exc

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._db.close()
