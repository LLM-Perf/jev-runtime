"""Coordinated, stopped-state registry + TokenSpeed receipt snapshots.

Both SQLite writer reservations remain held while taking/verifying the pair.
Fenced owners are stopped for this purpose. Pending receipts are never deleted
here; completed-receipt retirement is a separate explicit operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from jev_runtime.registry_backup import (
    blockers,
    canonical,
    connect,
    digest,
    private_directory,
    save,
    snapshot,
    summary,
    sync_directory,
    verify_snapshot,
)


def receipt_summary(db):
    if db.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Unsupported receipt schema")
    names = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    columns = db.execute("PRAGMA table_info(receipts)").fetchall()
    retired = "retired_receipts" in names
    if names not in (["receipts"], ["receipts", "retired_receipts"]) or [
        (r[1], r[2], r[3], r[5]) for r in columns
    ] != [
        ("namespace", "TEXT", 1, 1),
        ("request_id", "TEXT", 1, 2),
        ("state", "TEXT", 1, 0),
    ]:
        raise ValueError("Incompatible receipt database")
    if retired:
        retired_columns = db.execute("PRAGMA table_info(retired_receipts)").fetchall()
        if [(r[1], r[2], r[3], r[5]) for r in retired_columns] != [
            ("namespace", "TEXT", 1, 1),
            ("request_id", "TEXT", 1, 2),
            ("operation_id", "TEXT", 1, 0),
        ]:
            raise ValueError("Incompatible retired receipt table")
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("Receipt integrity check failed")
    h, count, pending = hashlib.sha256(), 0, 0
    for row in db.execute(
        "SELECT namespace,request_id,state FROM receipts ORDER BY namespace,request_id"
    ):
        if row[2] not in {"pending", "completed"}:
            raise ValueError("Invalid receipt state")
        data = canonical(row)
        h.update(len(data).to_bytes(8, "big") + data)
        count += 1
        pending += row[2] == "pending"
    retired_count = 0
    if retired:
        for row in db.execute(
            "SELECT namespace,request_id,operation_id FROM retired_receipts "
            "ORDER BY namespace,request_id"
        ):
            data = canonical(row)
            h.update(len(data).to_bytes(8, "big") + data)
            retired_count += 1
    summary = {"rows": count, "pending": pending, "content_sha256": h.hexdigest()}
    if retired:
        summary["retired"] = retired_count
    return summary


@contextmanager
def stopped_pair(source: Path):
    receipts = source.with_suffix(".tokenspeed-receipts.db")
    identities = [(p.stat().st_dev, p.stat().st_ino) for p in (source, receipts)]
    with ExitStack() as stack:
        registry = stack.enter_context(closing(connect(source, writable=True)))
        native = stack.enter_context(closing(connect(receipts, writable=True)))
        registry.execute("BEGIN IMMEDIATE")
        native.execute("BEGIN IMMEDIATE")
        try:
            state, completion = summary(registry), receipt_summary(native)
            outstanding = blockers(registry)
            if outstanding or completion["pending"]:
                raise ValueError(
                    "Pair is not stopped/drained: "
                    + json.dumps(
                        {"registry": outstanding, "pending_receipts": completion["pending"]}
                    )
                )
            yield state, completion
            if identities != [(p.stat().st_dev, p.stat().st_ino) for p in (source, receipts)]:
                raise ValueError("Source database identity changed")
        finally:
            native.rollback()
            registry.rollback()


def copy_database(source: Path, target: Path):
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    deadline = time.monotonic() + 30

    def progress(*args):
        if time.monotonic() > deadline:
            raise TimeoutError("Pair backup exceeded 30 seconds")

    with closing(connect(source)) as reader, closing(sqlite3.connect(target)) as writer:
        reader.backup(writer, pages=256, progress=progress, sleep=0.05)
        writer.execute("PRAGMA journal_mode=DELETE")
    with target.open("rb") as file:
        os.fsync(file.fileno())


def snapshot_pair(source: Path, destination: Path) -> dict:
    with stopped_pair(source) as (state, completion):
        destination = private_directory(destination)
        core = snapshot(source, destination / "registry")
        if core["summary"] != state:
            raise ValueError("Registry changed under writer reservation")
        path = destination / "receipts.sqlite3"
        copy_database(source.with_suffix(".tokenspeed-receipts.db"), path)
        with closing(connect(path)) as db:
            if receipt_summary(db) != completion:
                raise ValueError("Receipt copy differs from source")
        manifest = {
            "format": "jev-paired-snapshot-v1",
            "created_at": time.time(),
            "registry_manifest_sha256": core["manifest_sha256"],
            "receipts_sha256": digest(path),
            "receipts_summary": completion,
        }
    # Publication occurs only after source identity checks and both copy checks.
    save(destination / "manifest.json", manifest)
    sync_directory(destination.parent)
    return {"path": str(destination), "manifest_sha256": digest(destination / "manifest.json")}


def verify_pair(directory: Path, expected: str) -> dict:
    if directory.is_symlink() or {p.name for p in directory.iterdir()} != {
        "manifest.json",
        "registry",
        "receipts.sqlite3",
    }:
        raise ValueError("Unexpected pair inventory")
    path = directory / "manifest.json"
    receipt_path = directory / "receipts.sqlite3"
    if path.is_symlink() or receipt_path.is_symlink() or digest(path) != expected:
        raise ValueError("Pair manifest mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "jev-paired-snapshot-v1":
        raise ValueError("Unsupported pair manifest")
    verify_snapshot(directory / "registry", manifest["registry_manifest_sha256"])
    if digest(receipt_path) != manifest["receipts_sha256"]:
        raise ValueError("Receipt snapshot hash mismatch")
    with closing(connect(receipt_path)) as db:
        if db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
            raise ValueError("Receipt snapshot requires standalone DELETE journal mode")
        if receipt_summary(db) != manifest["receipts_summary"]:
            raise ValueError("Receipt snapshot state mismatch")
    return manifest


def restore_pair(directory: Path, expected: str, source: Path, destination: Path) -> dict:
    manifest = verify_pair(directory, expected)
    core = verify_snapshot(directory / "registry", manifest["registry_manifest_sha256"])
    if source.resolve() == (directory / "registry/registry.sqlite3").resolve():
        raise ValueError("Source must be independent of backup")
    with stopped_pair(source) as (state, completion):
        if state != core["summary"] or completion != manifest["receipts_summary"]:
            raise ValueError("Source pair changed since snapshot")
        destination = private_directory(destination)
        operation = uuid.uuid4().hex
        intent = {"operation_id": operation, "database_name": "registry.sqlite3"}
        save(destination / "pair-restore-intent.json", intent)
        registry_path = destination / "registry.sqlite3"
        native_path = registry_path.with_suffix(".tokenspeed-receipts.db")
        copy_database(directory / "registry/registry.sqlite3", registry_path)
        copy_database(directory / "receipts.sqlite3", native_path)
        with closing(connect(registry_path)) as db:
            if summary(db) != state:
                raise ValueError("Restored registry content mismatch")
        with closing(connect(native_path)) as db:
            if receipt_summary(db) != completion:
                raise ValueError("Restored receipt content mismatch")
        result = {
            "format": "jev-paired-restore-v1",
            "operation_id": operation,
            "snapshot_sha256": expected,
            "activated": False,
            "registry_sha256": digest(registry_path),
            "receipts_sha256": digest(native_path),
        }
    save(destination / "pair-restore.json", result)
    sync_directory(destination.parent)
    return {
        **result,
        "path": str(destination),
        "receipt_sha256": digest(destination / "pair-restore.json"),
    }
