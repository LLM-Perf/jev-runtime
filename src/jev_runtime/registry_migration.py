"""Explicit offline schema upgrades and compatible-state code rollback staging."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from jev_runtime.registry_backup import (
    blockers,
    connect,
    digest,
    private_directory,
    save,
    summary,
    sync_directory,
    verify_snapshot,
)
from jev_runtime.registry_schema import (
    CURRENT_FORMAT,
    LEGACY_FORMAT,
    LEGACY_SCHEMA_SQL,
    PRE_QUIESCENCE_FORMAT,
    PROTOCOL_SQL,
    QUIESCENCE_TABLES,
    identify,
    require_migration_publication,
    statements,
)


def inspect(source: Path) -> dict:
    with closing(connect(source)) as db:
        db.execute("BEGIN")
        return {**identify(db), "summary": summary(db), "migration_blockers": blockers(db)}


def validate_transition(before: str, target: str) -> None:
    if (before, target) not in {
        (LEGACY_FORMAT, CURRENT_FORMAT),
        (PRE_QUIESCENCE_FORMAT, CURRENT_FORMAT),
        (CURRENT_FORMAT, LEGACY_FORMAT),
    }:
        raise ValueError(
            "Unsupported registry migration transition; choose a supported format pair"
        )


def transform(db: sqlite3.Connection, target: str) -> dict:
    before = identify(db)["format"]
    validate_transition(before, target)
    previous = summary(db)
    db.execute("BEGIN IMMEDIATE")
    try:
        if target == CURRENT_FORMAT:
            if before == PRE_QUIESCENCE_FORMAT:
                for sql in statements(LEGACY_SCHEMA_SQL):
                    if any(
                        sql.startswith("CREATE TABLE IF NOT EXISTS " + name + " ")
                        for name in QUIESCENCE_TABLES
                    ):
                        db.execute(sql)
                # This old format has no admission gate. Introduce it CLOSED,
                # requiring explicit offline resume before new workers can serve.
                db.execute(
                    "INSERT INTO backend_controls SELECT backend,1,'QUIESCING' FROM ("
                    "SELECT backend FROM bundles UNION SELECT backend FROM workers "
                    "UNION SELECT backend FROM adapters "
                    "UNION SELECT backend FROM admission_policies"
                    ") WHERE backend IS NOT NULL"
                )
            for sql in statements(PROTOCOL_SQL):
                db.execute(sql)
            db.execute("INSERT INTO owner_protocols SELECT owner,0 FROM owners")
            db.execute("PRAGMA user_version=1")
        else:
            db.execute("DROP TRIGGER require_owner_protocol")
            db.execute("DROP TABLE owner_protocols")
            db.execute("PRAGMA user_version=0")
        if identify(db)["format"] != target:
            raise ValueError("Migration produced a different schema than requested")
        after = summary(db)
        common = sorted(previous["tables"].keys() & after["tables"].keys())
        if any(previous["tables"][name] != after["tables"][name] for name in common):
            raise ValueError("Migration changed existing business state; refusing publication")
        removed = sorted(previous["tables"].keys() - after["tables"].keys())
        if removed != (["owner_protocols"] if target == LEGACY_FORMAT else []):
            raise ValueError("Migration removed an unexpected table")
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return {
        "source_format": before,
        "target_format": target,
        "preserved_tables": common,
        "added_tables": sorted(after["tables"].keys() - previous["tables"].keys()),
        "removed_tables": removed,
        "summary": after,
    }


def stage_migration(
    snapshot_dir: Path,
    manifest_sha256: str,
    source: Path,
    destination: Path,
    target_format: str = CURRENT_FORMAT,
) -> dict:
    manifest = verify_snapshot(snapshot_dir, manifest_sha256)
    if source.resolve() == (snapshot_dir / "registry.sqlite3").resolve():
        raise ValueError("Source registry must be independent of the snapshot")
    identity = (source.stat().st_dev, source.stat().st_ino)
    with closing(connect(source, writable=True)) as original:
        original.execute("BEGIN IMMEDIATE")
        try:
            source_format = identify(original)["format"]
            validate_transition(source_format, target_format)
            state = summary(original)
            if state != manifest["summary"]:
                raise ValueError("Source changed since snapshot; take a fresh snapshot")
            outstanding = blockers(original)
            if outstanding:
                raise ValueError(
                    "Source is not stopped/drained: " + json.dumps(outstanding, sort_keys=True)
                )
            if (source.stat().st_dev, source.stat().st_ino) != identity:
                raise ValueError("Source registry was replaced during migration validation")
            destination = private_directory(destination)
            operation = uuid.uuid4().hex
            save(
                destination / "migration-intent.json",
                {
                    "operation_id": operation,
                    "database_name": "registry.sqlite3",
                    "source_format": source_format,
                    "target_format": target_format,
                },
            )
            pending = destination / "registry.pending.sqlite3"
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with (
                (snapshot_dir / "registry.sqlite3").open("rb") as reader,
                os.fdopen(fd, "wb") as writer,
            ):
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
            if digest(pending) != manifest["database_sha256"]:
                raise ValueError("Migration input differs from the verified snapshot")
            with closing(connect(pending, writable=True)) as staged:
                result = transform(staged, target_format)
            with pending.open("rb") as stream:
                os.fsync(stream.fileno())
            if (source.stat().st_dev, source.stat().st_ino) != identity:
                raise ValueError("Source registry was replaced during staging")
            receipt = {
                "format": "jev-registry-migration-v1",
                "operation_id": operation,
                "created_at": time.time(),
                "source_path": str(source.resolve()),
                "source_identity": {"device": identity[0], "inode": identity[1]},
                "snapshot_manifest_sha256": manifest_sha256,
                "intent_sha256": digest(destination / "migration-intent.json"),
                "source_summary": state,
                "database_sha256": digest(pending),
                "database_bytes": pending.stat().st_size,
                **result,
                "activated": False,
            }
            # Receipt first, final database name last. A crash before publication
            # leaves no startable final database. Startup checks the intent too.
            save(destination / "migration.json", receipt)
            os.link(pending, destination / "registry.sqlite3")
            pending.unlink()
            sync_directory(destination)
            sync_directory(destination.parent)
        finally:
            original.rollback()
    return {
        "path": str(destination),
        "receipt_sha256": digest(destination / "migration.json"),
        "database_sha256": receipt["database_sha256"],
        **result,
        "activated": False,
    }


def verify_migration(directory: Path, receipt_sha256: str) -> dict:
    if len(receipt_sha256) != 64 or any(c not in "0123456789abcdef" for c in receipt_sha256):
        raise ValueError("Explicit lowercase SHA256 receipt digest required")
    if directory.is_symlink():
        raise ValueError("Migration directory must not be a symlink")
    if {p.name for p in directory.iterdir()} != {
        "migration-intent.json",
        "migration.json",
        "registry.sqlite3",
    }:
        raise ValueError("Migration inventory is incomplete or contains unexpected files")
    for name in ("migration-intent.json", "migration.json", "registry.sqlite3"):
        p = directory / name
        if p.is_symlink() or not p.is_file():
            raise ValueError("Migration publication is incomplete or contains symlinks")
    require_migration_publication(directory / "registry.sqlite3")
    receipt_path = directory / "migration.json"
    if digest(receipt_path) != receipt_sha256:
        raise ValueError("Migration receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text())
    if digest(directory / "migration-intent.json") != receipt["intent_sha256"]:
        raise ValueError("Migration intent hash mismatch")
    database = directory / "registry.sqlite3"
    if (
        digest(database) != receipt["database_sha256"]
        or database.stat().st_size != receipt["database_bytes"]
    ):
        raise ValueError("Staged database changed; verify before first startup")
    with closing(connect(database)) as db:
        if identify(db)["format"] != receipt["target_format"] or summary(db) != receipt["summary"]:
            raise ValueError("Staged state differs from migration receipt")
    return receipt
