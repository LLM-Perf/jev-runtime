"""Consistent private SQLite snapshots and guarded, non-activating restore staging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from jev_runtime.registry import REGISTRY_SCHEMA_SQL, Registry


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def schema(db: sqlite3.Connection) -> list:
    return db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name,tbl_name"
    ).fetchall()


@lru_cache(maxsize=1)
def expected_schema() -> bytes:
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        return canonical(schema(db))


def connect(path: Path, *, writable: bool = False) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Registry must be an existing regular file, not a symlink")
    mode = "rw" if writable else "ro"
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=" + mode, uri=True, timeout=5)
    db.execute("PRAGMA trusted_schema=OFF")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA synchronous=FULL")
    if not writable:
        db.execute("PRAGMA query_only=ON")
    return db


def summary(db: sqlite3.Connection) -> dict:
    structure = canonical(schema(db))
    if structure != expected_schema() or db.execute("PRAGMA user_version").fetchone()[0] != 0:
        raise ValueError("Registry schema is not supported by this backup tool version")
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("Registry integrity check failed")
    if db.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Registry contains foreign-key violations")
    tables, overall = {}, hashlib.sha256()
    names = [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name")
    ]
    for name in names:
        checksum, count = hashlib.sha256(), 0
        # The schema is exactly the known ordinary-table schema at this point.
        quoted = '"' + name.replace('"', '""') + '"'
        for row in db.execute(f"SELECT rowid,* FROM {quoted} ORDER BY rowid"):
            payload = canonical(
                [{"blob": value.hex()} if isinstance(value, bytes) else value for value in row]
            )
            checksum.update(len(payload).to_bytes(8, "big") + payload)
            count += 1
        tables[name] = {"rows": count, "sha256": checksum.hexdigest()}
        overall.update(canonical([name, tables[name]]))
    return {
        "schema_sha256": hashlib.sha256(structure).hexdigest(),
        "content_sha256": overall.hexdigest(),
        "tables": tables,
    }


def blockers(db: sqlite3.Connection) -> dict[str, int]:
    result = {}
    for table in (
        "leases",
        "lease_work",
        "lease_tenants",
        "admission_tickets",
        "raw_work",
        "recovery_claims",
    ):
        count = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if count:
            result[table] = count
    for name, query in {
        "preparing_bundles": "SELECT count(*) FROM bundles WHERE state='PREPARING'",
        "resident_or_uncertain_adapters": "SELECT count(*) FROM adapters WHERE state!='UNLOADED'",
        "pending_cancel_commands": (
            "SELECT count(*) FROM cancel_commands WHERE state IN ('PENDING','RUNNING')"
        ),
    }.items():
        count = db.execute(query).fetchone()[0]
        if count:
            result[name] = count
    for (identity,) in db.execute("SELECT identity FROM owners"):
        state = Registry._owner_status(json.loads(identity))
        if state != "dead":
            key = "owners_" + state
            result[key] = result.get(key, 0) + 1
    return result


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path: Path, value: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    # A final receipt is never visible with partially written JSON. Hard-link
    # publication is exclusive and cannot replace an existing receipt.
    os.link(temporary, path)
    temporary.unlink()
    sync_directory(path.parent)


def private_directory(path: Path) -> Path:
    path = path.absolute()
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    return path


def snapshot(source: Path, destination: Path, timeout_seconds: float = 30) -> dict:
    if not 0 < timeout_seconds <= 3600:
        raise ValueError("Snapshot timeout must be greater than zero and at most 3600 seconds")
    # Open the source before creating anything; a typo must not create a registry.
    with closing(connect(source)) as original:
        destination = private_directory(destination)
        path = destination / "registry.sqlite3"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        deadline = time.monotonic() + timeout_seconds

        def progress(status, remaining, total):
            if time.monotonic() > deadline:
                raise TimeoutError("Registry snapshot deadline exceeded")

        with closing(sqlite3.connect(path)) as target:
            original.backup(target, pages=256, progress=progress, sleep=0.05)
            target.execute("PRAGMA journal_mode=DELETE")
            target.execute("PRAGMA trusted_schema=OFF")
            state = summary(target)
            outstanding = blockers(target)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "format": "jev-registry-snapshot-v1",
            "created_at": time.time(),
            "source_path": str(source.resolve()),
            "sqlite_version": sqlite3.sqlite_version,
            "database_sha256": digest(path),
            "database_bytes": path.stat().st_size,
            "summary": state,
            "restore_blockers_at_snapshot": outstanding,
        }
        save(destination / "manifest.json", manifest)
        sync_directory(destination)
        sync_directory(destination.parent)
    return {
        "path": str(destination),
        "manifest_sha256": digest(destination / "manifest.json"),
        "summary": state,
        "restore_blockers_at_snapshot": outstanding,
    }


def verify_snapshot(directory: Path, manifest_sha256: str) -> dict:
    if len(manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha256):
        raise ValueError("An explicit lowercase SHA256 snapshot manifest digest is required")
    if directory.is_symlink() or {p.name for p in directory.iterdir()} != {
        "registry.sqlite3",
        "manifest.json",
    }:
        raise ValueError("Snapshot inventory is incomplete or contains unexpected files")
    manifest_path, database = directory / "manifest.json", directory / "registry.sqlite3"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, database)):
        raise ValueError("Snapshot files must not be symlinks")
    if digest(manifest_path) != manifest_sha256:
        raise ValueError("Snapshot manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "jev-registry-snapshot-v1":
        raise ValueError("Unsupported registry snapshot")
    if (
        digest(database) != manifest["database_sha256"]
        or database.stat().st_size != manifest["database_bytes"]
    ):
        raise ValueError("Snapshot database hash or size mismatch")
    with closing(connect(database)) as db:
        if db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
            raise ValueError("Snapshot must be a standalone database without a WAL")
        if summary(db) != manifest["summary"]:
            raise ValueError("Snapshot content differs from its manifest")
    return manifest


def stage_restore(
    snapshot_dir: Path, manifest_sha256: str, source: Path, destination: Path
) -> dict:
    manifest = verify_snapshot(snapshot_dir, manifest_sha256)
    if source.resolve() == (snapshot_dir / "registry.sqlite3").resolve():
        raise ValueError("The source registry must be independent of the backup snapshot")
    identity = (source.stat().st_dev, source.stat().st_ino)
    with closing(connect(source, writable=True)) as original:
        # Reserve the writer throughout validation/publication. This cannot fence
        # a future restart after staging; the operator must keep old workers stopped.
        original.execute("BEGIN IMMEDIATE")
        try:
            state = summary(original)
            if state != manifest["summary"]:
                raise ValueError(
                    "Source changed since snapshot; take a new snapshot before staging"
                )
            outstanding = blockers(original)
            if outstanding:
                raise ValueError(
                    "Source is not safely stopped/drained: "
                    + json.dumps(outstanding, sort_keys=True)
                )
            if (source.stat().st_dev, source.stat().st_ino) != identity:
                raise ValueError("Source registry file was replaced during validation")
            destination = private_directory(destination)
            target = destination / "registry.sqlite3"
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with (
                (snapshot_dir / "registry.sqlite3").open("rb") as reader,
                os.fdopen(fd, "wb") as writer,
            ):
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
            if digest(target) != manifest["database_sha256"]:
                raise ValueError("Staged database differs from verified snapshot")
            with closing(connect(target)) as staged:
                if summary(staged) != state:
                    raise ValueError("Staged registry state differs from the stopped source")
            if (source.stat().st_dev, source.stat().st_ino) != identity:
                raise ValueError("Source registry file was replaced during staging")
            receipt = {
                "format": "jev-registry-staged-restore-v1",
                "created_at": time.time(),
                "source_path": str(source.resolve()),
                "source_identity": {"device": source.stat().st_dev, "inode": source.stat().st_ino},
                "snapshot_manifest_sha256": manifest_sha256,
                "database_sha256": digest(target),
                "summary": state,
                "activated": False,
                "qualification": (
                    "Inactive copy of identical, stopped same-host state; "
                    "no lost-database recovery or schema migration"
                ),
            }
            save(destination / "restore.json", receipt)
            sync_directory(destination)
            sync_directory(destination.parent)
        finally:
            original.rollback()
    return {
        "path": str(destination),
        "receipt_sha256": digest(destination / "restore.json"),
        "database_sha256": receipt["database_sha256"],
        "summary": state,
        "activated": False,
    }
