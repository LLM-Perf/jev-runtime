"""Exact SQLite format contracts; ordinary startup never migrates existing state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path

LEGACY_SCHEMA_SQL = """
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
                CREATE TABLE IF NOT EXISTS backend_controls (
                    backend TEXT PRIMARY KEY, generation INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('OPEN','QUIESCING'))
                );
                CREATE TABLE IF NOT EXISTS worker_quiescence (
                    owner TEXT PRIMARY KEY REFERENCES workers(owner), protocol INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS raw_work (
                    id TEXT PRIMARY KEY, backend TEXT NOT NULL,
                    owner TEXT NOT NULL REFERENCES owners(owner),
                    request_id TEXT NOT NULL, phase TEXT NOT NULL,
                    UNIQUE(backend,request_id)
                );
                CREATE TABLE IF NOT EXISTS recovery_claims (
                    resource TEXT PRIMARY KEY, backend TEXT NOT NULL,
                    owner TEXT NOT NULL REFERENCES owners(owner)
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

CURRENT_FORMAT = "versioned-v1"
LEGACY_FORMAT = "legacy-quiescence-v0"
PRE_QUIESCENCE_FORMAT = "legacy-pre-quiescence-v0"
QUIESCENCE_TABLES = {"backend_controls", "worker_quiescence", "raw_work", "recovery_claims"}
PROTOCOL_SQL = """
CREATE TABLE owner_protocols (
    owner TEXT PRIMARY KEY REFERENCES owners(owner) DEFERRABLE INITIALLY DEFERRED,
    protocol INTEGER NOT NULL CHECK(protocol IN (0,1))
);
CREATE TRIGGER require_owner_protocol BEFORE INSERT ON owners
WHEN NOT EXISTS (SELECT 1 FROM owner_protocols WHERE owner=NEW.owner AND protocol=1)
BEGIN
    SELECT RAISE(ABORT, 'Registry protocol required; use compatible code or explicit migration');
END;
"""
REGISTRY_SCHEMA_SQL = LEGACY_SCHEMA_SQL + PROTOCOL_SQL + "\nPRAGMA user_version=1;\n"


def statements(script: str):
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending.strip()
            pending = ""
    if pending.strip():
        raise ValueError("Incomplete registry schema SQL")


def schema(db: sqlite3.Connection) -> list:
    return [
        tuple(row)
        for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name,tbl_name"
        )
    ]


def canonical_schema(db: sqlite3.Connection) -> bytes:
    return json.dumps(schema(db), sort_keys=True, separators=(",", ":")).encode()


def legacy_script(format_id: str) -> str:
    if format_id == LEGACY_FORMAT:
        return LEGACY_SCHEMA_SQL
    if format_id != PRE_QUIESCENCE_FORMAT:
        raise ValueError("Unknown legacy registry format")
    return (
        "\n".join(
            sql
            for sql in statements(LEGACY_SCHEMA_SQL)
            if not any(
                sql.startswith("CREATE TABLE IF NOT EXISTS " + name + " ")
                for name in QUIESCENCE_TABLES
            )
        )
        + "\n"
    )


@lru_cache(maxsize=3)
def expected_schema(format_id: str = CURRENT_FORMAT) -> bytes:
    script = REGISTRY_SCHEMA_SQL if format_id == CURRENT_FORMAT else legacy_script(format_id)
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(script)
        return canonical_schema(db)


def identify(db: sqlite3.Connection) -> dict:
    structure = canonical_schema(db)
    version = db.execute("PRAGMA user_version").fetchone()[0]
    for format_id in (CURRENT_FORMAT, LEGACY_FORMAT, PRE_QUIESCENCE_FORMAT):
        if version == (1 if format_id == CURRENT_FORMAT else 0) and structure == expected_schema(
            format_id
        ):
            return {
                "format": format_id,
                "user_version": version,
                "schema_sha256": hashlib.sha256(structure).hexdigest(),
            }
    raise ValueError("Registry schema/version is unknown or altered; refusing implicit migration")


def initialize(db: sqlite3.Connection) -> None:
    # Serialize the fresh-database check with creation. executescript would commit
    # first and split schema creation across transactions, so execute each DDL.
    db.execute("BEGIN IMMEDIATE")
    try:
        if not schema(db) and db.execute("PRAGMA user_version").fetchone()[0] == 0:
            for sql in statements(REGISTRY_SCHEMA_SQL):
                if not sql.startswith("PRAGMA journal_mode"):
                    db.execute(sql)
        found = identify(db)
        if found["format"] != CURRENT_FORMAT:
            raise ValueError(
                "Registry migration required: stop and drain old workers, snapshot, "
                "then use jevctl registry stage-migration before starting this runtime"
            )
        db.commit()
    except BaseException:
        db.rollback()
        raise


def register_owner(db: sqlite3.Connection, owner: str, identity: dict) -> None:
    # This deferred FK and trigger form the client compatibility handshake.
    # Legacy constructors insert into owners directly and fail before joining
    # admission or issuing any engine work. Existing clients must be stopped
    # before migration; a database cannot revoke code already holding GPU work.
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("INSERT INTO owner_protocols VALUES(?,1)", (owner,))
        db.execute("INSERT INTO owners VALUES(?,?,?)", (owner, json.dumps(identity), time.time()))
        db.commit()
    except BaseException:
        db.rollback()
        raise


def require_migration_publication(path: Path) -> None:
    intent = path.parent / "migration-intent.json"
    if not intent.exists():
        return
    declared = json.loads(intent.read_text())
    if declared.get("database_name") != path.name:
        return
    receipt = path.parent / "migration.json"
    if not receipt.is_file() or not path.is_file():
        raise ValueError("Registry migration is incomplete; verify publication before startup")
    completed = json.loads(receipt.read_text())
    if (
        completed.get("format") != "jev-registry-migration-v1"
        or completed.get("operation_id") != declared.get("operation_id")
        or not declared.get("operation_id")
    ):
        raise ValueError("Registry migration publication does not match its intent")
