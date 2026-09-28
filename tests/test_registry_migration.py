import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from typer.testing import CliRunner

from jev_runtime import registry_migration
from jev_runtime.cli import app
from jev_runtime.registry import Registry
from jev_runtime.registry_backup import connect, digest, snapshot, summary
from jev_runtime.registry_migration import inspect, stage_migration, verify_migration
from jev_runtime.registry_schema import (
    CURRENT_FORMAT,
    LEGACY_FORMAT,
    LEGACY_SCHEMA_SQL,
    PRE_QUIESCENCE_FORMAT,
    identify,
    legacy_script,
)


def old_registry(tmp_path, bundle, format_id=LEGACY_FORMAT):
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as db:
        db.executescript(legacy_script(format_id))
        db.execute(
            "INSERT INTO owners VALUES('old',?,1)", (json.dumps(Registry._process_identity()),)
        )
        db.execute("INSERT INTO workers VALUES('old','fixture:0','STOPPED')")
        db.execute(
            "INSERT INTO bundles VALUES(?,?,?,'READY','fixture:0',NULL,1)",
            (bundle.reference, bundle.digest, bundle.model_dump_json()),
        )
        db.execute("INSERT INTO routes VALUES('model',?,7)", (bundle.reference,))
        db.execute("INSERT INTO events(ts,action,details) VALUES(1,'retained','{}')")
        if format_id == LEGACY_FORMAT:
            db.execute("INSERT INTO backend_controls VALUES('fixture:0',5,'QUIESCING')")
        db.commit()
    return path


def stopped(monkeypatch):
    # This is a CPU-only owner simulation. DSW uses real Linux process identities.
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: "dead"))


def stage(path, tmp_path, **kwargs):
    backup = tmp_path / "backup"
    snap = snapshot(path, backup)
    return stage_migration(backup, snap["manifest_sha256"], path, tmp_path / "migrated", **kwargs)


@pytest.mark.parametrize("format_id", [LEGACY_FORMAT, PRE_QUIESCENCE_FORMAT])
def test_startup_rejects_legacy_without_writing(tmp_path, bundle, format_id):
    path = old_registry(tmp_path, bundle, format_id)
    before = digest(path)
    with pytest.raises(ValueError, match="migration required"):
        Registry(path)
    assert digest(path) == before
    assert inspect(path)["format"] == format_id


@pytest.mark.parametrize("format_id", [LEGACY_FORMAT, PRE_QUIESCENCE_FORMAT])
def test_upgrade_preserves_state_and_blocks_unnegotiated_old_owner(
    tmp_path, bundle, monkeypatch, format_id
):
    stopped(monkeypatch)
    path = old_registry(tmp_path, bundle, format_id)
    before = inspect(path)
    result = stage(path, tmp_path)
    dest = tmp_path / "migrated"
    checked = verify_migration(dest, result["receipt_sha256"])
    assert checked["source_format"] == format_id and checked["target_format"] == CURRENT_FORMAT
    assert inspect(path) == before
    with closing(connect(dest / "registry.sqlite3", writable=True)) as db:
        assert identify(db)["user_version"] == 1
        assert db.execute("SELECT * FROM routes").fetchall() == [("model", bundle.reference, 7)]
        assert db.execute("SELECT * FROM owner_protocols").fetchall() == [("old", 0)]
        assert db.execute("SELECT state FROM backend_controls").fetchall() == [("QUIESCING",)]
        # The old constructor executes this DDL then inserts an owner directly.
        db.executescript(LEGACY_SCHEMA_SQL)
        with pytest.raises(sqlite3.IntegrityError, match="Registry protocol required"):
            db.execute("INSERT INTO owners VALUES('legacy-new','{}',2)")
        db.rollback()
        assert db.execute("SELECT COUNT(*) FROM owners").fetchone()[0] == 1
    registry = Registry(dest / "registry.sqlite3")
    try:
        assert registry.list()["routes"][0]["generation"] == 7
        assert registry.inspect(bundle.reference)["digest"] == bundle.digest
        generation = 5 if format_id == LEGACY_FORMAT else 1
        assert registry.backend_control("fixture:0")["generation"] == generation
        registry.resume_backend("fixture:0", generation)
        assert registry.backend_control("fixture:0")["generation"] == generation + 1
    finally:
        registry.close()


def test_explicit_downgrade_preserves_latest_state_and_rejects_new_runtime(
    runtime, tmp_path, monkeypatch
):
    stopped(monkeypatch)
    runtime.registry.disable("model", 1)
    runtime.registry.activate("model", "test@1", 2)
    result = stage(runtime.registry.path, tmp_path, target_format=LEGACY_FORMAT)
    path = tmp_path / "migrated/registry.sqlite3"
    verify_migration(path.parent, result["receipt_sha256"])
    assert result["removed_tables"] == ["owner_protocols"]
    assert inspect(path)["format"] == LEGACY_FORMAT
    with pytest.raises(ValueError, match="migration required"):
        Registry(path)
    with closing(connect(path, writable=True)) as db:
        db.execute("INSERT INTO owners VALUES('rollback','{}',3)")
        db.commit()
        assert db.execute("SELECT generation FROM routes").fetchone()[0] == 3


@pytest.mark.parametrize("state", ["alive", "unknown"])
def test_migration_requires_process_death_not_only_stopped_rows(
    tmp_path, bundle, monkeypatch, state
):
    path = old_registry(tmp_path, bundle)
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: state))
    with pytest.raises(ValueError, match="owners_" + state):
        stage(path, tmp_path)
    assert not (tmp_path / "migrated").exists()


@pytest.mark.parametrize("kind", ["lease", "raw", "recovery", "prepare", "cancel", "adapter"])
def test_migration_keeps_uncertain_work(tmp_path, bundle, monkeypatch, kind):
    path = old_registry(tmp_path, bundle)
    stopped(monkeypatch)
    with sqlite3.connect(path) as db:
        if kind == "lease":
            db.execute("INSERT INTO leases VALUES('l',?,'old','r',1)", (bundle.reference,))
        elif kind == "raw":
            db.execute("INSERT INTO raw_work VALUES('w','fixture:0','old','r','uncertain')")
        elif kind == "recovery":
            db.execute("INSERT INTO recovery_claims VALUES('r','fixture:0','old')")
        elif kind == "prepare":
            db.execute("UPDATE bundles SET state='PREPARING'")
        elif kind == "adapter":
            db.execute(
                "INSERT INTO adapters VALUES('a','{}','d','{}','fixture:0',"
                "'READY',NULL,'old',NULL,1)"
            )
        else:
            db.execute(
                "INSERT INTO cancel_commands VALUES('c','l','old','r','default','PENDING',1)"
            )
    before = inspect(path)
    with pytest.raises(ValueError, match="not stopped/drained"):
        stage(path, tmp_path)
    assert inspect(path) == before and not (tmp_path / "migrated").exists()


@pytest.mark.parametrize("mutation", ["future-version", "extra-table", "missing-trigger"])
def test_unknown_version_or_schema_cannot_start_or_snapshot(tmp_path, mutation):
    path = tmp_path / "registry.db"
    Registry(path).close()
    with sqlite3.connect(path) as db:
        db.execute(
            {
                "future-version": "PRAGMA user_version=2",
                "extra-table": "CREATE TABLE future(x)",
                "missing-trigger": "DROP TRIGGER require_owner_protocol",
            }[mutation]
        )
    before = digest(path)
    with pytest.raises(ValueError, match="schema/version"):
        Registry(path)
    with pytest.raises(ValueError, match="schema/version"):
        snapshot(path, tmp_path / "backup")
    assert digest(path) == before


def test_fresh_concurrent_startup_registers_every_owner_atomically(tmp_path):
    path = tmp_path / "registry.db"

    def open_owner(_):
        registry = Registry(path)
        owner = registry.owner
        registry.close()
        return owner

    with ThreadPoolExecutor(max_workers=8) as pool:
        owners = set(pool.map(open_owner, range(16)))
    with sqlite3.connect(path) as db:
        assert owners == {row[0] for row in db.execute("SELECT owner FROM owners")}
        assert (
            db.execute("SELECT count(*) FROM owner_protocols WHERE protocol=1").fetchone()[0] == 16
        )
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def test_stale_backup_and_wrong_transition_do_not_allocate(tmp_path, bundle, monkeypatch):
    stopped(monkeypatch)
    path = old_registry(tmp_path, bundle)
    backup = tmp_path / "backup"
    snap = snapshot(path, backup)
    with pytest.raises(ValueError, match="Unsupported"):
        stage_migration(backup, snap["manifest_sha256"], path, tmp_path / "invalid", LEGACY_FORMAT)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE routes SET generation=8")
    with pytest.raises(ValueError, match="changed since snapshot"):
        stage_migration(backup, snap["manifest_sha256"], path, tmp_path / "stale")
    assert not (tmp_path / "invalid").exists() and not (tmp_path / "stale").exists()


@pytest.mark.parametrize("phase", ["receipt", "database-publication"])
def test_interrupted_migration_cannot_create_an_empty_startable_registry(
    tmp_path, bundle, monkeypatch, phase
):
    stopped(monkeypatch)
    path = old_registry(tmp_path, bundle)
    before = inspect(path)
    save, link = registry_migration.save, registry_migration.os.link

    def fail_save(path, value):
        if path.name == "migration.json" and phase == "receipt":
            raise OSError("injected receipt failure")
        return save(path, value)

    def fail_link(source, target):
        if target.name == "registry.sqlite3" and phase == "database-publication":
            raise OSError("injected publication failure")
        return link(source, target)

    monkeypatch.setattr(registry_migration, "save", fail_save)
    monkeypatch.setattr(registry_migration.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        stage(path, tmp_path)
    final = tmp_path / "migrated/registry.sqlite3"
    with pytest.raises(ValueError, match="incomplete"):
        Registry(final)
    assert not final.exists()
    assert inspect(path) == before


def test_writer_lock_covers_transformation_and_publication(tmp_path, bundle, monkeypatch):
    stopped(monkeypatch)
    path = old_registry(tmp_path, bundle)
    transform = registry_migration.transform
    observed = []

    def contender(db, target):
        with sqlite3.connect(path, timeout=0.01) as other:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("UPDATE routes SET generation=99")
        observed.append(True)
        return transform(db, target)

    monkeypatch.setattr(registry_migration, "transform", contender)
    stage(path, tmp_path)
    assert observed == [True]


def test_cli_inspection_and_migration_verify_without_new_owner(tmp_path, bundle, monkeypatch):
    stopped(monkeypatch)
    path = old_registry(tmp_path, bundle)
    runner = CliRunner()
    r = runner.invoke(app, ["registry", "inspect", str(path)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["format"] == LEGACY_FORMAT
    result = stage(path, tmp_path)
    r = runner.invoke(
        app, ["registry", "verify-migration", str(tmp_path / "migrated"), result["receipt_sha256"]]
    )
    assert r.exit_code == 0, r.output
    with closing(connect(path)) as db:
        assert summary(db)["tables"]["owners"]["rows"] == 1


@pytest.mark.parametrize("change", ["intent", "receipt", "database", "extra"])
def test_migration_verifier_rejects_modified_artifact(tmp_path, bundle, monkeypatch, change):
    stopped(monkeypatch)
    path = old_registry(tmp_path, bundle)
    result = stage(path, tmp_path)
    directory = tmp_path / "migrated"
    if change in {"intent", "receipt"}:
        name = "migration-intent.json" if change == "intent" else "migration.json"
        with (directory / name).open("a") as f:
            f.write(" ")
    elif change == "database":
        with sqlite3.connect(directory / "registry.sqlite3") as db:
            db.execute("UPDATE routes SET generation=100")
    else:
        (directory / "unexpected").touch()
    with pytest.raises(ValueError):
        verify_migration(directory, result["receipt_sha256"])
