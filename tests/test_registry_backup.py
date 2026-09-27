from __future__ import annotations

import json
import sqlite3
import stat
from contextlib import closing

import pytest
from typer.testing import CliRunner

from jev_runtime.cli import app
from jev_runtime.registry import Registry
from jev_runtime.registry_backup import (
    connect,
    digest,
    snapshot,
    stage_restore,
    summary,
    verify_snapshot,
)


def dead(monkeypatch):
    # Unit-only ownership simulation; real Linux process death is tested on DSW.
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: "dead"))


def test_online_snapshot_preserves_committed_wal_and_inflight_work(runtime, tmp_path):
    registry = runtime.registry
    lease = registry.acquire("model", "snapshot-request", None, runtime.backend_identity)
    registry.record_branches(lease.lease_id, ["branch-one", "branch-two"])
    assert registry.path.with_name(registry.path.name + "-wal").stat().st_size > 0
    destination = tmp_path / "snapshot"
    record = snapshot(registry.path, destination)
    manifest = verify_snapshot(destination, record["manifest_sha256"])
    assert manifest["restore_blockers_at_snapshot"]["leases"] == 1
    with closing(connect(destination / "registry.sqlite3")) as copied:
        assert copied.execute("SELECT id,request_id FROM leases").fetchall() == [
            (lease.lease_id, "snapshot-request")
        ]
        assert copied.execute("SELECT phase,branches FROM lease_work").fetchall() == [
            ("inflight", '["branch-one", "branch-two"]')
        ]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in destination.iterdir())
    # Later source changes do not mutate the completed snapshot.
    registry.release(lease.lease_id)
    verify_snapshot(destination, record["manifest_sha256"])
    with pytest.raises(ValueError, match="changed since snapshot"):
        stage_restore(destination, record["manifest_sha256"], registry.path, tmp_path / "restore")


def test_staged_restore_preserves_every_table_and_generations(runtime, tmp_path, monkeypatch):
    registry = runtime.registry
    registry.disable("model", 1)
    registry.activate("model", "test@1", 2)
    dead(monkeypatch)
    backup = tmp_path / "snapshot"
    result = snapshot(registry.path, backup)
    before = registry.list()
    restored = tmp_path / "staged"
    record = stage_restore(backup, result["manifest_sha256"], registry.path, restored)
    assert record["activated"] is False
    assert digest(restored / "registry.sqlite3") == digest(backup / "registry.sqlite3")
    with (
        closing(connect(restored / "registry.sqlite3")) as copied,
        closing(connect(registry.path)) as source,
    ):
        assert summary(copied) == summary(source) == result["summary"]
    assert registry.list() == before
    reopened = Registry(restored / "registry.sqlite3")
    assert reopened.list()["routes"][0]["generation"] == 3
    assert reopened.inspect("test@1")["digest"] == registry.inspect("test@1")["digest"]
    with pytest.raises(Exception, match="generation"):
        reopened.activate("model", "test@1", 1)
    reopened.close()


@pytest.mark.parametrize("owner_state", ["alive", "unknown"])
def test_stopped_flag_does_not_override_live_or_unknown_process(
    runtime, tmp_path, monkeypatch, owner_state
):
    registry = runtime.registry
    registry.stop_worker()
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: owner_state))
    result = snapshot(registry.path, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="owners_" + owner_state):
        stage_restore(
            tmp_path / "snapshot", result["manifest_sha256"], registry.path, tmp_path / "restore"
        )
    assert not (tmp_path / "restore").exists()


def test_dead_owner_does_not_discard_retained_lease(runtime, tmp_path, monkeypatch):
    registry = runtime.registry
    lease = registry.acquire("model", "retained", None, runtime.backend_identity)
    registry.record_branches(lease.lease_id, ["one"])
    dead(monkeypatch)
    result = snapshot(registry.path, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="leases"):
        stage_restore(
            tmp_path / "snapshot", result["manifest_sha256"], registry.path, tmp_path / "restore"
        )
    assert registry.has_lease(lease.lease_id)
    assert not (tmp_path / "restore").exists()
    registry.release(lease.lease_id)


@pytest.mark.parametrize("change", ["database", "manifest", "extra", "symlink"])
def test_snapshot_tampering_is_rejected(runtime, tmp_path, change):
    directory = tmp_path / "snapshot"
    record = snapshot(runtime.registry.path, directory)
    if change == "database":
        with sqlite3.connect(directory / "registry.sqlite3") as db:
            db.execute("UPDATE routes SET generation=98")
    elif change == "manifest":
        (directory / "manifest.json").write_text("{}")
    elif change == "extra":
        (directory / "registry.sqlite3-wal").write_bytes(b"unexpected")
    else:
        (directory / "registry.sqlite3").unlink()
        (directory / "registry.sqlite3").symlink_to(runtime.registry.path)
    with pytest.raises(ValueError):
        verify_snapshot(directory, record["manifest_sha256"])


def test_unknown_schema_is_not_silently_migrated(runtime, tmp_path):
    with runtime.registry._connection() as db:
        db.execute("CREATE TABLE future_data(value TEXT)")
    with pytest.raises(ValueError, match="schema"):
        snapshot(runtime.registry.path, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot/manifest.json").exists()
    with runtime.registry._connection() as db:
        assert db.execute("SELECT name FROM sqlite_schema WHERE name='future_data'").fetchone()


def test_missing_source_and_existing_destinations_are_preserved(runtime, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="existing regular"):
        snapshot(tmp_path / "missing.db", tmp_path / "bad")
    assert not (tmp_path / "missing.db").exists() and not (tmp_path / "bad").exists()
    directory = tmp_path / "snapshot"
    record = snapshot(runtime.registry.path, directory)
    with pytest.raises(FileExistsError):
        snapshot(runtime.registry.path, directory)
    dead(monkeypatch)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "preserve").write_text("original")
    with pytest.raises(FileExistsError):
        stage_restore(directory, record["manifest_sha256"], runtime.registry.path, destination)
    assert (destination / "preserve").read_text() == "original"


def test_cli_snapshot_and_verify_do_not_register_an_owner(runtime, tmp_path):
    runner = CliRunner()
    with runtime.registry._connection() as db:
        before = db.execute("SELECT count(*) FROM owners").fetchone()[0]
    result = runner.invoke(
        app, ["registry", "snapshot", str(runtime.registry.path), str(tmp_path / "snapshot")]
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    checked = runner.invoke(
        app, ["registry", "verify-snapshot", str(tmp_path / "snapshot"), record["manifest_sha256"]]
    )
    assert checked.exit_code == 0, checked.output
    with runtime.registry._connection() as db:
        assert db.execute("SELECT count(*) FROM owners").fetchone()[0] == before


def test_restore_holds_source_writer_reservation_until_publication(runtime, tmp_path, monkeypatch):
    from jev_runtime import registry_backup

    dead(monkeypatch)
    copied = registry_backup.shutil.copyfileobj
    observations = []

    def contender(reader, writer):
        with sqlite3.connect(runtime.registry.path, timeout=0.01) as connection:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                connection.execute("UPDATE routes SET generation=100")
            observations.append("writer blocked")
        copied(reader, writer)

    result = snapshot(runtime.registry.path, tmp_path / "snapshot")
    monkeypatch.setattr(registry_backup.shutil, "copyfileobj", contender)
    stage_restore(
        tmp_path / "snapshot",
        result["manifest_sha256"],
        runtime.registry.path,
        tmp_path / "restore",
    )
    assert observations == ["writer blocked"]
    assert runtime.registry.list()["routes"][0]["generation"] == 1


def test_failed_receipt_write_leaves_incomplete_copy_without_changing_source(
    runtime, tmp_path, monkeypatch
):
    from jev_runtime import registry_backup

    dead(monkeypatch)
    result = snapshot(runtime.registry.path, tmp_path / "snapshot")
    before = runtime.registry.list()

    def fail(path, value):
        raise OSError("injected publication failure")

    monkeypatch.setattr(registry_backup, "save", fail)
    with pytest.raises(OSError, match="publication failure"):
        stage_restore(
            tmp_path / "snapshot",
            result["manifest_sha256"],
            runtime.registry.path,
            tmp_path / "restore",
        )
    assert runtime.registry.list() == before
    assert not (tmp_path / "restore/restore.json").exists()
