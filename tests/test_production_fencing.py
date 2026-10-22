import json
import sqlite3
from contextlib import closing

import pytest
from jev_tokenspeed.receipts import CompletionReceipts

from jev_runtime.errors import JevError
from jev_runtime.owner_fence import (
    FENCE_ACK,
    fence_stale_owners,
    fenced_owner,
    maintenance_owner_blockers,
    observation,
)
from jev_runtime.paired_backup import restore_pair, snapshot_pair
from jev_runtime.receipt_retirement import RETIRE_ACK, retire_completed_receipts
from jev_runtime.registry import Registry
from jev_runtime.registry_backup import connect
from jev_runtime.registry_schema import initialize, register_owner


def fresh(tmp_path):
    path = tmp_path / "registry.sqlite3"
    db = sqlite3.connect(path)
    try:
        initialize(db)
    finally:
        db.close()
    return path


def add_owner(path, owner, identity):
    db = sqlite3.connect(path)
    try:
        db.execute("PRAGMA foreign_keys=ON")
        register_owner(db, owner, identity)
    finally:
        db.close()


def lost_identity():
    return {
        "host": "other-host",
        "pid": 1,
        "boot_id": "boot-other",
        "start_ticks": "10",
        "pid_namespace": "pid:[other]",
    }


def test_fence_refuses_live_owner_and_writes_nothing(tmp_path):
    path = fresh(tmp_path)
    registry = Registry(path)
    try:
        with pytest.raises(ValueError, match="alive"):
            fence_stale_owners(path, "lost_host", FENCE_ACK)
        with closing(connect(path, writable=True)) as db:
            count = db.execute(
                "SELECT count(*) FROM events WHERE action='deployment_fence'"
            ).fetchone()[0]
            assert count == 0
    finally:
        registry.close()


def test_lost_host_fence_is_explicit_idempotent_and_blocks_old_generation(tmp_path):
    path = fresh(tmp_path)
    add_owner(path, "old", lost_identity())
    with pytest.raises(ValueError, match="acknowledgement"):
        fence_stale_owners(path, "lost_host", "yes")
    with pytest.raises(ValueError, match="container_namespace"):
        fence_stale_owners(path, "container_namespace", FENCE_ACK)
    first = fence_stale_owners(path, "lost_host", FENCE_ACK)
    second = fence_stale_owners(path, "lost_host", FENCE_ACK)
    assert first["idempotent"] is False and first["generation"] == 1
    assert second["idempotent"] is True and second["generation"] == 1
    with closing(connect(path, writable=True)) as db:
        assert maintenance_owner_blockers(db) == {}
        stored = json.loads(db.execute("SELECT identity FROM owners").fetchone()[0])
        assert fenced_owner(db, stored, observation())
    current = Registry(path)
    try:
        assert current._status(stored) == "fenced"
        with sqlite3.connect(path) as db:
            db.execute(
                "INSERT INTO events(ts,action,details) VALUES(?,?,?)",
                (1, "deployment_fence", json.dumps({"generation": 2})),
            )
        with pytest.raises(JevError) as exc:
            current.stop_worker()
        assert exc.value.code == "owner_fenced"
        with pytest.raises(JevError) as ready:
            current.require_backend_open("fixture")
        assert ready.value.code == "owner_fenced"
        replacement = Registry(path)
        try:
            replacement.stop_worker()
        finally:
            replacement.close()
    finally:
        current.close()


def test_fenced_inflight_lease_is_recoverable_and_not_a_restore_blocker(tmp_path):
    path = fresh(tmp_path)
    add_owner(path, "old", lost_identity())
    with closing(connect(path, writable=True)) as db:
        db.execute(
            "INSERT INTO bundles VALUES(?,?,?,?,?,?,?)",
            ("model@1", "d" * 64, "{}", "VALIDATED", "fixture", None, 1),
        )
        db.execute(
            "INSERT INTO leases VALUES(?,?,?,?,?)", ("lease", "model@1", "old", "request", 1)
        )
        db.execute("INSERT INTO lease_work VALUES(?,?,'inflight')", ("lease", "[]"))
        db.commit()
    fence_stale_owners(path, "lost_host", FENCE_ACK)
    registry = Registry(path)
    try:
        candidate = registry.recovery_candidates()[0]
        assert candidate["owner_status"] == "fenced" and candidate["recoverable"] is True
    finally:
        registry.close()


def test_container_namespace_and_unverifiable_identity_use_distinct_reasons(tmp_path):
    path = fresh(tmp_path)
    observed = observation()
    add_owner(
        path,
        "moved",
        {
            "host": observed["host"],
            "pid": 1,
            "boot_id": observed["boot_id"],
            "start_ticks": "1",
            "pid_namespace": "pid:[replaced-container]",
        },
    )
    with pytest.raises(ValueError, match="lost_host"):
        fence_stale_owners(path, "lost_host", FENCE_ACK)
    assert fence_stale_owners(path, "container_namespace", FENCE_ACK)["fenced_owners"] == 1

    blank_dir = tmp_path / "blank"
    blank_dir.mkdir()
    other = fresh(blank_dir)
    add_owner(other, "blank", {"host": observed["host"], "pid": 4})
    fenced = fence_stale_owners(other, "unverifiable_identity", FENCE_ACK)
    assert fenced["reason"] == "unverifiable_identity"


def test_receipt_retirement_keeps_pending_rows_and_rejects_reuse(tmp_path):
    path = fresh(tmp_path)
    add_owner(path, "old", lost_identity())
    fence_stale_owners(path, "lost_host", FENCE_ACK)
    store = CompletionReceipts(path.with_suffix(".tokenspeed-receipts.db"), "model")
    store.reserve("pending-one")
    store.reserve("done")
    store.complete("done")
    store.reserve("done-two")
    store.complete("done-two")
    with pytest.raises(ValueError, match="acknowledgement"):
        retire_completed_receipts(path, "model", 10, "delete")
    with closing(connect(path, writable=True)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute(
            "INSERT INTO bundles VALUES(?,?,?,?,?,?,?)",
            ("model@1", "d" * 64, "{}", "VALIDATED", "fixture", None, 1),
        )
        db.execute("INSERT INTO leases VALUES(?,?,?,?,?)", ("lease", "model@1", "old", "done", 1))
        db.commit()
    with pytest.raises(ValueError, match="not stopped"):
        retire_completed_receipts(path, "model", 10, RETIRE_ACK)
    assert store.completed("done")
    with closing(connect(path, writable=True)) as db:
        db.execute("DELETE FROM leases")
        db.commit()
    first = retire_completed_receipts(path, "model", 1, RETIRE_ACK)
    assert first["retired"] == 1 and first["remaining_completed"] == 1 and first["pending"] == 1
    assert not store.completed("done") and store.completed("done-two")
    with pytest.raises(JevError, match="already dispatched"):
        store.reserve("done")
    assert store.completed("pending-one") is False
    second = retire_completed_receipts(path, "model", 10, RETIRE_ACK)
    assert second["retired"] == 1 and second["pending"] == 1
    with pytest.raises(ValueError, match="No completed"):
        retire_completed_receipts(path, "model", 10, RETIRE_ACK)
    native = sqlite3.connect(path.with_suffix(".tokenspeed-receipts.db"))
    try:
        assert native.execute("SELECT state FROM receipts").fetchall() == [("pending",)]
        assert native.execute("SELECT count(*) FROM retired_receipts").fetchone()[0] == 2
    finally:
        native.close()


def test_retired_receipt_round_trips_through_paired_snapshot(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda _: "dead"))
    store = CompletionReceipts(
        runtime.registry.path.with_suffix(".tokenspeed-receipts.db"), "fixture"
    )
    store.reserve("one")
    store.complete("one")
    retired = retire_completed_receipts(runtime.registry.path, "fixture", 8, RETIRE_ACK)
    assert retired["retired"] == 1
    backup = tmp_path / "pair"
    result = snapshot_pair(runtime.registry.path, backup)
    restored = restore_pair(
        backup, result["manifest_sha256"], runtime.registry.path, tmp_path / "restored"
    )
    native = CompletionReceipts(tmp_path / "restored/registry.tokenspeed-receipts.db", "fixture")
    assert restored["activated"] is False and not native.completed("one")
    with pytest.raises(JevError, match="already dispatched"):
        native.reserve("one")
