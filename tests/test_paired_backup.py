import json
import sqlite3

import pytest
from jev_tokenspeed.receipts import CompletionReceipts

from jev_runtime.paired_backup import restore_pair, snapshot_pair, verify_pair
from jev_runtime.registry import Registry


def store(runtime):
    return CompletionReceipts(
        runtime.registry.path.with_suffix(".tokenspeed-receipts.db"), "fixture"
    )


def stopped(monkeypatch):
    # Unit-only owner simulation. Not real host/namespace recovery evidence.
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda _: "dead"))


def test_pair_retains_receipts_routes_and_duplicate_protection(runtime, tmp_path, monkeypatch):
    receipts = store(runtime)
    receipts.reserve("one")
    receipts.complete("one")
    stopped(monkeypatch)
    backup = tmp_path / "pair"
    result = snapshot_pair(runtime.registry.path, backup)
    assert verify_pair(backup, result["manifest_sha256"])["receipts_summary"]["rows"] == 1
    destination = tmp_path / "restored"
    restored = restore_pair(backup, result["manifest_sha256"], runtime.registry.path, destination)
    assert restored["activated"] is False
    native = CompletionReceipts(destination / "registry.tokenspeed-receipts.db", "fixture")
    assert native.completed("one")
    with pytest.raises(Exception, match="already dispatched"):
        native.reserve("one")
    registry = Registry(destination / "registry.sqlite3")
    try:
        assert registry.list()["routes"] == runtime.registry.list()["routes"]
    finally:
        registry.close()


@pytest.mark.parametrize("state", ["alive", "unknown", "pending", "lease"])
def test_unsafe_pair_rejected_without_destination(runtime, tmp_path, monkeypatch, state):
    receipts = store(runtime)
    stopped(monkeypatch)
    if state in {"alive", "unknown"}:
        monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda _: state))
    elif state == "pending":
        receipts.reserve("unfinished")
    else:
        lease = runtime.registry.acquire("model", "one", None, runtime.backend_identity)
    with pytest.raises(ValueError, match="not stopped/drained"):
        snapshot_pair(runtime.registry.path, tmp_path / "pair")
    assert not (tmp_path / "pair").exists()
    if state == "lease":
        runtime.registry.release(lease.lease_id)


@pytest.mark.parametrize("change", ["registry", "receipts", "backup"])
def test_changed_or_tampered_pair_rejected(runtime, tmp_path, monkeypatch, change):
    receipts = store(runtime)
    stopped(monkeypatch)
    result = snapshot_pair(runtime.registry.path, tmp_path / "pair")
    if change == "registry":
        runtime.registry.disable("model", 1)
    elif change == "receipts":
        receipts.reserve("later")
        receipts.complete("later")
    else:
        with sqlite3.connect(tmp_path / "pair/receipts.sqlite3") as db:
            db.execute("INSERT INTO receipts VALUES ('x','y','completed')")
    with pytest.raises(ValueError):
        restore_pair(
            tmp_path / "pair", result["manifest_sha256"], runtime.registry.path, tmp_path / "staged"
        )
    assert not (tmp_path / "staged").exists()


def test_incomplete_publication_cannot_start_registry(tmp_path):
    (tmp_path / "pair-restore-intent.json").write_text(
        json.dumps({"database_name": "registry.sqlite3", "operation_id": "incomplete"})
    )
    with pytest.raises(ValueError, match="incomplete"):
        Registry(tmp_path / "registry.sqlite3")
    assert not (tmp_path / "registry.sqlite3").exists()
