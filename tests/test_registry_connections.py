import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_shared_connection_rolls_back_and_preserves_full_durability(runtime):
    registry = runtime.registry
    with registry._connection() as first:
        assert first.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert first.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with pytest.raises(RuntimeError, match="injected"):
        with registry._transaction() as db:
            db.execute("UPDATE routes SET generation=99")
            raise RuntimeError("injected")
    assert registry.list()["routes"][0]["generation"] == 1
    with registry._connection() as next_connection:
        assert next_connection is first and not next_connection.in_transaction
    registry.close()
    with registry._connection() as reopened:
        assert reopened is not first
        assert reopened.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_one_owner_serializes_threaded_transactions(runtime):
    def operation(index):
        snapshot = runtime.registry.acquire("model", str(index), None, runtime.backend_identity)
        runtime.registry.record_branches(snapshot.lease_id, [f"branch-{index}"])
        runtime.registry.release(snapshot.lease_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(operation, range(80)))
    assert not runtime.registry.list()["leases"]


def test_unclosed_wal_connection_retains_committed_dispatch_after_process_exit(runtime):
    script = """
import json, os, sys
from jev_runtime.registry import Registry
registry = Registry(sys.argv[1])
snapshot = registry.acquire('model', 'abrupt-exit', None, sys.argv[2])
registry.record_branches(snapshot.lease_id, ['jev-durable.q.0'])
print(json.dumps({'owner': registry.owner, 'lease_id': snapshot.lease_id}), flush=True)
os._exit(17)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(runtime.registry.path), runtime.backend_identity],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 17, process.stderr
    result = json.loads(process.stdout)
    row = runtime.registry.recovery_candidates()[0]
    assert row["lease_id"] == result["lease_id"]
    assert row["engine_request_ids"] == ["jev-durable.q.0"]
    assert row["phase"] == "inflight"
