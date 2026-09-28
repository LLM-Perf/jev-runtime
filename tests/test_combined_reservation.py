"""Atomic lease/admission boundary and speculative compilation race coverage."""

import multiprocessing
import os
import sqlite3
import sys

import pytest

from jev_runtime.errors import JevError
from jev_runtime.registry import Registry, StaleRoute
from jev_runtime.schema import DecisionRequest, Policy, TextInput
from tests.test_shared_admission import setup_pair


def body(question, **kwargs):
    return DecisionRequest(
        model="model", input=TextInput(text="refund"), questions=(question,), **kwargs
    )


def counts(path):
    with sqlite3.connect(path) as db:
        return tuple(
            db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            for table in ("leases", "lease_work", "lease_tenants", "admission_tickets")
        )


@pytest.mark.parametrize("failure", ["lease", "journal", "admission", None])
def test_combined_commit_visibility_and_rollback(tmp_path, bundle, monkeypatch, failure):
    a, b, pool, peer = setup_pair(tmp_path, bundle)
    resolved = a.resolve("model", None, "fixture:0")
    original_pin, original_journal, original_admit = a._acquire, a._record_branches, pool._try_admit

    def checkpoint(name):
        assert counts(b.path) == (0, 0, 0, 0)
        if failure == name:
            raise RuntimeError("injected " + name)

    def pin(*args, **kwargs):
        result = original_pin(*args, **kwargs)
        checkpoint("lease")
        return result

    def journal(*args, **kwargs):
        result = original_journal(*args, **kwargs)
        checkpoint("journal")
        return result

    def admit(*args, **kwargs):
        result = original_admit(*args, **kwargs)
        checkpoint("admission")
        return result

    monkeypatch.setattr(a, "_acquire", pin)
    monkeypatch.setattr(a, "_record_branches", journal)
    monkeypatch.setattr(pool, "_try_admit", admit)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="injected"):
                pool.reserve(resolved, "model", "atomic", None, "default", 9, ["b", "a"])
            assert counts(b.path) == (0, 0, 0, 0)
        else:
            snapshot, admitted = pool.reserve(
                resolved, "model", "atomic", None, "default", 9, ["b", "a"]
            )
            assert admitted and counts(b.path) == (1, 1, 1, 1)
            with sqlite3.connect(b.path) as db:
                assert db.execute("SELECT branches FROM lease_work").fetchone()[0] == '["a", "b"]'
            assert peer.snapshot()["expanded_branches"] == 2
            a.release(snapshot.lease_id)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("switch_back", [False, True])
async def test_switch_during_compile_recompiles_with_authoritative_lease(
    runtime, bundle, question, monkeypatch, switch_back
):
    second = bundle.model_copy(update={"version": 2, "policy": Policy(min_probability=0.99)})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    runtime.backend.calls.clear()
    compiled_versions = []
    compile_request = runtime._compile_request

    def compile_then_switch(request, selected, engine_id):
        result = compile_request(request, selected, engine_id)
        compiled_versions.append(selected.reference)
        if len(compiled_versions) == 1:
            assert counts(runtime.registry.path) == (0, 0, 0, 0)
            runtime.registry.activate("model", second.reference, 1)
            if switch_back:
                runtime.registry.activate("model", bundle.reference, 2)
            else:
                # No GPU work was authorized by the tentative old compilation.
                runtime.registry.retire(bundle.reference)
        else:
            assert counts(runtime.registry.path) == (1, 1, 1, 0)
        return result

    monkeypatch.setattr(runtime, "_compile_request", compile_then_switch)
    result = await runtime.decide(body(question))
    assert compiled_versions == ["test@1", "test@1" if switch_back else "test@2"]
    assert result.generation == (3 if switch_back else 2)
    assert result.bundle == compiled_versions[-1]
    assert result.bundle_digest == (bundle if switch_back else second).digest
    assert result.answers[question.id].abstained is (not switch_back)
    assert len(runtime.backend.calls) == 1
    assert counts(runtime.registry.path) == (0, 0, 0, 0)


@pytest.mark.parametrize("action", ["disable", "switch_explicit"])
async def test_stale_route_cannot_dispatch(runtime, bundle, question, monkeypatch, action):
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    runtime.backend.calls.clear()
    compile_request = runtime._compile_request

    def compile_then_switch(*args):
        result = compile_request(*args)
        if action == "disable":
            runtime.registry.disable("model", 1)
        else:
            runtime.registry.activate("model", second.reference, 1)
        return result

    monkeypatch.setattr(runtime, "_compile_request", compile_then_switch)
    with pytest.raises(JevError) as exc:
        await runtime.decide(body(question, bundle=bundle.reference))
    assert exc.value.code == ("route_unavailable" if action == "disable" else "bundle_not_active")
    assert not runtime.backend.calls
    assert counts(runtime.registry.path) == (0, 0, 0, 0)


async def test_speculative_validation_error_is_rechecked_under_lease(
    runtime, question, monkeypatch
):
    original = runtime._compile_request
    attempts = []

    def compile_request(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise JevError("bundle_not_ready", "Simulate stale preparation observation", 503)
        assert counts(runtime.registry.path) == (1, 1, 1, 0)
        return original(*args)

    monkeypatch.setattr(runtime, "_compile_request", compile_request)
    assert (await runtime.decide(body(question))).status == "completed"
    assert len(attempts) == 2 and len(runtime.backend.calls) == 1


def crash_after_reservation(path, manifest):
    from jev_runtime.schema import Bundle
    from tests.test_shared_admission import serving

    registry = Registry(path)
    pool = serving(registry, {})
    bundle = Bundle.model_validate_json(manifest)
    registry.upload(bundle)
    _, lease = registry.begin_prepare(bundle.reference, "fixture:0", "prepare")
    registry.finish_prepare(bundle.reference, lease)
    registry.record_worker_prepared(bundle.reference)
    registry.activate("model", bundle.reference, 0)
    resolved = registry.resolve("model", None, "fixture:0")
    pool.reserve(resolved, "model", "crashed", None, "tenant", 9, ["branch-a", "branch-b"])
    os._exit(23)  # No finally/connection close and no dispatch.


def test_process_exit_after_combined_commit_retains_recoverable_work(tmp_path, bundle):
    path = str(tmp_path / "crash.db")
    process = multiprocessing.get_context("spawn").Process(
        target=crash_after_reservation, args=(path, bundle.model_dump_json())
    )
    process.start()
    process.join(10)
    try:
        assert process.exitcode == 23
        assert counts(path) == (1, 1, 1, 1)
        observer = Registry(path)
        try:
            candidate = observer.recovery_candidates()[0]
            if sys.platform == "linux":
                assert candidate["owner_status"] == "dead"
                assert observer.recovery_snapshot("crashed", "fixture:0") == candidate
            else:
                assert candidate["owner_status"] == "unknown"
                with pytest.raises(JevError) as exc:
                    observer.recovery_snapshot("crashed", "fixture:0")
                assert exc.value.code == "recovery_not_confirmed"
            assert candidate["engine_request_ids"] == ["branch-a", "branch-b"]
            with sqlite3.connect(path) as db:
                assert db.execute(
                    "SELECT state,tokens,branches FROM admission_tickets"
                ).fetchone() == ("ADMITTED", 9, 2)
                assert db.execute("SELECT tenant FROM lease_tenants").fetchone()[0] == "tenant"
            # This test inspects retention; recovery must still confirm engine cancellation.
        finally:
            observer.close()
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)


def test_changed_digest_rejects_forged_compilation_input(tmp_path, bundle):
    from jev_runtime.registry import ResolvedRoute

    a, b, pool, _ = setup_pair(tmp_path, bundle)
    try:
        candidate = ResolvedRoute(
            bundle.model_copy(update={"policy": Policy(min_probability=0.99)}), 1
        )
        with pytest.raises(StaleRoute):
            pool.reserve(candidate, "model", "mismatch", None, "default", 9, ["a"])
        assert counts(b.path) == (0, 0, 0, 0)
    finally:
        a.close()
        b.close()
