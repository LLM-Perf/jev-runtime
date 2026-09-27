import json
from types import SimpleNamespace

import pytest

from tests.integration import run_native_validation as runner


def test_dead_or_reused_engine_fails_before_readiness_request(monkeypatch):
    record = {"identity": {"pid": 123, "start_ticks": "10"}, "port": 18795}
    for identity in (None, {"pid": 123, "start_ticks": "11"}):
        monkeypatch.setattr(
            runner.dsw_service, "process_identity", lambda pid, value=identity: value
        )
        with pytest.raises(RuntimeError, match="identity changed"):
            runner.wait_ready(record, "test", 5)


def test_rank_evidence_rejects_restart_or_missing_worker():
    assert runner.worker_ranks("(EngineCore pid=123) world_size=1 rank=0", 1) == {0: 123}
    log = "\n".join(f"(Worker_TP{rank} pid={123 + rank})" for rank in range(4))
    assert runner.worker_ranks(log + "\n" + log, 4) == dict(enumerate(range(123, 127)))
    for invalid in (log.replace("Worker_TP3", "Worker_TP2"), log + "\n(Worker_TP3 pid=999)"):
        with pytest.raises(RuntimeError, match="distinct live worker"):
            runner.worker_ranks(invalid, 4)


def test_cleanup_retains_orphan_group_and_never_claims_exit(monkeypatch, tmp_path):
    record = {"identity": {"pid": 123}, "gpus_before": [{"index": 4, "uuid": "gpu4"}]}
    # The API parent may have exited while its worker remains alive. The existing
    # stop operation does not signal an unverified parent/group in this case.
    monkeypatch.setattr(runner.dsw_service, "stop", lambda args: None)
    monkeypatch.setattr(runner, "group_members", lambda pgid: [456])
    monkeypatch.setattr(runner.dsw_service, "get_gpu", lambda index: {"uuid": "gpu4"})
    result = runner.cleanup(tmp_path, record, timeout=0)
    assert not result["passed"]
    assert result["remaining_process_group_members"] == [456]
    assert json.loads((tmp_path / "cleanup.json").read_text()) == result


def test_failure_after_launch_still_cleans_up_and_preserves_attempt(monkeypatch, tmp_path):
    run = tmp_path / "attempt"
    args = SimpleNamespace(
        run_dir=run,
        source_commit="a" * 40,
        runtime_source_commit="b" * 40,
        engine="vllm",
        model_path=tmp_path,
        tokenizer_path=None,
        gpus="3,4,5,6",
        port=18795,
        memory_fraction=0.065,
        reserve_mib=3072,
        readout_dtype="model",
        startup_timeout=5,
        chat_template_path=None,
    )
    record = {"identity": {"pid": 123}}

    def launch(*unused):
        runner.save(run / "process.json", record)
        runner.save(run / "keys.json", {"api": "test"})
        return 0

    def not_ready(*unused):
        raise TimeoutError("fixture startup deadline")

    cleaned = []
    monkeypatch.setattr(runner, "execute", launch)
    monkeypatch.setattr(runner, "wait_ready", not_ready)
    monkeypatch.setattr(
        runner, "cleanup", lambda path, value: cleaned.append(value) or {"passed": True}
    )
    report = runner.validate(args)
    assert not report["passed"] and report["cleanup_passed"]
    assert report["failure"]["stage"] == "readiness"
    assert cleaned == [record]
    assert json.loads((run / "validation.json").read_text()) == report
    with pytest.raises(FileExistsError):
        runner.validate(args)
    assert cleaned == [record]
