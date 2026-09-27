import pytest

from deployment import dsw_service


def test_explicit_gpu_mapping_preserves_order_and_rejects_aliases():
    assert dsw_service.device_indices(7, None) == [7]
    assert dsw_service.device_indices(7, "6, 7") == [6, 7]
    for invalid in ("", "7,", "6,6", "-1,7", "GPU-unknown"):
        with pytest.raises(ValueError):
            dsw_service.device_indices(7, invalid)


@pytest.mark.parametrize("engine,fraction", [("vllm", 0.1), ("sglang", 0.85)])
def test_insufficient_nonprimary_gpu_blocks_launch(engine, fraction, monkeypatch):
    def gpu(index):
        return {
            "uuid": f"gpu-{index}",
            "total_mib": "80000",
            "free_mib": "12000" if index == 6 else "5000",
        }

    monkeypatch.setattr(dsw_service, "get_gpu", gpu)
    with pytest.raises(SystemExit, match="GPU 7"):
        dsw_service.check_gpu_budgets([6, 7], engine, fraction, 1000)


def test_budget_uses_each_device_memory_denominator_and_records_mapping(monkeypatch):
    def gpu(index):
        return {"uuid": f"gpu-{index}", "total_mib": "80000", "free_mib": "12000"}

    monkeypatch.setattr(dsw_service, "get_gpu", gpu)
    result = dsw_service.check_gpu_budgets([6, 7], "sglang", 0.65, 3072)
    assert [x["index"] for x in result] == [6, 7]
    with pytest.raises(SystemExit):
        dsw_service.check_gpu_budgets([6, 7], "vllm", 0.65, 3072)
    monkeypatch.setattr(dsw_service, "get_gpu", lambda index: gpu(6))
    with pytest.raises(SystemExit, match="distinct GPUs"):
        dsw_service.check_gpu_budgets([6, 7], "sglang", 0.65, 3072)


@pytest.mark.parametrize(
    "engine,flag", [("vllm", "--tensor-parallel-size"), ("sglang", "--tp-size")]
)
@pytest.mark.parametrize("readout", ["model", "float32"])
@pytest.mark.parametrize("explicit_template", [False, True])
@pytest.mark.parametrize("explicit_tokenizer", [False, True])
def test_launch_binds_explicit_devices_to_engine_tp_and_saved_manifest(
    engine, flag, readout, explicit_template, explicit_tokenizer, monkeypatch, tmp_path
):
    import hashlib
    import json
    from types import SimpleNamespace

    model = tmp_path / "model"
    model.mkdir()
    (model / "jev-source.json").write_text(
        json.dumps({"model_id": "fixture", "revision": "a" * 40})
    )
    args = SimpleNamespace(
        run_dir=tmp_path / "run",
        model_path=model,
        tokenizer_path=None,
        gpu=7,
        gpus="6,7",
        engine=engine,
        gateway=False,
        port=0,
        memory_fraction=0.07,
        reserve_mib=3072,
        engine_url=None,
        api_workers=1,
        admission_config=None,
        health_config=None,
        tenant=[],
        adapters_root=None,
        engine_run_dir=None,
        readout_dtype=readout,
        chat_template_path=None,
        chat_template_sha256=None,
        chat_template_format="json",
    )
    if explicit_template:
        args.chat_template_path = tmp_path / "template.json"
        raw = json.dumps({"chat_template": "{{ messages[0].content }}"}).encode()
        args.chat_template_path.write_bytes(raw)
        args.chat_template_sha256 = hashlib.sha256(raw).hexdigest()
    if explicit_tokenizer:
        args.tokenizer_path = tmp_path / "tokenizer-profile"
        args.tokenizer_path.mkdir()
    monkeypatch.setattr(
        dsw_service,
        "get_gpu",
        lambda index: {
            "uuid": f"gpu-{index}",
            "total_mib": "80000",
            "free_mib": "12000",
        },
    )
    captured = {}

    def start(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(dsw_service.subprocess, "Popen", start)
    monkeypatch.setattr(dsw_service, "process_identity", lambda pid: {"pid": pid})
    dsw_service.launch(args)
    command = captured["command"]
    assert command[command.index(flag) + 1] == "2"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert captured["start_new_session"]
    record = json.loads((args.run_dir / "process.json").read_text())
    assert record["tensor_parallel_size"] == 2
    assert [gpu["index"] for gpu in record["gpus_before"]] == [6, 7]
    assert record["gpu_before"]["uuid"] == "gpu-6"
    expected = "bfloat16" if readout == "model" else readout
    assert record["readout_dtype"] == expected
    tokenizer = args.tokenizer_path if explicit_tokenizer else model
    assert record["tokenizer_path"] == str(tokenizer)
    assert json.loads((args.run_dir / "config.json").read_text())["tokenizer"] == str(tokenizer)
    tokenizer_flag = "--tokenizer" if engine == "vllm" else "--tokenizer-path"
    assert (tokenizer_flag in command) == explicit_tokenizer
    if explicit_tokenizer:
        assert command[command.index(tokenizer_flag) + 1] == str(tokenizer)
    assert json.loads((args.run_dir / "config.json").read_text())["readout_dtype"] == expected
    assert ("--chat-template" in command) == explicit_template
    if explicit_template:
        snapshot = record["chat_template_snapshot"]
        assert command[command.index("--chat-template") + 1] == snapshot["path"]
        assert record["chat_template_source"]["sha256"] == args.chat_template_sha256
        assert snapshot["sha256"] == hashlib.sha256(b"{{ messages[0].content }}").hexdigest()
        assert (args.run_dir / "chat_template.jinja").read_text() == "{{ messages[0].content }}"
    if engine == "vllm":
        assert ("--hf-overrides" in command) == (readout == "float32")
        if readout == "float32":
            assert json.loads(command[command.index("--hf-overrides") + 1]) == {
                "head_dtype": "float32"
            }
    else:
        assert ("--enable-fp32-lm-head" in command) == (readout == "float32")
