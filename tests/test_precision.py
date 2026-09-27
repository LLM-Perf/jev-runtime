from types import SimpleNamespace

import httpx
import pytest
from conftest import ControlledEngine
from pydantic import ValidationError

from jev_runtime.backends.base import Capabilities
from jev_runtime.backends.sglang import SGLangHTTP, SGLangNative
from jev_runtime.backends.vllm import VLLMNative
from jev_runtime.config import Settings, model_identity
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import Bundle, Calibration


def test_readout_changes_bundle_and_calibration_contract_without_rewriting_legacy(bundle, question):
    legacy = bundle.model_dump(mode="json")
    assert "readout_dtype" not in legacy["model"]
    assert Bundle.model_validate(legacy).digest == bundle.digest
    bf16 = bundle.model_copy(
        update={
            "model": bundle.model.model_copy(update={"readout_dtype": "bfloat16"}),
            "candidate_policy": "fixed",
            "questions": (question,),
        }
    )
    fp32 = bf16.model_copy(
        update={"model": bf16.model.model_copy(update={"readout_dtype": "float32"})}
    )
    assert (
        bf16.digest != fp32.digest and bf16.scoring_contract_digest != fp32.scoring_contract_digest
    )
    calibration = Calibration(
        method="temperature",
        contract_digest=bf16.scoring_contract_digest,
        dataset_digest="data",
        report_digest="report",
    )
    invalid = fp32.model_dump(mode="json")
    invalid["calibration"] = calibration.model_dump(mode="json")
    with pytest.raises(ValidationError, match="different scoring contract"):
        Bundle.model_validate(invalid)


def test_offline_build_binds_default_and_explicit_readout(compiler):
    settings = Settings(backend="sglang", model_id="fixture", model_revision="a" * 40)
    assert model_identity(settings, compiler).readout_dtype == "bfloat16"
    assert (
        model_identity(
            settings.model_copy(update={"readout_dtype": "float32"}), compiler
        ).readout_dtype
        == "float32"
    )


@pytest.mark.parametrize(
    "model_dtype,readout_dtype",
    [("bfloat16", "float32"), ("float16", "bfloat16"), ("bfloat16", None)],
)
async def test_start_rejects_mismatched_or_unreported_engine_precision_before_dispatch(
    tmp_path, compiler, bundle, model_dtype, readout_dtype
):
    engine = ControlledEngine()

    async def probe():
        return Capabilities(
            engine="vllm",
            version="0.30.0",
            model_id="fixture",
            model_dtype=model_dtype,
            readout_dtype=readout_dtype,
        )

    engine.probe = probe
    model = bundle.model.model_copy(update={"readout_dtype": "bfloat16"})
    runtime = Runtime(
        engine,
        compiler,
        Registry(tmp_path / "precision.db"),
        "fixture:precision",
        "fixture",
        expected_model=model,
    )
    try:
        with pytest.raises(JevError) as error:
            await runtime.start()
        assert error.value.code == "engine_precision_mismatch"
        assert not engine.calls and not runtime.control_healthy
    finally:
        await runtime.close()


async def test_prepare_rejects_old_or_wrong_readout_before_canary(runtime, bundle):
    runtime.expected_model = bundle.model.model_copy(update={"readout_dtype": "bfloat16"})
    runtime.backend.calls.clear()
    for version, readout in [(2, None), (3, "float32")]:
        wrong = bundle.model_copy(
            update={
                "version": version,
                "model": bundle.model.model_copy(update={"readout_dtype": readout}),
            }
        )
        runtime.registry.upload(wrong)
        with pytest.raises(JevError) as error:
            await runtime.prepare(wrong.reference)
        assert error.value.code == "model_mismatch"
        assert not runtime.backend.calls
    right = bundle.model_copy(update={"version": 4, "model": runtime.expected_model})
    runtime.registry.upload(right)
    await runtime.prepare(right.reference)
    assert runtime.is_prepared(right.reference) and runtime.backend.calls


@pytest.mark.parametrize("head", ["bfloat16", "float32"])
async def test_native_probes_read_actual_precision_and_restrict_frozen_lora(head):
    model = SimpleNamespace(
        model_path="fixture",
        context_len=2048,
        dtype="torch.bfloat16",
        quantization=None,
        head_dtype="torch." + head,
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        max_logprobs=128,
        logprobs_mode="raw_logprobs",
    )
    args = SimpleNamespace(
        quantization=None,
        enable_fp32_lm_head=head == "float32",
        enable_lora=True,
        tokenizer_worker_num=1,
        tp_size=1,
        pp_size=1,
        dp_size=1,
    )
    sg = SGLangNative(SimpleNamespace(model_config=model, server_args=args), "0.5.19")
    sg.managed_lora = True
    vc = SimpleNamespace(
        model_config=model,
        vllm_config=SimpleNamespace(
            lora_config=object(),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                worker_extension_cls="jev_vllm.worker.LoRAWorkerExtension",
            ),
        ),
    )
    vv = VLLMNative(vc, "fixture", 2048, "0.30.0")
    vv.managed_lora = True
    for adapter in (sg, vv):
        capabilities = await adapter.probe()
        assert capabilities.model_dtype == "bfloat16" and capabilities.readout_dtype == head
        assert capabilities.lora == (head == "bfloat16")


@pytest.mark.parametrize(
    "dtype,fp32,quant,expected",
    [
        ("bfloat16", False, None, "bfloat16"),
        ("auto", False, None, None),
        ("bfloat16", True, None, "float32"),
        ("bfloat16", True, "fp8", None),
    ],
)
async def test_sglang_http_never_guesses_auto_or_quantized_readout(dtype, fp32, quant, expected):
    async def reply(request):
        return httpx.Response(
            200,
            json={"model_path": "fixture"}
            if request.url.path == "/get_model_info"
            else {"dtype": dtype, "enable_fp32_lm_head": fp32, "quantization": quant},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(reply), base_url="http://fixture")
    adapter = SGLangHTTP("http://fixture", "fixture", client=client)
    try:
        assert (await adapter.probe()).readout_dtype == expected
    finally:
        await adapter.close()
