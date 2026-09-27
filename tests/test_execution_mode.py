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


def test_mode_binding_preserves_legacy_and_invalidates_old_calibration(bundle, question, compiler):
    assert "batch_invariant" not in bundle.model_dump(mode="json")["model"]
    assert Bundle.model_validate(bundle.model_dump()).digest == bundle.digest
    settings = Settings(backend="vllm", model_id="fixture", model_revision="a" * 40)
    assert model_identity(settings, compiler).batch_invariant is None
    assert (
        model_identity(
            settings.model_copy(update={"batch_invariant": True}), compiler
        ).batch_invariant
        is True
    )
    original = bundle.model_copy(update={"candidate_policy": "fixed", "questions": (question,)})
    changed = original.model_copy(
        update={"model": original.model.model_copy(update={"batch_invariant": True})}
    )
    assert changed.digest != original.digest
    assert changed.scoring_contract_digest != original.scoring_contract_digest
    calibrated = changed.model_dump(mode="json")
    calibrated["calibration"] = Calibration(
        method="temperature",
        contract_digest=original.scoring_contract_digest,
        dataset_digest="data",
        report_digest="report",
    ).model_dump()
    with pytest.raises(ValidationError, match="different scoring contract"):
        Bundle.model_validate(calibrated)


@pytest.mark.parametrize(
    "expected,observed", [(True, False), (True, None), (None, True), (False, True)]
)
async def test_start_rejects_mode_mismatch_before_canary(
    tmp_path, compiler, bundle, expected, observed
):
    engine = ControlledEngine()

    async def probe():
        return Capabilities(
            engine="vllm",
            version="0.30.0",
            model_id="fixture",
            model_dtype="bfloat16",
            readout_dtype="bfloat16",
            batch_invariant=observed,
        )

    engine.probe = probe
    model = bundle.model.model_copy(
        update={"readout_dtype": "bfloat16", "batch_invariant": expected}
    )
    runtime = Runtime(
        engine, compiler, Registry(tmp_path / "mode.db"), "mode", "fixture", expected_model=model
    )
    try:
        with pytest.raises(JevError) as error:
            await runtime.start()
        assert error.value.code == "engine_execution_mismatch"
        assert not engine.calls
    finally:
        await runtime.close()


async def test_prepare_refuses_unbound_bundle_on_invariant_runtime(runtime, bundle):
    runtime.expected_model = bundle.model.model_copy(update={"batch_invariant": True})
    runtime.backend.calls.clear()
    old = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(old)
    with pytest.raises(JevError) as error:
        await runtime.prepare(old.reference)
    assert error.value.code == "model_mismatch" and not runtime.backend.calls
    right = bundle.model_copy(update={"version": 3, "model": runtime.expected_model})
    runtime.registry.upload(right)
    await runtime.prepare(right.reference)
    assert runtime.is_prepared(right.reference) and runtime.backend.calls


async def test_native_modes_are_reported_and_not_added_to_frozen_lora_profile():
    model = SimpleNamespace(
        model_path="fixture",
        context_len=2048,
        dtype="torch.bfloat16",
        quantization=None,
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        max_logprobs=128,
        logprobs_mode="raw_logprobs",
    )
    args = SimpleNamespace(
        enable_deterministic_inference=True,
        enable_lora=True,
        tokenizer_worker_num=1,
        tp_size=1,
        dp_size=1,
        pp_size=1,
    )
    sg = SGLangNative(SimpleNamespace(model_config=model, server_args=args), "0.5.19")
    vv = VLLMNative(
        SimpleNamespace(model_config=model), "fixture", 2048, "0.30.0", batch_invariant=True
    )
    for backend in (sg, vv):
        backend.managed_lora = True
        caps = await backend.probe()
        assert caps.batch_invariant is True and caps.lora is False and caps.verified is False


@pytest.mark.parametrize("observed", [True, False, None])
async def test_attached_sglang_reports_mode_without_guessing(observed):
    async def reply(request):
        return httpx.Response(200, json={"enable_deterministic_inference": observed})

    client = httpx.AsyncClient(transport=httpx.MockTransport(reply), base_url="http://fixture")
    backend = SGLangHTTP("http://fixture", "fixture", client=client)
    try:
        assert (await backend.probe()).batch_invariant is observed
    finally:
        await backend.close()
