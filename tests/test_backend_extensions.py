import asyncio
import json
import math
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from jev_tokenspeed.backend import (
    TokenSpeedNative,
    generate_payload,
    parse_result,
    validate_profile,
)
from jev_tokenspeed.plugin import verify_source
from typer.testing import CliRunner

from jev_runtime.backends import discovery
from jev_runtime.backends.base import ScoreInput, ScoreResult
from jev_runtime.backends.remote import ScoringHTTP
from jev_runtime.cli import app
from jev_runtime.compiler import split_label_sequences
from jev_runtime.errors import JevError
from jev_runtime.schema import DecisionRequest, Option, Policy, Question
from jev_runtime.scoring import assemble


def request():
    return ScoreInput("r", "q", (1, 2, 3), (10,))


def native_output(rid="r"):
    return {
        "output_ids": [10],
        "meta_info": {
            "id": rid,
            "finish_reason": {"type": "length", "length": 1},
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "cached_tokens": 2,
            "output_token_logprobs": [[-2.0, 10, None]],
        },
    }


def profile():
    return SimpleNamespace(
        sampling_backend="triton_full",
        enable_output_logprobs=True,
        enforce_eager=True,
        quantization=None,
        speculative_algorithm=None,
        speculative_config=None,
        disaggregation_mode="null",
        pipeline_parallel_size=1,
        mapping=SimpleNamespace(world_size=1),
        rl_control_port=None,
        numerics="auto",
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
    )


def test_installed_tokenspeed_discovery_is_lazy():
    before = set(sys.modules)
    assert any(row["name"] == "tokenspeed" for row in discovery.backend_inventory())
    result = CliRunner().invoke(app, ["backend", "list"])
    assert result.exit_code == 0 and "tokenspeed" in result.stdout
    assert not any(
        name == "tokenspeed" or name.startswith("tokenspeed.") for name in set(sys.modules) - before
    )


def test_backend_discovery_rejects_unknown_and_shadowed_providers(monkeypatch):
    settings = SimpleNamespace(backend="missing")
    monkeypatch.setattr(discovery, "entry_points", lambda **kw: [])
    with pytest.raises(JevError, match="Install"):
        discovery.create_backend(settings, None)
    called = []
    entry = SimpleNamespace(name="sglang", load=lambda: called.append(True))
    monkeypatch.setattr(discovery, "entry_points", lambda **kw: [entry])
    with pytest.raises(JevError) as exc:
        discovery.create_backend(SimpleNamespace(backend="sglang"), None)
    assert exc.value.code == "backend_conflict" and not called
    entry.name = "external"
    monkeypatch.setattr(discovery, "entry_points", lambda **kw: [entry, entry])
    with pytest.raises(JevError):
        discovery.create_backend(SimpleNamespace(backend="external"), None)


@pytest.mark.parametrize("kind", ["choice", "boolean", "rank", "score"])
@pytest.mark.parametrize("mode", ["joint-label", "independent-candidate"])
def test_lowering_preserves_readout_and_exposes_every_label(compiler, bundle, kind, mode):
    bundle = bundle.model_copy(
        update={"template": bundle.template.model_copy(update={"mode": mode})}
    )
    options = (
        ()
        if kind == "boolean"
        else (
            Option(id="a", description="A", value=0 if kind == "score" else None),
            Option(id="b", description="B", value=1 if kind == "score" else None),
            Option(id="c", description="C", value=2 if kind == "score" else None),
        )
    )
    question = Question(id="q", type=kind, instruction="Classify", options=options)
    original = compiler.compile("data", question, bundle, "root")
    lowered = split_label_sequences(original)
    expected = [
        ScoreResult(seq.request_id, tuple(-2.0 - i for i in range(len(seq.label_ids))))
        for seq in original.sequences
    ]
    actual = []
    offset = 0
    for seq, scores in zip(original.sequences, expected, strict=True):
        for i, label in enumerate(seq.label_ids):
            leaf = lowered.sequences[offset]
            assert leaf.input_ids == seq.input_ids and leaf.label_ids == (label,)
            actual.append(ScoreResult(leaf.request_id, (scores.logprobs[i],)))
            offset += 1
    assert assemble(original, expected, bundle) == assemble(lowered, actual, bundle)
    ids = [seq.request_id for seq in lowered.sequences]
    assert len(ids) == len(set(ids)) and len({len(rid) for rid in ids}) == 1


async def test_single_label_runtime_accounts_and_limits_expansion(runtime, bundle, question):
    original_score = runtime.backend.score

    async def score(sequence):
        result = await original_score(sequence)
        return replace(result, logprobs=(-2.0,), completion_tokens=1)

    runtime.backend.score = score
    runtime.capabilities = runtime.capabilities.model_copy(
        update={"label_scoring": "single", "max_label_tokens": 1}
    )
    body = DecisionRequest.model_validate(
        {"model": "model", "input": {"text": "data"}, "questions": [question.model_dump()]}
    )
    compiled, sequences, total = runtime._compile_request(body, bundle, "root")
    assert len(sequences) == 2 and total == sum(len(s.input_ids) for s in sequences)
    strict = bundle.model_copy(update={"policy": Policy(max_scoring_sequences=1)})
    with pytest.raises(JevError) as exc:
        runtime._validate_sequences(sequences, strict)
    assert exc.value.code == "branch_budget"
    strict = bundle.model_copy(
        update={"policy": Policy(max_expanded_tokens=len(sequences[0].input_ids))}
    )
    with pytest.raises(JevError) as exc:
        runtime._validate_sequences(sequences, strict)
    assert exc.value.code == "expanded_token_budget"
    result = await runtime.decide(body)
    assert result.usage.scoring_sequences == 2
    assert result.usage.logical_prompt_tokens == total
    assert result.usage.engine_completion_tokens == 2
    assert len(runtime.backend.calls) == 2
    assert not runtime.registry.recovery_candidates()


def test_tokenspeed_payload_and_usage():
    payload = generate_payload(request())
    assert "token_ids_logprob" not in payload
    assert payload["sampling_params"]["logit_bias"] == {"10": 16384.0}
    assert payload["sampling_params"]["max_new_tokens"] == 1
    result = parse_result(native_output(), request())
    assert result.logprobs == (-2.0,)
    assert result.prompt_tokens == 3 and result.completion_tokens == 1
    with pytest.raises(JevError):
        generate_payload(replace(request(), label_ids=(10, 11)))


@pytest.mark.parametrize(
    "change",
    [
        {"output_token_logprobs": [[0.1, 10, None]]},
        {"output_token_logprobs": [[math.nan, 10, None]]},
        {"output_token_logprobs": [[-2, 11, None]]},
        {"output_token_logprobs": []},
        {"output_token_logprobs": [[True, 10, None]]},
        {"completion_tokens": 0},
        {"completion_tokens": True},
        {"prompt_tokens": 4},
        {"cached_tokens": 4},
        {"id": "other"},
        {"finish_reason": {"type": "abort"}},
    ],
)
def test_tokenspeed_rejects_wrong_or_missing_evidence(change):
    output = native_output()
    output["meta_info"].update(change)
    with pytest.raises(JevError):
        parse_result(output, request())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_backend", "triton"),
        ("enable_output_logprobs", False),
        ("enforce_eager", False),
        ("quantization", "fp8"),
        ("speculative_algorithm", "EAGLE3"),
        ("speculative_config", "{}"),
        ("disaggregation_mode", "prefill"),
        ("pipeline_parallel_size", 2),
        ("mapping", SimpleNamespace(world_size=2)),
        ("rl_control_port", 30001),
        ("numerics", "rl-bitwise"),
        ("skip_tokenizer_init", True),
    ],
)
def test_tokenspeed_profile_guards(field, value):
    args = profile()
    validate_profile(args)
    setattr(args, field, value)
    with pytest.raises(JevError):
        validate_profile(args)


async def test_tokenspeed_cancellation_waits_for_terminal_and_retains_unknown(monkeypatch):
    module = ModuleType("tokenspeed.runtime.engine.io_struct")
    module.GenerateReqInput = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    started, terminal = asyncio.Event(), asyncio.Event()

    async def generate(obj):
        started.set()
        await terminal.wait()
        yield native_output(obj.rid)

    async def submit(coro):
        return await coro

    backend = TokenSpeedNative(SimpleNamespace(generate_request=generate), "fixture", submit)
    caller = asyncio.create_task(backend.score(request()))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    cleanup = asyncio.create_task(backend.cancel("r"))
    await asyncio.sleep(0)
    assert not cleanup.done() and "r" in backend._attempts
    terminal.set()
    await cleanup
    assert not backend._attempts
    with pytest.raises(JevError) as exc:
        await backend.cancel("unknown")
    assert exc.value.code == "cancellation_unconfirmed"


async def test_tokenspeed_transport_loss_is_not_confirmed_drain(monkeypatch):
    module = ModuleType("tokenspeed.runtime.engine.io_struct")
    module.GenerateReqInput = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    async def generate(obj):
        raise ConnectionError("lost engine")
        yield  # pragma: no cover

    async def submit(coro):
        return await coro

    backend = TokenSpeedNative(SimpleNamespace(generate_request=generate), "fixture", submit)
    with pytest.raises(ConnectionError):
        await backend.score(request())
    with pytest.raises(JevError):
        await backend.cancel("r")
    assert "r" in backend._attempts


def test_source_guard_rejects_unpinned_engine(tmp_path):
    with pytest.raises(JevError) as exc:
        verify_source(tmp_path)
    assert exc.value.code == "tokenspeed_source_mismatch"


@pytest.mark.parametrize("bad", ["identity", "nan", "positive", "processed", "arity", "usage"])
async def test_remote_contract_rejects_bad_scores(bad):
    data = {"request_id": "r", "logprobs": [-2.0], "raw_logprobs": True}
    data.update(
        {
            "identity": {"request_id": "other"},
            "nan": {"logprobs": ["NaN"]},
            "positive": {"logprobs": [1]},
            "processed": {"raw_logprobs": False},
            "arity": {"logprobs": [-1, -2]},
            "usage": {"prompt_tokens": True},
        }[bad]
    )
    async with httpx.AsyncClient(
        base_url="http://fixture",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=data)),
    ) as client:
        backend = ScoringHTTP("http://fixture", "tokenspeed", client=client)
        with pytest.raises(JevError):
            await backend.score(request())


async def test_remote_probe_identity_and_explicit_cancel_ack():
    async def handler(req):
        if req.url.path.endswith("scoring-capabilities"):
            return httpx.Response(
                200,
                json={"engine": "sglang", "version": "0", "model_id": "fixture", "api_workers": 1},
            )
        assert json.loads(req.content) == {"request_id": "r"}
        return httpx.Response(200, json={"cancelled": False})

    async with httpx.AsyncClient(
        base_url="http://fixture", transport=httpx.MockTransport(handler)
    ) as client:
        backend = ScoringHTTP("http://fixture", "tokenspeed", client=client)
        with pytest.raises(JevError) as exc:
            await backend.probe()
        assert exc.value.code == "engine_mismatch"
        with pytest.raises(JevError) as exc:
            await backend.cancel("r")
        assert exc.value.code == "cancellation_unconfirmed"


async def test_tokenspeed_native_app_serves_typed_results_with_owned_engine_loop(
    monkeypatch, tmp_path, compiler
):
    import threading

    from jev_tokenspeed import plugin

    from jev_runtime.config import Settings

    manager = SimpleNamespace(
        server_args=profile(),
        model_config=SimpleNamespace(dtype="bfloat16", model_path="fixture"),
        context_len=4096,
        tokenizer=compiler.tokenizer,
    )
    manager.server_args.model = "fixture"
    manager.server_args.revision = "a" * 40
    calls = []

    async def generate(obj):
        assert threading.current_thread() is not threading.main_thread()
        calls.append(obj)
        label = int(next(iter(obj.sampling_params["logit_bias"])))
        output = native_output(obj.rid)
        output["output_ids"] = [label]
        output["meta_info"].update(
            prompt_tokens=len(obj.input_ids),
            cached_tokens=0,
            output_token_logprobs=[[-2.0 - label / 1000, label, None]],
        )
        yield output

    manager.generate_request = generate
    closed = []
    engine = SimpleNamespace(
        tokenizer_manager=manager,
        llm=SimpleNamespace(run=asyncio.run),
        shutdown=lambda: closed.append(True),
    )
    modules = {
        "tokenspeed": {"__file__": str(tmp_path / "__init__.py")},
        "tokenspeed.runtime.entrypoints.engine": {"Engine": lambda **kw: engine},
        "tokenspeed.runtime.utils.server_args": {"ServerArgs": lambda **kw: manager.server_args},
        "tokenspeed.version": {"__version__": "fixture"},
        "tokenspeed.runtime.engine.io_struct": {
            "GenerateReqInput": lambda **kw: SimpleNamespace(**kw)
        },
    }
    for name, values in modules.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(plugin, "verify_source", lambda root: "a" * 40)
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_ADMIN_KEY", "test-admin")
    monkeypatch.delenv("JEV_CONFIG", raising=False)
    settings = Settings(
        backend="tokenspeed",
        model_id="fixture",
        model_revision="a" * 40,
        registry_path=str(tmp_path / "registry.sqlite3"),
        bootstrap_alias="model",
    )
    application = plugin.create_app(settings, {})
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client:
            response = await client.post(
                "/plugins/jev-runtime/v1/decisions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "model",
                    "input": {"text": "ready"},
                    "questions": [{"id": "q", "type": "boolean", "instruction": "Contains ready?"}],
                },
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["engine"]["name"] == "tokenspeed"
            assert result["usage"]["scoring_sequences"] == 2
            assert result["usage"]["engine_completion_tokens"] == 2
            assert len(calls) == 4  # two preparation labels, two serving labels
    assert closed == [True]
