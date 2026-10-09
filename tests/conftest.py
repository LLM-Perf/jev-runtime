import asyncio
import math
import threading

import pytest

from jev_runtime.backends.base import Capabilities, ScoreResult
from jev_runtime.compiler import Compiler
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import Bundle, ModelIdentity, Option, Question
from jev_runtime.shared_admission import SharedAdmission


@pytest.fixture
def engine_owner_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2)
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        assert not thread.is_alive()


class CharacterTokenizer:
    """Deterministic contract fixture; not evidence of an actual model's tokenizer."""

    chat_template = "fixture enable_thinking"
    special_tokens_map = {}

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class ControlledEngine:
    """A cancellation-controllable engine double. Never used by deployment code."""

    def __init__(self):
        self.gate = asyncio.Event()
        self.gate.set()
        self.started = asyncio.Event()
        self.cancelled = []
        self.calls = []
        self.fail_question = None
        self.fail_cancel = False

    async def probe(self):
        return Capabilities(
            engine="fixture",
            version="0",
            model_id="fixture",
            verified=False,
            model_dtype="bfloat16",
            readout_dtype="bfloat16",
        )

    async def score(self, request):
        self.calls.append(request)
        self.started.set()
        await self.gate.wait()
        if request.question_id == self.fail_question:
            raise RuntimeError("injected engine failure")
        weights = [math.exp(-i) for i in range(len(request.label_ids))]
        scores = tuple(math.log(0.8 * weight / sum(weights)) for weight in weights)
        return ScoreResult(request.request_id, scores, len(request.input_ids), 0, 0)

    async def cancel(self, request_id):
        if self.fail_cancel:
            raise RuntimeError("injected abort failure")
        self.cancelled.append(request_id)

    async def close(self):
        pass


@pytest.fixture
def compiler():
    return Compiler(CharacterTokenizer())


@pytest.fixture
def bundle(compiler):
    return Bundle(
        id="test",
        version=1,
        model=ModelIdentity(
            id="fixture",
            revision="a" * 40,
            tokenizer_digest=compiler.tokenizer_digest,
            template_digest=compiler.template_digest,
        ),
    )


@pytest.fixture
def question():
    return Question(
        id="intent",
        type="choice",
        instruction="Choose the relevant category",
        options=(
            Option(id="billing", description="Payment and refunds"),
            Option(id="support", description="Technical assistance"),
        ),
    )


@pytest.fixture
async def runtime(tmp_path, compiler, bundle, request):
    engine = ControlledEngine()
    registry = Registry(tmp_path / "registry.db")
    instance = Runtime(
        engine,
        compiler,
        registry,
        "fixture:0",
        "fixture",
        admission=SharedAdmission(registry, "fixture:0", **getattr(request, "param", {})),
    )
    await instance.start()
    instance.registry.upload(bundle)
    await instance.prepare(bundle.reference)
    instance.registry.activate("model", bundle.reference, 0)
    instance.start_health_monitor()
    engine.calls.clear()
    engine.started.clear()
    yield instance
    await instance.close()
