"""Controllable CPU-only scoring fixture for HTTP rollout faults, never a model benchmark."""

import asyncio
import os

from jev_runtime.api import create_app
from jev_runtime.backends.base import Capabilities, ScoreResult
from jev_runtime.compiler import Compiler
from jev_runtime.config import load_settings, model_identity
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime


class Tokenizer:
    chat_template = "fixture"
    special_tokens_map = {}

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class Engine:
    async def probe(self):
        return Capabilities(engine="fixture", version="0", model_id="fixture")

    async def score(self, request):
        if request.question_id == "slow":
            await asyncio.sleep(8)
        return ScoreResult(
            request.request_id, tuple(-i - 0.5 for i in range(len(request.label_ids)))
        )

    async def cancel(self, request_id):
        pass

    async def close(self):
        pass


def create():
    settings = load_settings(os.environ["JEV_CONFIG"])
    compiler = Compiler(Tokenizer())
    registry = Registry(settings.registry_path)
    registry.tag_deployment(settings.deployment_id, settings.release_id, settings.workers)
    instance = Runtime(
        Engine(),
        compiler,
        registry,
        "fixture:engine",
        "fixture",
        expected_model=model_identity(settings, compiler),
    )
    return create_app(settings, instance)
