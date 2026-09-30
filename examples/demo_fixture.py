"""Offline end-to-end demo: the full Jev serving loop without a GPU.

Runs the real registry, admission control, bundle lifecycle, HTTP gateway and
Python SDK against a deterministic in-process engine double:

    python examples/demo_fixture.py

It demonstrates the serving contract -- typed answers, probability semantics,
usage accounting and a zero-downtime bundle hot-switch. The scores are fixed
per label position, so a passing demo says nothing about model quality,
calibration, GPU compatibility or performance.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import tempfile
from pathlib import Path

import httpx
import uvicorn

from jev_runtime.api import create_app
from jev_runtime.backends.base import Capabilities, ScoreResult
from jev_runtime.compiler import Compiler
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import Bundle, ModelIdentity
from jev_runtime.sdk import AsyncJevClient
from jev_runtime.shared_admission import SharedAdmission

API_KEY = "demo-key"
ADMIN_KEY = "demo-admin-key"
ALIAS = "decision-model"


class DemoTokenizer:
    """Character-level tokenizer; deterministic, not a real model tokenizer."""

    chat_template = "demo"
    special_tokens_map = {}

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class DemoEngine:
    """Deterministic engine double: fixed raw scores per label position."""

    async def probe(self):
        return Capabilities(
            engine="demo",
            version="0",
            model_id="demo",
            model_dtype="bfloat16",
            readout_dtype="bfloat16",
        )

    async def score(self, request):
        weights = [math.exp(-i) for i in range(len(request.label_ids))]
        scores = tuple(math.log(0.8 * weight / sum(weights)) for weight in weights)
        return ScoreResult(request.request_id, scores, len(request.input_ids), 0, 0)

    async def cancel(self, request_id):
        pass

    async def close(self):
        pass


def show(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


async def main() -> None:
    compiler = Compiler(DemoTokenizer())
    identity = ModelIdentity(
        id="demo",
        revision="a" * 40,
        tokenizer_digest=compiler.tokenizer_digest,
        template_digest=compiler.template_digest,
    )
    with tempfile.TemporaryDirectory(prefix="jev-demo-") as tmp:
        registry = Registry(Path(tmp) / "registry.db")
        engine = DemoEngine()
        runtime = Runtime(
            engine,
            compiler,
            registry,
            "demo:0",
            "demo",
            admission=SharedAdmission(registry, "demo:0"),
        )
        app = create_app(instance=runtime, api_key=API_KEY, admin_key=ADMIN_KEY)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        serving = asyncio.create_task(server.serve())
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        if not server.started:
            raise RuntimeError("Demo gateway did not start")
        print(f"Jev gateway (demo backend) listening on {base}")
        try:
            admin = httpx.AsyncClient(
                base_url=base, headers={"Authorization": f"Bearer {ADMIN_KEY}"}
            )
            data = httpx.AsyncClient(base_url=base, headers={"Authorization": f"Bearer {API_KEY}"})
            client = AsyncJevClient(base, api_key=API_KEY)
            try:
                for version in (1, 2):
                    bundle = Bundle(id="support-intent", version=version, model=identity)
                    for step, body in (
                        ("/admin/bundles", bundle.model_dump(mode="json")),
                        ("/admin/bundles/prepare", {"reference": bundle.reference}),
                        (
                            "/admin/bundles/activate",
                            {
                                "alias": ALIAS,
                                "reference": bundle.reference,
                                "expected_generation": version - 1,
                            },
                        ),
                    ):
                        response = await admin.post(step, json=body)
                        response.raise_for_status()
                    ready = (await data.get("/ready")).json()
                    show(f"bundle {bundle.reference} active, readiness", ready)

                    request = json.loads(Path(__file__).with_name("request.json").read_text())
                    decision = await client.decide(request)
                    show(f"decision on generation {decision.generation}", decision.model_dump())
            finally:
                await client.close()
                await data.aclose()
                await admin.aclose()
        finally:
            server.should_exit = True
            await serving
            await runtime.close()
    print(
        "\nDemo backend scores are deterministic per label position. "
        "This exercises the serving contract only; it is not model-quality, "
        "GPU-compatibility or performance evidence."
    )


if __name__ == "__main__":
    asyncio.run(main())
