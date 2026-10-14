"""Run with an isolated installed-wheel Python, without the checkout on sys.path."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
from pathlib import Path


def main():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    from jev_runtime.backends.discovery import backend_inventory

    installed = {}
    for distribution, module in (
        ("jev-runtime-core", "jev_runtime"),
        ("jev-vllm", "jev_vllm"),
        ("jev-sglang", "jev_sglang"),
        ("jev-tokenspeed", "jev_tokenspeed"),
    ):
        dist = metadata.distribution(distribution)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            raise RuntimeError("Smoke test requires wheels, not editable installs")
        path = Path(importlib.import_module(module).__file__).resolve()
        if "site-packages" not in path.parts:
            raise RuntimeError(f"{module} was imported outside site-packages")
        installed[distribution] = dist.version
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, "[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}",
    )
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], return_dict=False
    )
    if tokens != [1]:
        raise RuntimeError("Installed tokenizer did not execute its chat template correctly")
    if {item["name"] for item in backend_inventory()} != {"vllm", "sglang", "tokenspeed"}:
        raise RuntimeError("Installed backend inventory is incomplete")
    print(json.dumps({"passed": True, "installed": installed, "chat_template_tokens": tokens}))


if __name__ == "__main__":
    main()
