"""Installed-wheel smoke: synthetic tokenizer profile preservation and rendering.

Runs against the installed wheel only (python -I from outside the checkout), so
the repository cannot shadow the package. The tokenizer is trained locally; no
models are downloaded. Fails when the tokenizers extra omits the chat-template
rendering support (jinja2) that Transformers apply_chat_template requires.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

TEMPLATE = "{{ bos_token }}{% for m in messages %}{{m.content}}{% endfor %}"


def synthetic_checkpoint(path: Path) -> Path:
    path.mkdir()
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["abc1① 9 ²,abc1① Hello 退款 12345 😀\n\t"],
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<s>", "</s>"],
        ),
    )
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 0)]
    )
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "GPT2Tokenizer",
                "auto_map": {"AutoTokenizer": ["poison.Tokenizer", None]},
                "bos_token": "<s>",
                "eos_token": "</s>",
                "chat_template": TEMPLATE,
            }
        )
    )
    (path / "poison.py").write_text("raise RuntimeError('checkpoint code must not execute')")
    return path


def main() -> None:
    from jev_runtime.config import Settings, load_compiler, model_identity
    from jev_runtime.schema import Bundle, Option, Policy, Question, TemplateSpec
    from jev_runtime.tokenizer_profiles import preserve_fast_tokenizer, verify_tokenizer_profile

    with tempfile.TemporaryDirectory(prefix="jev-packaging-smoke-") as work:
        source = synthetic_checkpoint(Path(work) / "source")
        destination = Path(work) / "profile"
        manifest = preserve_fast_tokenizer(source, destination)
        assert manifest["kind"] == "serialized-fast-v1"
        assert manifest["validation"]["passed"]
        settings = Settings(
            backend="vllm",
            model_id="packaging-smoke",
            model_revision="a" * 40,
            tokenizer=str(destination),
        )
        compiler = load_compiler(settings)
        verify_tokenizer_profile(destination, compiler)
        bundle = Bundle(
            id="packaging-smoke",
            version=1,
            model=model_identity(settings, compiler),
            template=TemplateSpec(mode="joint-label", use_system_role=True),
            policy=Policy(max_options=8),
        )
        question = Question(
            id="choice",
            type="choice",
            instruction="Select the matching category.",
            options=(
                Option(id="a", description="Refund request"),
                Option(id="b", description="Other topic"),
            ),
        )
        compiled = compiler.compile("A refund is requested.", question, bundle, "smoke")
        assert len(compiled.sequences) == 1
        sequence = compiled.sequences[0]
        assert sequence.input_ids and len(sequence.label_ids) == 2
        print(
            json.dumps(
                {
                    "passed": True,
                    "profile_kind": manifest["kind"],
                    "template_renders": manifest["validation"]["template_renders"],
                    "sequence_tokens": len(sequence.input_ids),
                    "label_ids": list(sequence.label_ids),
                }
            )
        )


if __name__ == "__main__":
    main()
