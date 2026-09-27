"""Offline tokenizer artifacts; checkpoint Python is never imported or copied."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import random
import tempfile
from importlib.metadata import version
from pathlib import Path

GLM4_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
GLM4_PREFIX = ("[gMASK]", "<sop>")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_glm4_inputs(model_dir: Path) -> tuple[dict, dict[bytes, int], dict[str, int], dict]:
    names = ("config.json", "tokenizer_config.json", "tokenizer.model")
    data = {name: (model_dir / name).read_bytes() for name in names}
    model, config = (json.loads(data[name]) for name in names[:2])
    if model.get("model_type") != "chatglm" or config.get("tokenizer_class") != "ChatGLM4Tokenizer":
        raise ValueError("This converter requires the GLM4 tiktoken tokenizer profile")
    if not isinstance(config.get("chat_template"), str) or not config["chat_template"].strip():
        raise ValueError("GLM4 requires a nonempty serialized chat template")
    ranks = {}
    for line in data["tokenizer.model"].splitlines():
        token, rank = line.split()
        token = base64.b64decode(token, validate=True)
        if token in ranks:
            raise ValueError("Duplicate byte token in tiktoken vocabulary")
        ranks[token] = int(rank)
    if set(ranks.values()) != set(range(len(ranks))) or not all(
        bytes([i]) in ranks for i in range(256)
    ):
        raise ValueError("Tiktoken ranks must be contiguous and cover every byte")
    decoder = config.get("added_tokens_decoder", {})
    if not decoder:
        raise ValueError("GLM4 special-token metadata is missing")
    special = {}
    for key, token in sorted(decoder.items(), key=lambda pair: int(pair[0])):
        if (
            token.get("special") is not True
            or any(
                token.get(flag, False) is not False
                for flag in ("lstrip", "rstrip", "normalized", "single_word")
            )
            or not isinstance(token.get("content"), str)
            or not token["content"]
            or token["content"] in special
        ):
            raise ValueError("Unsupported or duplicate GLM4 special-token behavior")
        special[token["content"]] = int(key)
    if list(special.values()) != list(range(len(ranks), len(ranks) + len(special))):
        raise ValueError("Special-token IDs must immediately follow the byte vocabulary")
    if not all(token in special for token in GLM4_PREFIX):
        raise ValueError("GLM4 prefix tokens are missing")
    if set(config.get("additional_special_tokens", ())) != set(special):
        raise ValueError("GLM4 special-token declarations disagree")
    if config.get("split_special_tokens", False):
        raise ValueError("Split-special-token mode is not supported by this profile")
    files = {name: {"sha256": sha256(raw), "size_bytes": len(raw)} for name, raw in data.items()}
    return config, ranks, special, files


def validation_texts(special: dict[str, int]) -> list[str]:
    atoms = [
        "abc",
        "退款",
        "你好",
        " ",
        "  ",
        "\n",
        "\r\n",
        "\t",
        "1234567",
        "0.005",
        "😀",
        "é",
        "e\u0301",
        "\x00",
        "\u200b",
        *special,
    ]
    rng = random.Random(1931)
    return [
        "",
        "Hello world",
        "  Hello\tworld\r\n",
        "退款问题、账户密码。",
        *special,
        *("".join(rng.choices(atoms, k=rng.randint(1, 30))) for _ in range(256)),
    ]


def convert_glm4_tokenizer(model_dir: Path, output_dir: Path) -> dict:
    """Publish a new standard fast-tokenizer profile for single text prompts.

    Model weights and the source directory stay untouched. Existing outputs are
    rejected. Only a complete manifest is a successful profile; publication I/O
    failures can leave an incomplete directory, which must not be served.
    """
    source = model_dir.resolve(strict=True)
    destination = output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a new output directory; profiles cannot be overwritten")
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Tokenizer output must be outside the original checkpoint directory")
    config, ranks, special, inputs = read_glm4_inputs(source)
    try:
        import tiktoken
        from tokenizers.processors import TemplateProcessing
        from transformers import AutoTokenizer
        from transformers.convert_slow_tokenizer import TikTokenConverter
    except ImportError as exc:
        raise ImportError("Install jev-runtime-core[tokenizer-conversion]") from exc
    parameter = next(
        (
            name
            for name in ("extra_special_tokens", "additional_special_tokens")
            if name in inspect.signature(TikTokenConverter).parameters
        ),
        None,
    )
    if parameter is None:
        raise ValueError("This Transformers tiktoken converter interface is unsupported")
    backend = TikTokenConverter(
        vocab_file=str(source / "tokenizer.model"),
        pattern=GLM4_PATTERN,
        **{parameter: list(special)},
    ).converted()
    if any(backend.token_to_id(token) != rank for token, rank in special.items()):
        raise ValueError("Conversion changed a special-token ID")
    backend.post_processor = TemplateProcessing(
        single="[gMASK] <sop> $A",
        special_tokens=[(token, special[token]) for token in GLM4_PREFIX],
    )
    exported = {
        key: value for key, value in config.items() if key not in {"auto_map", "tokenizer_class"}
    }
    exported["tokenizer_class"] = "PreTrainedTokenizerFast"
    exported["model_input_names"] = ["input_ids", "attention_mask"]
    reference = tiktoken.Encoding(
        name="jev-glm4-reference",
        pat_str=GLM4_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=special,
    )
    with tempfile.TemporaryDirectory(prefix="jev-tokenizer-") as temporary:
        staging = Path(temporary)
        backend.save(str(staging / "tokenizer.json"))
        (staging / "tokenizer_config.json").write_text(
            json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        loaded = AutoTokenizer.from_pretrained(
            staging, trust_remote_code=False, local_files_only=True
        )
        texts = validation_texts(special)
        for text in texts[:20]:
            texts.append(
                loaded.apply_chat_template(
                    [
                        {"role": "system", "content": "Classify the supplied text."},
                        {"role": "user", "content": text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        prefix = [special[token] for token in GLM4_PREFIX]
        ledger = []
        for text in texts:
            ids = reference.encode(text, allowed_special="all")
            for add in (False, True):
                expected = prefix + ids if add else ids
                actual = loaded.encode(text, add_special_tokens=add)
                if actual != expected:
                    raise ValueError(
                        "Converted tokenizer failed exact tiktoken input-ID validation"
                    )
                if loaded.decode(
                    actual, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ) != reference.decode(actual):
                    raise ValueError("Converted tokenizer failed byte decoding validation")
                ledger.append((sha256(text.encode()), add, expected))
        # The fingerprint includes the complete loaded normalization/BPE pipeline.
        from jev_runtime.compiler import Compiler

        compiler = Compiler(loaded)
        manifest = {
            "schema_version": 1,
            "kind": "glm4-tiktoken-fast-v1",
            "implementation_sha256": sha256(Path(__file__).read_bytes()),
            "scope": (
                "single UTF-8 text and rendered chat prompts; "
                "no text-pair or multimodal certification"
            ),
            "source_path": str(source),
            "source_files": inputs,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
            "tiktoken_version": version("tiktoken"),
            "vocabulary_size": len(ranks),
            "special_token_ids": special,
            "validation": {
                "passed": True,
                "cases": len(ledger),
                "ledger_sha256": sha256(json.dumps(ledger).encode()),
            },
            "tokenizer_digest": compiler.tokenizer_digest,
            "tokenizer_implementation_digest": compiler.tokenizer_implementation_digest,
            "template_digest": compiler.template_digest,
            "files": {
                path.name: {"sha256": sha256(path.read_bytes()), "size_bytes": path.stat().st_size}
                for path in sorted(staging.iterdir())
            },
        }
        # Detect source changes during conversion before creating a public output.
        if read_glm4_inputs(source)[3] != inputs:
            raise ValueError("Source tokenizer changed during conversion")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()  # Exclusive creation; a racing invocation cannot overwrite it.
        for path in sorted(staging.iterdir()):
            with (destination / path.name).open("xb") as output:
                output.write(path.read_bytes())
        with (destination / "jev-tokenizer-profile.json").open("x", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
    return {"path": str(destination), **manifest}


def verify_tokenizer_profile(directory: Path, compiler=None) -> dict | None:
    """Verify a generated profile; ordinary checkpoint directories remain valid."""
    from jev_runtime.errors import JevError

    path = directory / "jev-tokenizer-profile.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
        if manifest["schema_version"] != 1 or manifest["kind"] != "glm4-tiktoken-fast-v1":
            raise ValueError("Unsupported tokenizer profile")
        if set(manifest["files"]) != {"tokenizer.json", "tokenizer_config.json"}:
            raise ValueError("Tokenizer profile file set differs")
        for name, expected in manifest["files"].items():
            data = (directory / name).read_bytes()
            if len(data) != expected["size_bytes"] or sha256(data) != expected["sha256"]:
                raise ValueError("Tokenizer profile bytes differ")
        if manifest["validation"]["passed"] is not True:
            raise ValueError("Tokenizer profile validation was not successful")
        if compiler is not None:
            for name in ("tokenizer_digest", "tokenizer_implementation_digest"):
                if getattr(compiler, name) != manifest[name]:
                    raise ValueError("Host tokenizer differs from the generated profile")
            if (
                compiler.explicit_chat_template is None
                and compiler.template_digest != manifest["template_digest"]
            ):
                raise ValueError("Host template differs from the generated profile")
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        raise JevError("tokenizer_profile_mismatch", str(exc), 409) from exc
    return manifest
