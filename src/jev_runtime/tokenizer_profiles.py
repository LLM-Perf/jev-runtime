"""Offline tokenizer artifacts; checkpoint Python is never imported or copied."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import random
import re
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


def publish_profile(destination: Path, files: dict[str, bytes], manifest: dict) -> None:
    """An interrupted publication must never look like an ordinary checkpoint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    pending = destination / "jev-tokenizer-profile.pending"
    with pending.open("xb") as output:
        output.write(b"Profile publication is incomplete; do not serve this directory.\n")
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
    with (destination / "jev-tokenizer-profile.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    pending.unlink()


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
        publish_profile(
            destination, {p.name: p.read_bytes() for p in sorted(staging.iterdir())}, manifest
        )
    return {"path": str(destination), **manifest}


FAST_PROFILE_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
}


def fast_profile_file(name: str) -> bool:
    """Only data files understood by the standard local fast-tokenizer loader."""
    return name in FAST_PROFILE_FILES or bool(
        re.fullmatch(r"additional_chat_templates/[A-Za-z0-9_-]+\.jinja", name)
    )


def fast_source_files(source: Path) -> dict[str, bytes]:
    files = {
        name: (source / name).read_bytes()
        for name in sorted(FAST_PROFILE_FILES)
        if (source / name).exists()
    }
    if not {"tokenizer.json", "tokenizer_config.json"} <= files.keys():
        raise ValueError("preserve-fast requires tokenizer.json and tokenizer_config.json")
    for path in sorted((source / "additional_chat_templates").glob("*.jinja")):
        name = path.relative_to(source).as_posix()
        if not fast_profile_file(name):
            raise ValueError("Unsupported serialized chat-template filename")
        files[name] = path.read_bytes()
    return files


def preserve_fast_tokenizer(
    model_dir: Path,
    output_dir: Path,
    *,
    chat_template: Path | None = None,
    chat_template_sha256: str | None = None,
) -> dict:
    """Bind a serialized Rust tokenizer without model-specific reconstruction.

    The reference is the serialized tokenizer.json, not an arbitrary Python
    wrapper. No checkpoint code runs. The original data and model stay unchanged.
    """
    source = model_dir.resolve(strict=True)
    destination = output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a new output directory; profiles cannot be overwritten")
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Tokenizer output must be outside the original checkpoint directory")
    if bool(chat_template) != bool(chat_template_sha256):
        raise ValueError("Supply chat-template and chat-template-sha256 together")
    original = fast_source_files(source)
    files = dict(original)
    config = json.loads(files["tokenizer_config.json"])
    if not isinstance(config, dict):
        raise ValueError("Tokenizer configuration must be a JSON object")
    if config.get("split_special_tokens", False):
        raise ValueError("Split-special-token mode is not supported by this profile")
    # Select exactly the serialized tokenizer.json, never a class reconstruction,
    # alternate versioned tokenizer, remote implementation or source path.
    removed = {
        "auto_map",
        "tokenizer_class",
        "tokenizer_file",
        "fast_tokenizer_files",
        "vocab_file",
        "merges_file",
        "vocab",
        "merges",
        "tokenizer_object",
        "__slow_tokenizer",
        "from_slow",
    }
    exported = {key: value for key, value in config.items() if key not in removed}
    exported["tokenizer_class"] = "PreTrainedTokenizerFast"
    override = None
    if chat_template is not None:
        raw = chat_template.read_bytes()
        if sha256(raw) != chat_template_sha256 or not raw.decode("utf-8").strip():
            raise ValueError("Explicit chat template is empty or its SHA256 differs")
        override = {"path": str(chat_template.resolve()), "sha256": sha256(raw)}
        # An explicit override replaces the entire default/named template set.
        files = {k: v for k, v in files.items() if not k.endswith(".jinja")}
        exported.pop("chat_template", None)
        files["chat_template.jinja"] = raw
    files["tokenizer_config.json"] = (
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install jev-runtime-core[tokenizers]") from exc
    reference = Tokenizer.from_str(files["tokenizer.json"].decode("utf-8"))
    raw_backend = json.loads(reference.to_str())
    if reference.padding is not None or reference.truncation is not None:
        raise ValueError("Serialized padding/truncation must be disabled for this text profile")
    if raw_backend["model"].get("dropout") not in (None, 0, 0.0):
        raise ValueError("Stochastic BPE dropout is unsupported by a deterministic profile")
    with tempfile.TemporaryDirectory(prefix="jev-preserve-fast-") as temporary:
        staging = Path(temporary)
        for name, data in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        loaded = AutoTokenizer.from_pretrained(
            staging, trust_remote_code=False, local_files_only=True
        )
        if json.loads(loaded.backend_tokenizer.to_str()) != raw_backend:
            raise ValueError("Standard fast loader changed the serialized tokenizer pipeline")
        from jev_runtime.compiler import Compiler

        # Reject a library that silently ignores a declared named/default file.
        # Files take precedence over config exactly as in the standard loader.
        declared = {}
        if "chat_template.jinja" in files:
            declared["default"] = files["chat_template.jinja"].decode("utf-8")
        for name, data in files.items():
            if name.startswith("additional_chat_templates/"):
                declared[Path(name).stem] = data.decode("utf-8")
        if not declared:
            declared = exported.get("chat_template")
            if isinstance(declared, list):
                declared = {entry["name"]: entry["template"] for entry in declared}
            if isinstance(declared, str):
                declared = {"default": declared}
        actual_templates = loaded.chat_template
        if isinstance(actual_templates, str):
            actual_templates = {"default": actual_templates}
        if not declared or actual_templates != declared or "default" not in declared:
            raise ValueError("Standard loader changed templates or no default template is declared")
        compiler = Compiler(loaded)
        tokens = {t["content"]: t["id"] for t in raw_backend["added_tokens"]}
        texts = validation_texts(tokens) + ["9 ²,abc1①", "I need a refund for order 9 ²,abc1①."]
        texts += [
            prefix + token + suffix
            for token in list(tokens)[:64]
            for prefix, suffix in [("x ", " y"), ("\t", "\n")]
        ]
        templates = loaded.chat_template
        templates = templates if isinstance(templates, dict) else {"default": templates}
        rendered = []
        for name, template in templates.items():
            if not isinstance(template, str) or not template.strip():
                raise ValueError("Serialized chat templates must be nonempty strings")
            for text in ("Hello", "退款 9 ²,abc1①", "\t\n", "😀"):
                for generation in (False, True):
                    prompt = loaded.apply_chat_template(
                        [{"role": "user", "content": text}],
                        chat_template=template,
                        tokenize=False,
                        add_generation_prompt=generation,
                    )
                    texts.append(prompt)
                    rendered.append(
                        (name, sha256(text.encode()), generation, sha256(prompt.encode()))
                    )
        ledger = []
        for text in texts:
            for add in (False, True):
                expected = reference.encode(text, add_special_tokens=add).ids
                actual = loaded.encode(text, add_special_tokens=add)
                if expected != actual:
                    raise ValueError("Fast profile failed exact serialized input-ID validation")
                for skip in (False, True):
                    if loaded.decode(
                        actual, skip_special_tokens=skip, clean_up_tokenization_spaces=False
                    ) != reference.decode(expected, skip_special_tokens=skip):
                        raise ValueError("Fast profile failed serialized decoding validation")
                ledger.append((sha256(text.encode()), add, expected))
        if json.loads(loaded.backend_tokenizer.to_str()) != raw_backend:
            raise ValueError("Validation mutated the serialized tokenizer pipeline")
        manifest = {
            "schema_version": 1,
            "kind": "serialized-fast-v1",
            "implementation_sha256": sha256(Path(__file__).read_bytes()),
            "scope": (
                "serialized tokenizer.json single text and standard HF rendered chat; "
                "no custom Python wrapper, text-pair or multimodal equivalence claim"
            ),
            "source_path": str(source),
            "source_tokenizer_class": config.get("tokenizer_class"),
            "removed_config_keys": sorted(config.keys() & removed),
            "source_files": {
                name: {"sha256": sha256(data), "size_bytes": len(data)}
                for name, data in original.items()
            },
            "explicit_chat_template": override,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
            "validation": {
                "passed": True,
                "cases": len(ledger),
                "decode_checks": len(ledger) * 2,
                "template_renders": len(rendered),
                "full_backend_equal": True,
                "ledger_sha256": sha256(json.dumps(ledger).encode()),
                "render_ledger_sha256": sha256(json.dumps(rendered).encode()),
            },
            "tokenizer_digest": compiler.tokenizer_digest,
            "tokenizer_implementation_digest": compiler.tokenizer_implementation_digest,
            "template_digest": compiler.template_digest,
            "files": {
                name: {"sha256": sha256(data), "size_bytes": len(data)}
                for name, data in sorted(files.items())
            },
        }
        if fast_source_files(source) != original:
            raise ValueError("Source tokenizer changed during profile creation")
        if chat_template is not None and sha256(chat_template.read_bytes()) != chat_template_sha256:
            raise ValueError("Explicit chat template changed during profile creation")
        publish_profile(destination, files, manifest)
    return {"path": str(destination), **manifest}


def verify_tokenizer_profile(directory: Path, compiler=None) -> dict | None:
    """Verify a generated profile; ordinary checkpoint directories remain valid."""
    from jev_runtime.errors import JevError

    pending = directory / "jev-tokenizer-profile.pending"
    if pending.exists() or pending.is_symlink():
        raise JevError(
            "tokenizer_profile_mismatch", "Tokenizer profile publication is incomplete", 409
        )
    path = directory / "jev-tokenizer-profile.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
        if manifest["schema_version"] != 1 or manifest["kind"] not in {
            "glm4-tiktoken-fast-v1",
            "serialized-fast-v1",
        }:
            raise ValueError("Unsupported tokenizer profile")
        required = {"tokenizer.json", "tokenizer_config.json"}
        names = set(manifest["files"])
        if not required <= names or any(not fast_profile_file(name) for name in names):
            raise ValueError("Tokenizer profile file set differs")
        if manifest["kind"] == "glm4-tiktoken-fast-v1" and names != required:
            raise ValueError("GLM4 tokenizer profile file set differs")
        actual = set()
        for item in directory.rglob("*"):
            if item.is_symlink():
                raise ValueError("Tokenizer profile contains a symlink")
            if item.is_file():
                actual.add(item.relative_to(directory).as_posix())
        if actual != names | {"jev-tokenizer-profile.json"}:
            raise ValueError("Tokenizer profile contains missing or unexpected files")
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
