import base64
import json

import pytest
import tiktoken
from tokenizers import normalizers
from transformers import AutoTokenizer
from typer.testing import CliRunner

from jev_runtime.cli import app
from jev_runtime.compiler import Compiler
from jev_runtime.config import (
    Settings,
    build_runtime,
    compiler_for_tokenizer,
    load_compiler,
    model_identity,
)
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, Option, Question
from jev_runtime.tokenizer_profiles import GLM4_PATTERN, convert_glm4_tokenizer, read_glm4_inputs


def checkpoint(path):
    path.mkdir()
    vocab = [bytes([i]) for i in range(256)] + [b"He", b"Hel", b"Hell", b"Hello"]
    (path / "tokenizer.model").write_bytes(
        b"".join(base64.b64encode(token) + f" {i}\n".encode() for i, token in enumerate(vocab))
    )
    special = ["[gMASK]", "<sop>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]
    config = {
        "tokenizer_class": "ChatGLM4Tokenizer",
        "auto_map": {"AutoTokenizer": ["untrusted_code.ChatGLM4Tokenizer", None]},
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": special,
        "added_tokens_decoder": {
            str(len(vocab) + i): {
                "content": token,
                "special": True,
                "normalized": False,
                "lstrip": False,
                "rstrip": False,
                "single_word": False,
            }
            for i, token in enumerate(special)
        },
        "chat_template": (
            "[gMASK]<sop>{% for m in messages %}<|{{m.role}}|>\n{{m.content}}{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}"
        ),
    }
    (path / "config.json").write_text(json.dumps({"model_type": "chatglm"}))
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    # A repository Python file must never be needed, imported or copied.
    (path / "untrusted_code.py").write_text("raise RuntimeError('must not execute')\n")
    return path


def settings(path):
    return Settings(
        backend="vllm", model_id="fixture", model_revision="a" * 40, tokenizer=str(path)
    )


def test_conversion_preserves_ids_prefix_template_and_compilation(tmp_path):
    source = checkpoint(tmp_path / "source")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "profile"
    result = CliRunner().invoke(app, ["tokenizer", "convert-glm4", str(source), str(destination)])
    assert result.exit_code == 0, result.output
    profile = json.loads(result.output)
    assert profile["validation"]["passed"] and profile["validation"]["cases"] > 500
    assert set(p.name for p in destination.iterdir()) == {
        "tokenizer.json",
        "tokenizer_config.json",
        "jev-tokenizer-profile.json",
    }
    assert "auto_map" not in json.loads((destination / "tokenizer_config.json").read_text())
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    original, ranks, special, _ = read_glm4_inputs(source)
    reference = tiktoken.Encoding(
        name="test", pat_str=GLM4_PATTERN, mergeable_ranks=ranks, special_tokens=special
    )
    compiler = load_compiler(settings(destination))
    text = "[gMASK]<sop><|user|>Hello 你好 😀\n123456789<|assistant|>"
    for add in (False, True):
        expected = ([special["[gMASK]"], special["<sop>"]] if add else []) + reference.encode(
            text, allowed_special="all"
        )
        assert compiler.tokenizer.encode(text, add_special_tokens=add) == expected
    assert compiler.chat_template == original["chat_template"]
    bundle = Bundle(id="test", version=1, model=model_identity(settings(destination), compiler))
    question = Question(
        id="q",
        type="choice",
        instruction="Choose",
        options=(Option(id="a", description="One"), Option(id="b", description="Two")),
    )
    result = compiler.compile("退款", question, bundle, "request")
    assert len(result.sequences) == 1 and len(result.sequences[0].label_ids) == 2


@pytest.mark.parametrize("mode", ["rank_gap", "duplicate_bytes", "id_gap", "strip", "class"])
def test_invalid_conversion_does_not_publish_a_profile(tmp_path, mode):
    source = checkpoint(tmp_path / "source")
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    vocab = source / "tokenizer.model"
    if mode == "rank_gap":
        vocab.write_bytes(vocab.read_bytes().replace(b" 259\n", b" 300\n"))
    elif mode == "duplicate_bytes":
        with vocab.open("ab") as f:
            f.write(b"AA== 260\n")
    elif mode == "id_gap":
        config["added_tokens_decoder"]["300"] = config["added_tokens_decoder"].pop("260")
    elif mode == "strip":
        config["added_tokens_decoder"]["260"]["lstrip"] = True
    else:
        config["tokenizer_class"] = "SomeOtherTokenizer"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        convert_glm4_tokenizer(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_existing_destination_and_source_tree_are_protected(tmp_path):
    source = checkpoint(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        convert_glm4_tokenizer(source, output)
    assert (output / "keep").read_text() == "untouched"
    with pytest.raises(ValueError, match="outside"):
        convert_glm4_tokenizer(source, source / "derived")


def test_modified_profile_file_is_rejected_before_tokenizer_load(tmp_path, monkeypatch):
    destination = tmp_path / "profile"
    convert_glm4_tokenizer(checkpoint(tmp_path / "source"), destination)
    with (destination / "tokenizer.json").open("ab") as f:
        f.write(b"\n")
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **kw: pytest.fail("must reject before loading")
    )
    with pytest.raises(JevError) as error:
        load_compiler(settings(destination))
    assert error.value.code == "tokenizer_profile_mismatch"


async def test_host_pipeline_change_cannot_bypass_profile_via_supplied_compiler(tmp_path):
    destination = tmp_path / "profile"
    convert_glm4_tokenizer(checkpoint(tmp_path / "source"), destination)
    tokenizer = AutoTokenizer.from_pretrained(destination, trust_remote_code=False)
    tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(JevError) as error:
        compiler_for_tokenizer(settings(destination), tokenizer)
    assert error.value.code == "tokenizer_profile_mismatch"
    with pytest.raises(JevError) as error:
        await build_runtime(
            settings(destination), native_backend=object(), compiler=Compiler(tokenizer)
        )
    assert error.value.code == "tokenizer_profile_mismatch"
