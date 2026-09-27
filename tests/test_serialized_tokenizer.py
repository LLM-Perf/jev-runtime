import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer
from typer.testing import CliRunner

from jev_runtime.cli import app
from jev_runtime.config import load_compiler
from jev_runtime.errors import JevError
from jev_runtime.tokenizer_profiles import preserve_fast_tokenizer, verify_tokenizer_profile
from tests.test_tokenizer_profiles import settings

TEMPLATE = "{{ bos_token }}{% for m in messages %}{{m.content}}{% endfor %}"


def checkpoint(path, template=True):
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
    config = {
        "tokenizer_class": "GPT2Tokenizer",
        "auto_map": {"AutoTokenizer": ["poison.Tokenizer", None]},
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    if template:
        config["chat_template"] = TEMPLATE
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    (path / "poison.py").write_text("raise RuntimeError('checkpoint code must not execute')")
    return path


def test_preserve_fast_cli_keeps_entire_digits_pipeline_and_source(tmp_path):
    source = checkpoint(tmp_path / "source")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "profile"
    result = CliRunner().invoke(app, ["tokenizer", "preserve-fast", str(source), str(destination)])
    assert result.exit_code == 0, result.output
    manifest = json.loads(result.output)
    assert manifest["validation"]["full_backend_equal"]
    assert manifest["validation"]["cases"] > 500
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert (destination / "tokenizer.json").read_bytes() == before["tokenizer.json"]
    assert not (destination / "poison.py").exists()
    assert manifest["removed_config_keys"] == ["auto_map", "tokenizer_class"]
    compiled = load_compiler(settings(destination))
    reference = Tokenizer.from_file(str(source / "tokenizer.json"))
    assert json.loads(compiled.tokenizer.backend_tokenizer.to_str()) == json.loads(
        reference.to_str()
    )
    for add in [False, True]:
        assert (
            compiled.tokenizer.encode("9 ²,abc1①", add_special_tokens=add)
            == reference.encode("9 ²,abc1①", add_special_tokens=add).ids
        )
    assert compiled.tokenizer.encode("Hello", add_special_tokens=True)[0] == 0
    assert verify_tokenizer_profile(destination, compiled)["kind"] == "serialized-fast-v1"


def test_template_files_and_named_templates_are_bound(tmp_path):
    source = checkpoint(tmp_path / "source")
    (source / "chat_template.jinja").write_text("file-default {{ messages[0].content }}")
    (source / "additional_chat_templates").mkdir()
    (source / "additional_chat_templates/tool_use.jinja").write_text(
        "tools {{ messages[0].content }}"
    )
    destination = tmp_path / "profile"
    manifest = preserve_fast_tokenizer(source, destination)
    compiled = load_compiler(settings(destination))
    assert compiled.chat_template == {
        "default": "file-default {{ messages[0].content }}",
        "tool_use": "tools {{ messages[0].content }}",
    }
    assert manifest["validation"]["template_renders"] == 16
    (destination / "additional_chat_templates/tool_use.jinja").write_text("changed")
    with pytest.raises(JevError, match="bytes differ"):
        verify_tokenizer_profile(destination)


def test_explicit_template_requires_hash_and_replaces_named_set(tmp_path):
    source = checkpoint(tmp_path / "source", template=False)
    (source / "additional_chat_templates").mkdir()
    (source / "additional_chat_templates/tool_use.jinja").write_text("{{ raise_exception('old') }}")
    override = tmp_path / "override.jinja"
    override.write_text(TEMPLATE)
    destination = tmp_path / "profile"
    with pytest.raises(ValueError, match="together"):
        preserve_fast_tokenizer(source, destination, chat_template=override)
    with pytest.raises(ValueError, match="SHA256 differs"):
        preserve_fast_tokenizer(
            source, destination, chat_template=override, chat_template_sha256="0" * 64
        )
    manifest = preserve_fast_tokenizer(
        source,
        destination,
        chat_template=override,
        chat_template_sha256=hashlib.sha256(override.read_bytes()).hexdigest(),
    )
    assert "additional_chat_templates/tool_use.jinja" in manifest["source_files"]
    assert "additional_chat_templates/tool_use.jinja" not in manifest["files"]
    assert load_compiler(settings(destination)).chat_template == TEMPLATE


@pytest.mark.parametrize("mode", ["missing", "split", "padding", "dropout", "extra_special"])
def test_unsupported_or_changed_backend_never_publishes(tmp_path, mode):
    source = checkpoint(tmp_path / "source")
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    backend = Tokenizer.from_file(str(source / "tokenizer.json"))
    if mode == "missing":
        (source / "tokenizer.json").unlink()
    elif mode == "split":
        config["split_special_tokens"] = True
    elif mode == "padding":
        backend.enable_padding(pad_id=1, pad_token="</s>")
        backend.save(str(source / "tokenizer.json"))
    elif mode == "dropout":
        backend.model.dropout = 0.5
        backend.save(str(source / "tokenizer.json"))
    else:
        config["pad_token"] = "<new-token-not-in-checkpoint>"
    config_path.write_text(json.dumps(config))
    destination = tmp_path / "profile"
    with pytest.raises(ValueError):
        preserve_fast_tokenizer(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["extra_file", "symlink", "manifest_traversal"])
def test_loader_rejects_unbound_profile_files_before_transformers(tmp_path, monkeypatch, mode):
    destination = tmp_path / "profile"
    preserve_fast_tokenizer(checkpoint(tmp_path / "source"), destination)
    if mode == "extra_file":
        (destination / "chat_template.jinja").write_text("silently override declared template")
    elif mode == "symlink":
        raw = (destination / "tokenizer.json").read_bytes()
        (tmp_path / "outside.json").write_bytes(raw)
        (destination / "tokenizer.json").unlink()
        (destination / "tokenizer.json").symlink_to(tmp_path / "outside.json")
    else:
        path = destination / "jev-tokenizer-profile.json"
        data = json.loads(path.read_text())
        data["files"]["../outside.json"] = data["files"]["tokenizer.json"]
        path.write_text(json.dumps(data))
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **kw: pytest.fail("loaded unbound profile")
    )
    with pytest.raises(JevError) as error:
        load_compiler(settings(destination))
    assert error.value.code == "tokenizer_profile_mismatch"


def test_source_change_during_load_is_rejected_and_destination_protected(tmp_path, monkeypatch):
    source = checkpoint(tmp_path / "source")
    destination = tmp_path / "profile"
    original = AutoTokenizer.from_pretrained

    def changed(*args, **kwargs):
        loaded = original(*args, **kwargs)
        with (source / "tokenizer.json").open("ab") as file:
            file.write(b"\n")
        return loaded

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", changed)
    with pytest.raises(ValueError, match="Source tokenizer changed"):
        preserve_fast_tokenizer(source, destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="outside"):
        preserve_fast_tokenizer(source, source / "derived")
    destination.mkdir()
    (destination / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        preserve_fast_tokenizer(source, destination)
    assert (destination / "keep").read_text() == "untouched"


@pytest.mark.parametrize("kind", ["serialized", "glm4"])
def test_interrupted_publication_cannot_load_as_an_ordinary_checkpoint(tmp_path, monkeypatch, kind):
    from jev_runtime.tokenizer_profiles import convert_glm4_tokenizer
    from tests.test_tokenizer_profiles import checkpoint as glm_checkpoint

    source = (checkpoint if kind == "serialized" else glm_checkpoint)(tmp_path / "source")
    convert = preserve_fast_tokenizer if kind == "serialized" else convert_glm4_tokenizer
    destination = tmp_path / "profile"
    original = Path.open

    def fail(path, *args, **kwargs):
        if path == destination / "tokenizer_config.json" and args == ("xb",):
            raise OSError("injected publication write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError, match="injected"):
        convert(source, destination)
    assert (destination / "jev-tokenizer-profile.pending").is_file()
    assert not (destination / "jev-tokenizer-profile.json").exists()
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **kw: pytest.fail("loaded incomplete profile")
    )
    with pytest.raises(JevError, match="publication is incomplete"):
        load_compiler(settings(destination))


def test_absent_processor_allows_only_the_exact_single_text_identity(tmp_path, monkeypatch):
    source = checkpoint(tmp_path / "source")
    raw = Tokenizer.from_file(str(source / "tokenizer.json"))
    raw.post_processor = None
    raw.save(str(source / "tokenizer.json"))
    result = preserve_fast_tokenizer(source, tmp_path / "profile")
    assert result["validation"]["single_text_backend_equivalent"]
    loaded = load_compiler(settings(tmp_path / "profile")).tokenizer
    for add in (False, True):
        assert (
            loaded.encode("Hello", add_special_tokens=add)
            == raw.encode("Hello", add_special_tokens=add).ids
        )
    original = AutoTokenizer.from_pretrained

    def adds_prefix(*args, **kwargs):
        tokenizer = original(*args, **kwargs)
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 0)]
        )
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", adds_prefix)
    with pytest.raises(ValueError, match="changed the serialized"):
        preserve_fast_tokenizer(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
