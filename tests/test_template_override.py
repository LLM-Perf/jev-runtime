import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from jev_runtime.compiler import Compiler
from jev_runtime.config import Settings, build_runtime, compiler_for_tokenizer, model_identity
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, Option, Question
from jev_runtime.templates import TemplateFile, read_template
from tests.conftest import CharacterTokenizer

TEMPLATE = (
    "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:\n{% endif %}"
)


def tokenizer():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "A": 1, "B": 2, "C": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


def template_file(tmp_path, template=TEMPLATE, format="jinja"):
    raw = (json.dumps({"chat_template": template}) if format == "json" else template).encode()
    path = tmp_path / ("template." + format)
    path.write_bytes(raw)
    return TemplateFile(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), format=format)


@pytest.mark.parametrize("format", ["jinja", "json"])
def test_explicit_file_snapshot_rejects_later_tampering(tmp_path, format):
    spec = template_file(tmp_path, format=format)
    assert read_template(spec) == TEMPLATE
    with open(spec.path, "ab") as file:
        file.write(b" ")
    with pytest.raises(JevError) as error:
        read_template(spec)
    assert error.value.code == "template_file_mismatch"


@pytest.mark.parametrize("raw", [b"[]", b'{"chat_template":null}', b"{}", b"{", b"\xff"])
def test_hash_match_does_not_make_invalid_json_a_template(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    spec = TemplateFile(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), format="json")
    with pytest.raises(JevError) as error:
        read_template(spec)
    assert error.value.code == "template_file_invalid"


def test_missing_host_template_can_be_explicit_without_mutating_shared_tokenizer(tmp_path):
    host = tokenizer()
    before = host.backend_tokenizer.to_str()
    with pytest.raises(JevError, match="explicit chat template"):
        Compiler(host)
    settings = Settings(
        backend="vllm",
        model_id="fixture",
        model_revision="a" * 40,
        chat_template=template_file(tmp_path, format="json"),
    )
    compiler = compiler_for_tokenizer(settings, host)
    bundle = Bundle(id="fixture", version=1, model=model_identity(settings, compiler))
    question = Question(
        id="choice",
        type="choice",
        instruction="Choose",
        options=(Option(id="one", description="First"), Option(id="two", description="Second")),
    )
    result = compiler.compile("Example", question, bundle, "req")
    assert len(result.sequences) == 1 and len(result.sequences[0].label_ids) == 2
    assert host.chat_template is None and host.backend_tokenizer.to_str() == before
    assert compiler.profile()["chat_template_source"] == "explicit"
    changed = Compiler(host, chat_template=TEMPLATE + "\n")
    with pytest.raises(JevError) as error:
        changed.verify_bundle(bundle)
    assert error.value.code == "template_mismatch"


def test_custom_renderer_cannot_silently_ignore_explicit_override():
    # This fixture ignores chat_template kwargs, as some non-HF wrappers can.
    with pytest.raises(JevError) as error:
        Compiler(CharacterTokenizer(), chat_template=TEMPLATE)
    assert error.value.code == "template_override_unsupported"
    assert Compiler(CharacterTokenizer()).profile()["chat_template_source"] == "host"


async def test_runtime_rejects_compiler_that_ignored_configured_template(tmp_path):
    settings = Settings(
        backend="vllm",
        model_id="fixture",
        model_revision="a" * 40,
        registry_path=str(tmp_path / "registry.db"),
        chat_template=template_file(tmp_path),
    )
    compiler = Compiler(tokenizer(), chat_template=TEMPLATE + "different")
    with pytest.raises(JevError) as error:
        await build_runtime(settings, native_backend=object(), compiler=compiler)
    assert error.value.code == "template_mismatch"
    assert not (tmp_path / "registry.db").exists()
