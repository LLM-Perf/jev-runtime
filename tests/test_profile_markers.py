import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from jev_runtime.config import Settings, load_compiler
from jev_runtime.errors import JevError
from jev_runtime.tokenizer_profiles import verify_tokenizer_profile


def settings(path):
    return Settings(
        backend="vllm", model_id="fixture", model_revision="a" * 40, tokenizer=str(path)
    )


@pytest.mark.parametrize("kind", ["dangling_symlink", "symlink", "loop", "directory", "fifo"])
def test_nonregular_manifest_is_rejected_before_reading_or_loading(tmp_path, monkeypatch, kind):
    marker = tmp_path / "jev-tokenizer-profile.json"
    if kind == "directory":
        marker.mkdir()
    elif kind == "fifo":
        os.mkfifo(marker)
    elif kind == "loop":
        marker.symlink_to(marker.name)
    else:
        target = tmp_path / "manifest-target.json"
        if kind == "symlink":
            target.write_text("{}")
        marker.symlink_to(target)

    monkeypatch.setattr(
        Path, "read_text", lambda *a, **kw: pytest.fail("must reject before reading manifest")
    )
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **kw: pytest.fail("must reject before loading")
    )
    with pytest.raises(JevError) as error:
        load_compiler(settings(tmp_path))
    assert error.value.code == "tokenizer_profile_mismatch"


@pytest.mark.parametrize("content", [b"{", b"\xff", b"[]", b"{}"])
def test_malformed_manifest_is_rejected_before_loading(tmp_path, monkeypatch, content):
    (tmp_path / "jev-tokenizer-profile.json").write_bytes(content)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **kw: pytest.fail("must reject before loading")
    )
    with pytest.raises(JevError) as error:
        load_compiler(settings(tmp_path))
    assert error.value.code == "tokenizer_profile_mismatch"


def test_ordinary_checkpoint_without_manifest_still_loads(tmp_path, monkeypatch, compiler):
    assert verify_tokenizer_profile(tmp_path) is None
    loader = Mock(return_value=compiler.tokenizer)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    loaded = load_compiler(settings(tmp_path))
    loader.assert_called_once_with(str(tmp_path), trust_remote_code=False)
    assert loaded.tokenizer_digest == compiler.tokenizer_digest
