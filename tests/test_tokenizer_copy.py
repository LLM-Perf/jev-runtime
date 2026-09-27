import copy
import json

import pytest
from jev_vllm import tokenizer as integration


class Backend:
    def __init__(self):
        self.value = {"vocab": ["a", "b"]}

    def to_str(self):
        return json.dumps(self.value)


class Prototype:
    def __init__(self):
        self.backend_tokenizer = Backend()
        self.chat_template = "test"
        self.special_tokens_map = {"eos_token": "b"}
        self.split_special_tokens = False

    def get_vocab(self):
        return {value: i for i, value in enumerate(self.backend_tokenizer.value["vocab"])}


def rebuild_pool():
    pass


rebuild_pool.__module__ = "vllm.tokenizers.hf"
rebuild_pool.__qualname__ = "maybe_make_thread_pool"


class Host(Prototype):
    def __init__(self):
        self.prototype = Prototype()
        self.__dict__.update(self.prototype.__dict__)

    def encode(self):
        pass

    def __reduce__(self):
        return rebuild_pool, (self.prototype, 1)


Host.encode.__module__ = "vllm.tokenizers.hf"
Host.encode.__qualname__ = "maybe_make_thread_pool.<locals>.TokenizerPool.encode"


def test_private_host_copy_preserves_identity_without_sharing_mutable_backend(monkeypatch):
    host = Host()
    monkeypatch.setattr(integration, "supports_backend_ids_only", lambda _: True)
    private, copied = integration.compiler_tokenizer(host)
    assert copied and private is not host and private is not host.prototype
    assert private.backend_tokenizer.to_str() == host.backend_tokenizer.to_str()
    private.backend_tokenizer.value["vocab"].append("new")
    private.special_tokens_map["eos_token"] = "new"
    assert host.get_vocab() == {"a": 0, "b": 1}
    assert host.special_tokens_map["eos_token"] == "b"


@pytest.mark.parametrize("mode", ["custom_encode", "shared_backend", "changed_identity"])
def test_optional_copy_falls_back_without_changing_host(monkeypatch, mode):
    host = Host()
    monkeypatch.setattr(integration, "supports_backend_ids_only", lambda _: mode != "custom_encode")
    if mode != "custom_encode":
        private = copy.deepcopy(host.prototype)
        if mode == "shared_backend":
            private.backend_tokenizer = host.backend_tokenizer
        else:
            private.backend_tokenizer.value["vocab"].append("different")
        monkeypatch.setattr(integration.copy, "deepcopy", lambda _: private)
    selected, copied = integration.compiler_tokenizer(host)
    assert selected is host and not copied
    assert host.get_vocab() == {"a": 0, "b": 1}
