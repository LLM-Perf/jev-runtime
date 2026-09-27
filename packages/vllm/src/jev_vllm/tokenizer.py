"""A private compiler copy of vLLM's verified standard HF pool prototype."""

from __future__ import annotations

import copy

from jev_runtime.compiler import supports_backend_ids_only


def compiler_tokenizer(host):
    encode = getattr(host, "encode", None)
    if (getattr(encode, "__module__", None), getattr(encode, "__qualname__", None)) != (
        "vllm.tokenizers.hf",
        "maybe_make_thread_pool.<locals>.TokenizerPool.encode",
    ):
        return host, False
    # vLLM 0.30's public pickle reduction retains the exact pre-pool object.
    # Adjacent pool calls can borrow different copies, so the compiler must
    # never bypass the live pool by calling its shared backend directly.
    try:
        rebuild, args = host.__reduce__()
        if (rebuild.__module__, rebuild.__qualname__) != (
            "vllm.tokenizers.hf",
            "maybe_make_thread_pool",
        ) or len(args) != 2:
            return host, False
        prototype = args[0]
        if not supports_backend_ids_only(prototype):
            return host, False
        private = copy.deepcopy(prototype)
        if not supports_backend_ids_only(private):
            return host, False
        if (
            private is prototype
            or private.backend_tokenizer is host.backend_tokenizer
            or private.backend_tokenizer is prototype.backend_tokenizer
            or host.backend_tokenizer.to_str() != private.backend_tokenizer.to_str()
            or host.get_vocab() != private.get_vocab()
            or host.chat_template != private.chat_template
            or host.special_tokens_map != private.special_tokens_map
            or host.split_special_tokens != private.split_special_tokens
        ):
            return host, False
        return private, True
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        # This is an optional CPU optimization; unsupported wrappers continue
        # through their original thread-safe public encoding API.
        return host, False
