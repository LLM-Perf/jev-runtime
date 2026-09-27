from __future__ import annotations

import json
import string
import threading
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from jev_runtime.backends.base import ScoreInput
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, Question, content_digest


@dataclass(frozen=True)
class CompiledQuestion:
    question: Question
    mode: str
    keys: tuple[str, ...]
    sequences: tuple[ScoreInput, ...]


class Compiler:
    def __init__(self, tokenizer: Any, cache_tokens: int = 262144, cache_entries: int = 256):
        if cache_tokens < 0 or cache_entries < 0:
            raise ValueError("Compiler cache bounds cannot be negative")
        self.tokenizer = tokenizer
        self._cache_tokens = cache_tokens
        self._cache_entries = cache_entries
        self._cached_tokens = 0
        self._encoding_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        if not getattr(tokenizer, "chat_template", None):
            raise JevError("template_missing", "The tokenizer needs an explicit chat template")
        self.template_digest = content_digest(tokenizer.chat_template)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        # Record the complete normalization/pretokenization/BPE implementation
        # for experiment identity. This is separate from the existing bundle
        # vocabulary fingerprint so old manifests are not silently rewritten.
        self.tokenizer_implementation_digest = (
            content_digest(backend.to_str()) if backend is not None else None
        )
        self.tokenizer_digest = content_digest(
            {
                "vocab": sorted(tokenizer.get_vocab().items()),
                "special_tokens": tokenizer.special_tokens_map,
            }
        )

    def verify_bundle(self, bundle: Bundle) -> None:
        if bundle.model.tokenizer_digest != self.tokenizer_digest:
            raise JevError("tokenizer_mismatch", "Bundle tokenizer fingerprint does not match")
        if bundle.model.tokenizer_implementation_digest != self.tokenizer_implementation_digest:
            if bundle.model.tokenizer_implementation_digest is None:
                raise JevError(
                    "tokenizer_identity_incomplete",
                    "Rebuild a new bundle version with the serving tokenizer implementation digest",
                    409,
                )
            raise JevError(
                "tokenizer_implementation_mismatch",
                "Bundle tokenizer normalization/pretokenization/BPE implementation differs",
                409,
            )
        if bundle.model.template_digest != self.template_digest:
            raise JevError("template_mismatch", "Bundle chat template fingerprint does not match")

    def _render(self, body: str, bundle: Bundle) -> str:
        spec = bundle.template
        messages = []
        if spec.use_system_role:
            messages.append({"role": "system", "content": spec.system_prompt})
        else:
            body = spec.system_prompt + "\n\n" + body
        messages.append({"role": "user", "content": body})
        kwargs = {}
        if "enable_thinking" in str(self.tokenizer.chat_template):
            kwargs["enable_thinking"] = spec.thinking
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception as exc:
            raise JevError(
                "template_error", f"Template rendering failed: {type(exc).__name__}"
            ) from exc

    def _encode(
        self, prompt: str, labels: list[str], max_input_tokens: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # An exact rendered-prompt digest and exact labels identify this entry.
        # No suffix heuristic is safe for arbitrary tokenizer normalizers/BPE.
        # Raw input strings are not retained. Both token and entry counts bound
        # the per-worker cache; request IDs and bundle policy are never cached.
        key = (sha256(prompt.encode()).digest(), tuple(labels))
        with self._cache_lock:
            cached = self._encoding_cache.get(key)
            if cached is not None:
                self._encoding_cache.move_to_end(key)
        if cached is not None:
            if len(cached[0]) > max_input_tokens:
                raise JevError("context_budget", "Compiled prompt exceeds context budget", 413)
            return cached
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            raise JevError("context_budget", "Compiled prompt exceeds context budget", 413)
        label_ids = []
        for label in labels:
            extended = self.tokenizer.encode(prompt + label, add_special_tokens=False)
            if len(extended) != len(ids) + 1 or extended[:-1] != ids:
                raise JevError("label_encoding", "Answer labels are not single-token continuations")
            label_ids.append(extended[-1])
        if len(set(label_ids)) != len(label_ids):
            raise JevError("label_encoding", "Answer labels do not map to distinct tokens")
        encoded = tuple(ids), tuple(label_ids)
        size = len(ids) + len(label_ids)
        if self._cache_entries and size <= self._cache_tokens:
            with self._cache_lock:
                previous = self._encoding_cache.pop(key, None)
                if previous is not None:
                    self._cached_tokens -= sum(map(len, previous))
                self._encoding_cache[key] = encoded
                self._cached_tokens += size
                while (
                    self._cached_tokens > self._cache_tokens
                    or len(self._encoding_cache) > self._cache_entries
                ):
                    _, evicted = self._encoding_cache.popitem(last=False)
                    self._cached_tokens -= sum(map(len, evicted))
        return encoded

    @staticmethod
    def _label_sets(count: int, binary: bool = False) -> list[list[str]]:
        if binary:
            return [["Yes", "No"], ["yes", "no"], ["A", "B"], [" A", " B"]]
        sets = []
        if count <= 26:
            sets.extend(
                [list(string.ascii_uppercase[:count]), list(string.ascii_lowercase[:count])]
            )
        sets.extend([[str(i) for i in range(count)], [f" {i}" for i in range(count)]])
        # Digit-splitting tokenizers cannot encode labels 10..31 as one token.
        # Keep existing label choices first, then try a 64-symbol alphabet.
        # _encode still proves continuation stability and distinct token IDs.
        alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + "!?"
        if count <= len(alphabet):
            sets.extend([list(alphabet[:count]), [f" {label}" for label in alphabet[:count]]])
        return sets

    def compile(
        self, text: str, question: Question, bundle: Bundle, request_id: str
    ) -> CompiledQuestion:
        self.verify_bundle(bundle)
        if len(question.options) > bundle.policy.max_options:
            raise JevError("option_budget", "Question exceeds the candidate budget", 413)
        mode = bundle.template.mode
        keys = (
            ("true", "false")
            if question.type == "boolean"
            else tuple(o.id for o in question.options)
        )
        base = (
            "Task:\n"
            + question.instruction
            + "\n\nData (JSON string):\n"
            + json.dumps(text, ensure_ascii=False)
            + "\n\n"
        )
        descriptions = (
            ["The statement is true", "The statement is false"]
            if question.type == "boolean"
            else [o.description for o in question.options]
        )
        independent = mode == "independent-candidate" and question.type != "boolean"
        candidates = list(zip(keys, descriptions, strict=True)) if independent else [(None, None)]
        sequences = []
        for index, (candidate_id, description) in enumerate(candidates):
            binary = independent or question.type == "boolean"
            count = 2 if binary else len(keys)
            error = None
            for labels in self._label_sets(count, binary):
                if independent:
                    body = base + "Candidate:\n" + str(description)
                    body += (
                        f"\nDoes this candidate satisfy the task? "
                        f"{labels[0]}: yes; {labels[1]}: no."
                    )
                else:
                    body = (
                        base
                        + "Answers:\n"
                        + "\n".join(
                            f"{label}: {desc}"
                            for label, desc in zip(labels, descriptions, strict=True)
                        )
                    )
                body += "\nReturn exactly one answer label."
                prompt = self._render(body, bundle)
                try:
                    input_ids, label_ids = self._encode(
                        prompt, labels, bundle.policy.max_input_tokens
                    )
                except JevError as exc:
                    if exc.code != "label_encoding":
                        raise
                    error = exc
                    continue
                sequences.append(
                    ScoreInput(
                        request_id=f"{request_id}.{question.id}.{index}",
                        question_id=question.id,
                        input_ids=input_ids,
                        label_ids=label_ids,
                        candidate_id=candidate_id,
                        adapter_id=(
                            f"{bundle.model.adapter_id}@{bundle.model.adapter_revision}"
                            if bundle.model.adapter_id
                            else None
                        ),
                    )
                )
                break
            else:
                raise error or JevError("label_encoding", "No compatible answer label set")
        return CompiledQuestion(question, mode, keys, tuple(sequences))
