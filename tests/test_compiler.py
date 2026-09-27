import pytest

from jev_runtime.compiler import Compiler
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, Option, Question, content_digest


@pytest.mark.parametrize("count", [32, 64])
def test_large_joint_readout_handles_digit_splitting_tokenizers(compiler, bundle, count):
    question = Question(
        id="large",
        type="choice",
        instruction="Select the matching category.",
        options=tuple(Option(id=f"category{i}", description=f"Category {i}") for i in range(count)),
    )
    compiled = compiler.compile("A refund", question, bundle, "request")
    assert len(compiled.sequences) == 1
    sequence = compiled.sequences[0]
    assert len(sequence.label_ids) == len(set(sequence.label_ids)) == count
    assert compiled.keys == tuple(option.id for option in question.options)


def test_exact_cache_preserves_ids_and_new_dispatch_namespace(
    compiler, bundle, question, monkeypatch
):
    first = compiler.compile("refund", question, bundle, "first")

    def unexpected(*args, **kwargs):
        raise AssertionError("Exact encoding should already be cached")

    monkeypatch.setattr(compiler.tokenizer, "encode", unexpected)
    second = compiler.compile("refund", question, bundle, "second")
    assert first.sequences[0].input_ids == second.sequences[0].input_ids
    assert first.sequences[0].label_ids == second.sequences[0].label_ids
    assert first.sequences[0].request_id != second.sequences[0].request_id
    smaller = bundle.model_copy(
        update={"policy": bundle.policy.model_copy(update={"max_input_tokens": 1})}
    )
    with pytest.raises(JevError) as error:
        compiler.compile("refund", question, smaller, "third")
    assert error.value.code == "context_budget"


def test_encoding_cache_binds_labels_prompt_and_enforces_both_bounds(compiler):
    cached = Compiler(compiler.tokenizer, cache_tokens=10, cache_entries=1)
    first = cached._encode("abc", ["A", "B"], 10)
    changed = cached._encode("abc", ["C", "D"], 10)
    assert first[0] == changed[0] and first[1] != changed[1]
    assert len(cached._encoding_cache) == 1
    different = cached._encode("abcd", ["C", "D"], 10)
    assert first[0] != different[0]
    assert cached._cached_tokens == 6
    # An oversized entry must not displace or exceed the bounded cache.
    cached._encode("0123456789", ["A", "B"], 10)
    assert cached._cached_tokens == 6
    uncached = Compiler(compiler.tokenizer, cache_tokens=0)
    assert uncached._encode("abc", ["A", "B"], 10) == first
    assert not uncached._encoding_cache


def test_over_budget_stops_before_testing_label_continuations(compiler, monkeypatch):
    calls = []

    def encode(text, **kwargs):
        calls.append(text)
        return list(range(len(text)))

    monkeypatch.setattr(compiler.tokenizer, "encode", encode)
    with pytest.raises(JevError) as error:
        compiler._encode("too long", ["A", "B"], 2)
    assert error.value.code == "context_budget" and calls == ["too long"]


def test_backend_fingerprint_rejects_equal_vocabulary_with_different_rules(compiler, bundle):
    from copy import deepcopy
    from types import SimpleNamespace

    tokenizer = deepcopy(compiler.tokenizer)
    tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer":"NFC"}')
    actual = Compiler(tokenizer)
    bound = bundle.model_copy(
        update={
            "model": bundle.model.model_copy(
                update={"tokenizer_implementation_digest": actual.tokenizer_implementation_digest}
            )
        }
    )
    actual.verify_bundle(bound)
    assert compiler.tokenizer_digest == actual.tokenizer_digest
    with pytest.raises(JevError) as missing:
        actual.verify_bundle(bundle)
    assert missing.value.code == "tokenizer_identity_incomplete"
    tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer":"NFKC"}')
    different = Compiler(tokenizer)
    with pytest.raises(JevError) as mismatch:
        different.verify_bundle(bound)
    assert mismatch.value.code == "tokenizer_implementation_mismatch"
    assert bundle.scoring_contract_digest != bound.scoring_contract_digest


def test_old_manifest_digests_survive_optional_identity_extension(bundle):
    legacy = bundle.model_dump(mode="json")
    assert "tokenizer_implementation_digest" not in legacy["model"]
    loaded = Bundle.model_validate(legacy)
    assert loaded.digest == content_digest(legacy)
    assert loaded.model_dump(mode="json") == legacy


def test_fast_batch_encodes_full_context_with_bounded_chunks():
    from tests.conftest import CharacterTokenizer

    class FastTokenizer(CharacterTokenizer):
        is_fast = True

        def __init__(self):
            self.batches = []

        def __call__(self, texts, **kwargs):
            assert kwargs == {
                "add_special_tokens": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }
            self.batches.append(texts)
            return {"input_ids": [self.encode(text) for text in texts]}

    tokenizer = FastTokenizer()
    fast = Compiler(tokenizer, cache_entries=0)
    slow = Compiler(CharacterTokenizer(), cache_entries=0)
    labels = [chr(i) for i in range(33, 103)]
    assert fast._encode("context ", labels, 100) == slow._encode("context ", labels, 100)
    assert [len(batch) for batch in tokenizer.batches] == [32, 32, 6]
    assert all(text.startswith("context ") for batch in tokenizer.batches for text in batch)


@pytest.mark.parametrize("broken", ["retokenize", "drop", "unsupported"])
def test_fast_batch_never_skips_continuation_checks_or_breaks_custom_fallback(broken):
    from tests.conftest import CharacterTokenizer

    class FastTokenizer(CharacterTokenizer):
        is_fast = True

        def __call__(self, texts, **kwargs):
            if broken == "unsupported":
                raise NotImplementedError("custom tokenizer exposes only encode")
            results = [self.encode(text) for text in texts]
            if broken == "retokenize":
                results[0][0] += 1
            else:
                results.pop()
            return {"input_ids": results}

    compiler = Compiler(FastTokenizer(), cache_entries=0)
    if broken == "unsupported":
        assert compiler._encode("abc", ["A", "B"], 100) == ((97, 98, 99), (65, 66))
    else:
        with pytest.raises(JevError) as error:
            compiler._encode("abc", ["A", "B"], 100)
        assert error.value.code == (
            "label_encoding" if broken == "retokenize" else "tokenizer_contract"
        )
