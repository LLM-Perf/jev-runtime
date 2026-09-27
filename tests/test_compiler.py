import pytest

from jev_runtime.compiler import Compiler
from jev_runtime.errors import JevError
from jev_runtime.schema import Option, Question


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
