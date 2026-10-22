import math
import sys

import pytest
from pydantic import ValidationError

from jev_runtime.backends.base import ScoreResult
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, Option, Question, TemplateSpec
from jev_runtime.scoring import assemble, softmax


def test_softmax_extreme_values():
    result = softmax([-10000, -10001])
    assert result == pytest.approx([0.7310585786, 0.2689414214])


@pytest.mark.parametrize("values", [[], [float("nan"), 0], [float("inf"), 1]])
def test_invalid_scores_are_not_uniform(values):
    with pytest.raises(JevError, match="Missing or non-finite"):
        softmax(values)


@pytest.mark.parametrize(
    "values,temperature,expected",
    [
        # ties at a common offset stay uniform at any magnitude and temperature
        ([-1e308, -1e308], 0.01, [0.5, 0.5]),
        ([1e308, 1e308, 1e308], 1e-300, [1 / 3, 1 / 3, 1 / 3]),
        ([-1e308, -1e308], 5e-324, [0.5, 0.5]),
        # extreme magnitudes whose scaled gap genuinely overflows underflow to zero
        ([1e308], 1, [1.0]),
        ([1e308, -1e308], 1, [1.0, 0.0]),
        # mixed extreme signs at a large temperature: the scaled gap is finite
        ([-1e308, 1e308], 1e308, [0.11920292202211755, 0.8807970779778823]),
        (
            [-1e308, 0.0, 1e308],
            1e308,
            [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
        ),
        ([-1e308, 1e308], sys.float_info.max, [0.24739822911054554, 0.7526017708894543]),
        (
            [-sys.float_info.max, sys.float_info.max],
            sys.float_info.max,
            [0.11920292202211755, 0.8807970779778823],
        ),
        # a tiny positive temperature sharpens to the argmax
        ([1.0, 2.0], 5e-324, [0.0, 1.0]),
        ([2.0, 1.0], 5e-324, [1.0, 0.0]),
        ([5e-324, 0.0], 5e-324, [0.7310585786300049, 0.2689414213699951]),
        # close, large scores keep their fine-grained difference
        ([1e15, 1e15 - 0.25], 1, [0.5621765008857981, 0.4378234991142019]),
        ([1e15, 1e15 - 0.5], 0.5, [0.7310585786300049, 0.2689414213699951]),
    ],
)
def test_softmax_numerically_stable(values, temperature, expected):
    result = softmax(values, temperature)
    assert result == pytest.approx(expected, rel=1e-14, abs=1e-15)
    assert all(math.isfinite(p) for p in result)
    assert math.fsum(result) == pytest.approx(1, rel=1e-15)


@pytest.mark.parametrize(
    "temperature", [0, 0.0, -0.0, -1, -1e308, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_temperatures_fail_closed(temperature):
    with pytest.raises(JevError, match="[Tt]emperature") as error:
        softmax([1.0, 2.0], temperature)
    assert error.value.code == "invalid_temperature"


def test_choice_mass_is_not_conditional_probability(compiler, bundle, question):
    compiled = compiler.compile("refund", question, bundle, "r")
    result = ScoreResult(compiled.sequences[0].request_id, (math.log(0.001), math.log(0.009)))
    answer = assemble(compiled, [result], bundle)
    assert answer.value == "support"
    assert answer.probabilities["support"] == pytest.approx(0.9)
    assert answer.label_mass == pytest.approx(0.01)
    assert answer.calibration_status == "uncalibrated"


def test_missing_label_does_not_become_zero(compiler, bundle, question):
    compiled = compiler.compile("refund", question, bundle, "r")
    with pytest.raises(JevError, match="mismatched"):
        assemble(compiled, [ScoreResult(compiled.sequences[0].request_id, (-1,))], bundle)


def test_tie_abstains(compiler, bundle, question):
    compiled = compiler.compile("refund", question, bundle, "r")
    answer = assemble(compiled, [ScoreResult(compiled.sequences[0].request_id, (-1, -1))], bundle)
    assert answer.abstained and answer.value is None and answer.reason == "tie"


def test_score_requires_explicit_spacing(compiler, bundle):
    q = Question(
        id="grade",
        type="score",
        instruction="Grade",
        options=(
            Option(id="bad", description="Bad"),
            Option(id="good", description="Good"),
        ),
    )
    compiled = compiler.compile("text", q, bundle, "r")
    result = ScoreResult(compiled.sequences[0].request_id, (math.log(0.2), math.log(0.8)))
    assert assemble(compiled, [result], bundle).value == "good"
    q = q.model_copy(
        update={
            "options": (
                Option(id="bad", description="Bad", value=0),
                Option(id="good", description="Good", value=10),
            )
        }
    )
    compiled = compiler.compile("text", q, bundle, "r")
    assert assemble(compiled, [result], bundle).value == pytest.approx(8)


def test_independent_support_survives_normalization(compiler, bundle, question):
    bundle = Bundle.model_validate(
        {**bundle.model_dump(), "template": TemplateSpec(mode="independent-candidate").model_dump()}
    )
    compiled = compiler.compile("refund", question, bundle, "r")
    results = [
        ScoreResult(seq.request_id, (math.log(p), math.log(1 - p)))
        for seq, p in zip(compiled.sequences, [0.001, 0.009], strict=True)
    ]
    answer = assemble(compiled, results, bundle)
    assert answer.probabilities["support"] == pytest.approx(0.9)
    assert answer.support == pytest.approx({"billing": 0.001, "support": 0.009})
    assert answer.probability_semantics == "normalized_support"


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        Question(
            id="q",
            type="choice",
            instruction="q",
            options=(
                Option(id="x", description="a"),
                Option(id="x", description="b"),
            ),
        )


def test_label_encoding_checked_in_actual_context(compiler, bundle, question):
    original = compiler.tokenizer.encode

    def merged(text, **kwargs):
        ids = original(text, **kwargs)
        if not text.endswith("assistant:"):
            return ids[:-2] + [9999]
        return ids

    compiler.tokenizer.encode = merged
    with pytest.raises(JevError) as error:
        compiler.compile("hello", question, bundle, "r")
    assert error.value.code == "label_encoding"
