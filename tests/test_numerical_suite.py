import copy
import math

import pytest

from jev_runtime.schema import content_digest
from tests.integration.numerical_suite import STATES, compare, corpus, reference_readout, summarize


def fixture():
    cases = [
        {**case, "sequence": {"input_ids": [1, 2, i + 3], "label_ids": [5, 6]}}
        for i, case in enumerate(corpus())
    ]
    model = {"id": "fixture", "revision": "0" * 40}
    engine = {
        "complete": True,
        "model": model,
        "cases": cases,
        "corpus_digest": content_digest(corpus()),
        "observations": {
            state: [{"id": case["id"], "logprobs": [-1, -2]} for case in cases] for state in STATES
        },
    }
    reference = {
        "complete": True,
        "model": model,
        "cases_digest": content_digest(cases),
        "rows": [
            {
                "id": case["id"],
                "sequence_digest": content_digest(case["sequence"]),
                "logprobs": [-1, -2],
            }
            for case in cases
        ],
    }
    return engine, reference


def test_equal_full_denominator_does_not_imply_release_certification():
    result = summarize(*fixture())
    assert result["comparisons"] == result["passed_comparisons"] == 96
    assert len(result["state_differences"]) == 64
    assert result["case_count"] == 32
    assert result["near_tie_cases"] == 0
    assert result["full_release_gate_passed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e, r: e.update(complete=False),
        lambda e, r: r.update(complete=False),
        lambda e, r: r.update(model={"id": "other"}),
        lambda e, r: e["cases"][0]["sequence"]["label_ids"].reverse(),
        lambda e, r: e["observations"].pop(STATES[0]),
        lambda e, r: e["observations"][STATES[1]].pop(),
        lambda e, r: r["rows"].reverse(),
        lambda e, r: e["observations"][STATES[0]][0].update(logprobs=[-1]),
        lambda e, r: r["rows"][0].update(sequence_digest="wrong"),
        lambda e, r: e["observations"][STATES[2]][0].update(logprobs=[math.nan, -1]),
    ],
)
def test_rejects_incomplete_or_mismatched_evidence(mutation):
    engine, reference = fixture()
    mutation(engine, reference)
    with pytest.raises(ValueError):
        summarize(engine, reference)


def test_near_tie_mismatch_remains_failed_even_with_small_logprob_error():
    result = compare([-1.02, -1], [-1, -1.02])
    assert result["near_tie"] and not result["top_label_equal"]
    assert result["max_absolute_logprob_error"] < 0.15
    assert not result["passed"]


def test_common_offset_fails_logprob_budget_even_with_equal_conditional_probabilities():
    result = compare([-2, -3], [-1, -2])
    assert result["max_absolute_logprob_error"] == 1
    assert result["max_absolute_conditional_probability_error"] == 0
    assert result["actual_label_mass"] < result["reference_label_mass"]
    assert not result["passed"]


def test_one_state_failure_is_counted_and_input_is_not_mutated():
    engine, reference = fixture()
    engine["observations"][STATES[2]][7]["logprobs"][0] -= 0.2
    before = copy.deepcopy(engine)
    result = summarize(engine, reference)
    assert result["passed_comparisons"] == 95
    assert not result["all_comparisons_passed"]
    assert engine == before


@pytest.mark.parametrize("readout", ["bfloat16", "float32"])
def test_reference_precision_is_bound_to_saved_model(readout):
    source = {"model_id": "fixture", "revision": "a" * 40}
    engine = {
        "complete": True,
        "model": {
            "id": "fixture",
            "revision": "a" * 40,
            "dtype": "bfloat16",
            "readout_dtype": readout,
            "quantization": None,
        },
    }
    before = copy.deepcopy(engine)
    assert reference_readout(source, engine) == readout
    assert engine == before
    # A missing/unknown precision or different checkpoint is never guessed.
    for update in (
        {"readout_dtype": None},
        {"readout_dtype": "float16"},
        {"dtype": "float32"},
        {"quantization": "awq"},
        {"revision": "b" * 40},
        {"id": "other"},
    ):
        invalid = copy.deepcopy(engine)
        invalid["model"].update(update)
        with pytest.raises(ValueError):
            reference_readout(source, invalid)
    engine["complete"] = False
    with pytest.raises(ValueError):
        reference_readout(source, engine)
