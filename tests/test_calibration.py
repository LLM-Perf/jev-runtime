import math

import numpy as np
import pytest
from pydantic import ValidationError

from jev_runtime.calibration import (
    LabeledScores,
    bind_calibration,
    fit_platt,
    fit_temperature,
    quality_metrics,
)
from jev_runtime.schema import Bundle


def rows(prefix):
    return [
        LabeledScores(
            f"{prefix}-{i}",
            f"{prefix}-group-{i}",
            (math.log(0.99), math.log(0.01)),
            int(i % 5 == 0),
        )
        for i in range(50)
    ]


@pytest.mark.parametrize(
    "label", [True, False, 0.5, 1.0, "1", None, [], {}, np.int64(1), np.bool_(True)]
)
def test_labeled_scores_rejects_non_python_integer_labels(label):
    sample = LabeledScores("sample", "group", (0, -1), label)
    with pytest.raises(ValueError, match="integer"):
        sample.validate()


@pytest.mark.parametrize("label", [-1, 2])
def test_labeled_scores_rejects_out_of_range_labels(label):
    with pytest.raises(ValueError, match="target index"):
        LabeledScores("sample", "group", (0, -1), label).validate()


@pytest.mark.parametrize("label", [True, 1.0])
def test_quality_metrics_rejects_non_integer_labels_before_indexing(label):
    with pytest.raises(ValueError, match="integer"):
        quality_metrics([LabeledScores("sample", "group", (0, -1), label)])


@pytest.mark.parametrize("fitter", [fit_temperature, fit_platt])
@pytest.mark.parametrize("split", ["fit", "heldout"])
@pytest.mark.parametrize("label", [True, 1.0])
def test_fitters_reject_non_integer_labels_in_either_split(bundle, fitter, split, label):
    samples = {
        "fit": [LabeledScores("fit", "fit-group", (0, -1), 0)],
        "heldout": [LabeledScores("heldout", "heldout-group", (0, -1), 0)],
    }
    samples[split] = [LabeledScores(split, f"{split}-group", (0, -1), label)]
    with pytest.raises(ValueError, match="integer"):
        fitter(samples["fit"], samples["heldout"], bundle.scoring_contract_digest)


@pytest.mark.parametrize("label", [0, 1])
def test_quality_metrics_preserves_integer_label_indexing(label):
    sample = LabeledScores("sample", "group", (math.log(0.75), math.log(0.25)), label)
    sample.validate()
    result = quality_metrics([sample])
    assert result["accuracy"] == int(label == 0)
    assert result["nll"] == pytest.approx(-sample.logprobs[label])


def test_temperature_fit_uses_separate_groups_and_improves_heldout(bundle, question):
    fixed = Bundle.model_validate(
        {**bundle.model_dump(), "candidate_policy": "fixed", "questions": [question.model_dump()]}
    )
    artifact, report = fit_temperature(rows("fit"), rows("heldout"), fixed.scoring_contract_digest)
    assert artifact.temperature > 1
    assert report["heldout_calibrated"]["nll"] < report["heldout_uncalibrated"]["nll"]
    assert bind_calibration(fixed, artifact).calibration == artifact
    changed = Bundle.model_validate(
        {**fixed.model_dump(), "model": {**fixed.model.model_dump(), "revision": "b" * 40}}
    )
    with pytest.raises(ValidationError):
        bind_calibration(changed, artifact)


def test_group_leakage_rejected(bundle):
    with pytest.raises(ValueError, match="leakage"):
        fit_temperature(rows("same"), rows("same"), bundle.scoring_contract_digest)


def test_quality_metric_counts_and_empty_coverage():
    result = quality_metrics(rows("test"), temperature=10)
    assert result["samples"] == 50
    assert result["accuracy"] == pytest.approx(0.8)
    assert sum(item["count"] for item in result["ece_bins"]) == 50
    assert result["risk_coverage"][-1]["risk"] is None


def test_platt_corrects_base_rate_without_heldout_label_selection(bundle):
    fit = [LabeledScores(f"fit-{i}", f"fit-{i}", (0, 0), int(i % 4 == 0)) for i in range(80)]
    heldout = [
        LabeledScores(f"heldout-{i}", f"heldout-{i}", (0, 0), int(i % 4 == 0)) for i in range(80)
    ]
    artifact, report = fit_platt(fit, heldout, bundle.scoring_contract_digest)
    assert artifact.bias == pytest.approx(math.log(3), abs=1e-4)
    assert report["heldout_calibrated"]["nll"] < report["heldout_uncalibrated"]["nll"]
    changed = [LabeledScores(r.sample_id, r.group_id, r.logprobs, 1 - r.label) for r in heldout]
    other, _ = fit_platt(fit, changed, bundle.scoring_contract_digest)
    assert artifact.temperature == other.temperature and artifact.bias == other.bias


def test_platt_rejects_multiclass_and_task_dimension_drift(bundle):
    bad = [LabeledScores("a", "a", (0, -1, -2), 0)]
    with pytest.raises(ValueError, match="dimension"):
        fit_temperature(bad, rows("heldout"), bundle.scoring_contract_digest)
    with pytest.raises(ValueError, match="two classes"):
        fit_platt(bad, [LabeledScores("b", "b", (0, -1, -2), 0)], bundle.scoring_contract_digest)


def test_macro_f1_keeps_different_task_class_namespaces_separate():
    samples = [
        LabeledScores("a", "a", (0, -1), 0, "task-a"),
        LabeledScores("b", "b", (-1, 0), 1, "task-b"),
    ]
    assert quality_metrics(samples)["macro_f1"] == pytest.approx(0.5)


def test_temperature_upper_boundary_produces_valid_artifact(bundle):
    fit = [LabeledScores("fit", "fit", (0, -10), 1)]
    heldout = [LabeledScores("heldout", "heldout", (0, -10), 1)]
    artifact, report = fit_temperature(fit, heldout, bundle.scoring_contract_digest)
    assert artifact.temperature == pytest.approx(100)
    assert artifact.temperature <= 100
    assert math.isfinite(report["heldout_calibrated"]["nll"])
