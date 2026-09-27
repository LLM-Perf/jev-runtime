import math

import pytest
from pydantic import ValidationError

from jev_runtime.calibration import (
    LabeledScores,
    bind_calibration,
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
