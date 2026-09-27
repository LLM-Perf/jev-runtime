from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jev_runtime.errors import JevError
from jev_runtime.schema import Calibration, content_digest


@dataclass(frozen=True)
class LabeledScores:
    sample_id: str
    group_id: str
    logprobs: tuple[float, ...]
    label: int
    task_id: str = "default"

    def validate(self) -> None:
        if not self.sample_id or not self.group_id or not self.task_id:
            raise ValueError("Sample and leakage-group identifiers are required")
        if len(self.logprobs) < 2 or not 0 <= self.label < len(self.logprobs):
            raise ValueError("A target index must refer to one of at least two classes")
        if not all(math.isfinite(value) for value in self.logprobs):
            raise ValueError("All log probabilities must be finite")


def check_split(fit: list[LabeledScores], heldout: list[LabeledScores]) -> None:
    if not fit or not heldout:
        raise ValueError("Separate non-empty fit and held-out sets are required")
    for rows in (fit, heldout):
        for row in rows:
            row.validate()
        if len({row.sample_id for row in rows}) != len(rows):
            raise ValueError("Duplicate sample IDs within a split")
    if {r.sample_id for r in fit} & {r.sample_id for r in heldout}:
        raise ValueError("Sample leakage between calibration and held-out evaluation")
    if {r.group_id for r in fit} & {r.group_id for r in heldout}:
        raise ValueError("Group leakage between calibration and held-out evaluation")
    if {r.task_id for r in fit} != {r.task_id for r in heldout}:
        raise ValueError("Fit and held-out sets must cover the same tasks")
    dimensions = {}
    for row in fit + heldout:
        width = dimensions.setdefault(row.task_id, len(row.logprobs))
        if width != len(row.logprobs):
            raise ValueError("Class dimension changed within the same task")


def _probabilities(row: LabeledScores, temperature: float, bias: float = 0) -> np.ndarray:
    scores = np.asarray(row.logprobs, dtype=np.float64) / temperature
    if bias:
        if len(scores) != 2:
            raise ValueError("Platt bias only applies to binary distributions")
        scores[0] += bias
    scores -= scores.max()
    result = np.exp(scores)
    return result / result.sum()


def quality_metrics(
    rows: list[LabeledScores], temperature: float = 1, bias: float = 0, bins: int = 15
) -> dict:
    if (
        not rows
        or not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(bias)
        or bins < 1
    ):
        raise ValueError("Metrics require samples, positive temperature and bins")
    confidence, correctness, nll, brier = [], [], [], []
    per_class: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        row.validate()
        probs = _probabilities(row, temperature, bias)
        predicted = int(probs.argmax())
        confidence.append(float(probs.max()))
        correctness.append(int(predicted == row.label))
        nll.append(-math.log(max(float(probs[row.label]), np.finfo(np.float64).tiny)))
        truth = np.zeros_like(probs)
        truth[row.label] = 1
        brier.append(float(np.sum((probs - truth) ** 2)))
        for label in range(len(probs)):
            counts = per_class.setdefault((row.task_id, label), [0, 0, 0])
            counts[0] += int(predicted == label and row.label == label)
            counts[1] += int(predicted == label and row.label != label)
            counts[2] += int(predicted != label and row.label == label)
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness)
    histogram = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        accuracy = float(correctness[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(rows) * abs(accuracy - conf)
        histogram.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "confidence": conf,
            }
        )
    curves = []
    # Threshold selection belongs to a validation set, never this held-out report.
    for threshold in (0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        errors = int((1 - correctness[accepted]).sum())
        curves.append(
            {
                "threshold": threshold,
                "accepted": count,
                "coverage": count / len(rows),
                "errors": errors,
                "risk": errors / count if count else None,
                "risk_wilson_95": wilson_interval(errors, count) if count else None,
            }
        )
    f1s = [
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0 for tp, fp, fn in per_class.values()
    ]
    return {
        "samples": len(rows),
        "accuracy": float(correctness.mean()),
        "accuracy_wilson_95": wilson_interval(int(correctness.sum()), len(rows)),
        "macro_f1": float(np.mean(f1s)),
        "nll": float(np.mean(nll)),
        "brier_multiclass_sum": float(np.mean(brier)),
        "ece": ece,
        "ece_bins": histogram,
        "risk_coverage": curves,
    }


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= count or count == 0:
        raise ValueError("Invalid binomial counts")
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - half), min(1, center + half)]


def fit_temperature(
    fit: list[LabeledScores], heldout: list[LabeledScores], scoring_contract_digest: str
) -> tuple[Calibration, dict]:
    check_split(fit, heldout)

    def objective(log_temperature):
        temperature = math.exp(log_temperature)
        losses = []
        for row in fit:
            values = np.asarray(row.logprobs) / temperature
            peak = values.max()
            losses.append(float(peak + np.log(np.exp(values - peak).sum()) - values[row.label]))
        return math.fsum(losses) / len(losses)

    # Bounded scalar optimization, with endpoints and identity as explicit candidates.
    low, high = math.log(0.01), math.log(100)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    fl, fr = objective(left), objective(right)
    for _ in range(80):
        if fl < fr:
            high, right, fr = right, left, fl
            left = high - ratio * (high - low)
            fl = objective(left)
        else:
            low, left, fl = left, right, fr
            right = low + ratio * (high - low)
            fr = objective(right)
    candidates = [math.log(0.01), math.log(100), 0, (low + high) / 2]
    temperature = min(100.0, max(0.01, math.exp(min(candidates, key=objective))))
    dataset_digest = content_digest(
        {
            "fit": [vars(row) for row in fit],
            "heldout": [vars(row) for row in heldout],
        }
    )
    report = {
        "method": "temperature",
        "temperature": temperature,
        "scoring_contract_digest": scoring_contract_digest,
        "dataset_digest": dataset_digest,
        "fit_samples": len(fit),
        "heldout_uncalibrated": quality_metrics(heldout),
        "heldout_calibrated": quality_metrics(heldout, temperature),
        "selection": "temperature chosen using fit split only",
    }
    artifact = Calibration(
        method="temperature",
        temperature=temperature,
        contract_digest=scoring_contract_digest,
        dataset_digest=dataset_digest,
        report_digest=content_digest(report),
    )
    return artifact, report


def bind_calibration(bundle, artifact: Calibration):
    if bundle.candidate_policy != "fixed":
        raise JevError("calibration_scope", "This fitter requires a fixed task definition")
    # Full validation prevents reusing an artifact with another model/template/task.
    return type(bundle).model_validate(
        {**bundle.model_dump(), "calibration": artifact.model_dump()}
    )


def fit_platt(
    fit: list[LabeledScores], heldout: list[LabeledScores], scoring_contract_digest: str
) -> tuple[Calibration, dict]:
    """Fit positive-slope binary scaling using fit data only, with bounded Newton steps."""
    check_split(fit, heldout)
    if any(len(row.logprobs) != 2 for row in fit + heldout):
        raise ValueError("Platt calibration requires exactly two classes")
    x = np.asarray([r.logprobs[0] - r.logprobs[1] for r in fit], dtype=np.float64)
    y = np.asarray([int(r.label == 0) for r in fit], dtype=np.float64)
    design = np.column_stack((x, np.ones_like(x)))
    parameters = np.asarray([1.0, 0.0])

    def objective(value):
        z = design @ value
        return float(np.mean(np.logaddexp(0, z) - y * z))

    loss = objective(parameters)
    iterations = 0
    for _ in range(100):
        iterations += 1
        z = design @ parameters
        probability = np.exp(-np.logaddexp(0, -z))
        gradient = design.T @ (probability - y) / len(fit)
        weights = probability * (1 - probability)
        hessian = (design.T * weights) @ design / len(fit) + np.eye(2) * 1e-10
        step = np.linalg.solve(hessian, gradient)
        rate = 1.0
        accepted = False
        for _ in range(40):
            candidate = np.clip(parameters - rate * step, [0.01, -100], [100, 100])
            candidate_loss = objective(candidate)
            if candidate_loss < loss - 1e-12:
                parameters, loss, accepted = candidate, candidate_loss, True
                break
            rate *= 0.5
        if not accepted:
            break
    temperature, bias = float(1 / parameters[0]), float(parameters[1])
    dataset_digest = content_digest(
        {"fit": [vars(r) for r in fit], "heldout": [vars(r) for r in heldout]}
    )
    report = {
        "method": "platt",
        "temperature": temperature,
        "bias": bias,
        "scoring_contract_digest": scoring_contract_digest,
        "dataset_digest": dataset_digest,
        "fit_samples": len(fit),
        "iterations": iterations,
        "fit_nll": loss,
        "heldout_uncalibrated": quality_metrics(heldout),
        "heldout_calibrated": quality_metrics(heldout, temperature, bias),
        "selection": "positive slope and bias chosen using fit split only",
    }
    artifact = Calibration(
        method="platt",
        temperature=temperature,
        bias=bias,
        contract_digest=scoring_contract_digest,
        dataset_digest=dataset_digest,
        report_digest=content_digest(report),
    )
    return artifact, report
