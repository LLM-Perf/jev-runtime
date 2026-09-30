from __future__ import annotations

import math
from typing import Literal

from jev_runtime.backends.base import ScoreResult
from jev_runtime.compiler import CompiledQuestion
from jev_runtime.errors import JevError
from jev_runtime.schema import Answer, Bundle


def softmax(values: tuple[float, ...] | list[float], temperature: float = 1) -> list[float]:
    if not values or not all(math.isfinite(v) for v in values):
        raise JevError("invalid_scores", "Missing or non-finite model scores", 502)
    scaled = [v / temperature for v in values]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = math.fsum(exps)
    return [v / total for v in exps]


def assemble(compiled: CompiledQuestion, results: list[ScoreResult], bundle: Bundle) -> Answer:
    if len(results) != len(compiled.sequences):
        raise JevError("missing_branches", "Not all candidate branches completed", 502)
    for expected, actual in zip(compiled.sequences, results, strict=True):
        if expected.request_id != actual.request_id or len(actual.logprobs) != len(
            expected.label_ids
        ):
            raise JevError("score_contract", "Engine returned mismatched scores", 502)
        if not all(math.isfinite(v) for v in actual.logprobs):
            raise JevError("invalid_scores", "Engine returned non-finite scores", 502)
        if actual.raw_logprobs and any(v > 1e-5 for v in actual.logprobs):
            raise JevError("score_contract", "A raw log probability cannot be positive", 502)
    if compiled.label_groups:
        if sum(compiled.label_groups) != len(results) or any(
            len(result.logprobs) != 1 or not result.raw_logprobs for result in results
        ):
            raise JevError("score_contract", "Invalid single-label readout groups", 502)
        grouped = []
        start = 0
        for count in compiled.label_groups:
            group = results[start : start + count]
            grouped.append(ScoreResult(group[0].request_id, tuple(r.logprobs[0] for r in group)))
            start += count
        results = grouped
    question = compiled.question
    calibration = bundle.calibration
    temperature = calibration.temperature if calibration else 1
    bias = calibration.bias if calibration and calibration.method == "platt" else 0
    support = None
    label_mass = None
    semantics: Literal["conditional_label_distribution", "normalized_support"]
    independent = compiled.mode == "independent-candidate" and question.type != "boolean"
    if independent:
        scores = [
            softmax([r.logprobs[0] + bias * temperature, r.logprobs[1]], temperature)[0]
            for r in results
        ]
        total = math.fsum(scores)
        if total <= 0:
            raise JevError("invalid_scores", "All candidate support underflowed to zero", 502)
        probabilities = [s / total for s in scores]
        support = dict(zip(compiled.keys, scores, strict=True))
        semantics = "normalized_support"
    else:
        values = list(results[0].logprobs)
        if bias:
            values[0] += bias * temperature
        probabilities = softmax(values, temperature)
        semantics = "conditional_label_distribution"
        if results[0].raw_logprobs:
            label_mass = math.fsum(math.exp(v) for v in results[0].logprobs)
            if label_mass > 1.0001:
                raise JevError("score_contract", "Label mass exceeds one", 502)
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    ordered = sorted(probabilities, reverse=True)
    margin = ordered[0] - ordered[1]
    tie = margin <= bundle.policy.tie_epsilon
    reason = None
    if tie and bundle.policy.tie == "abstain":
        reason = "tie"
    elif bundle.policy.min_probability is not None and ordered[0] < bundle.policy.min_probability:
        reason = "probability_below_threshold"
    elif bundle.policy.min_margin is not None and margin < bundle.policy.min_margin:
        reason = "margin_below_threshold"
    elif support and bundle.policy.min_support is not None:
        if max(support.values()) < bundle.policy.min_support:
            reason = "support_below_threshold"
    levels = None
    value: str | bool | float | tuple[str, ...] | None = compiled.keys[best]
    if question.type == "boolean":
        value = best == 0
    elif question.type == "rank":
        value = tuple(
            compiled.keys[i]
            for i in sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
        )
    elif question.type == "score" and all(o.value is not None for o in question.options):
        # The all() guard above keeps every option value; the filter only narrows.
        levels = {
            o.id: float(option_value)
            for o in question.options
            if (option_value := o.value) is not None
        }
        value = math.fsum(
            p * levels[key] for key, p in zip(compiled.keys, probabilities, strict=True)
        )
    return Answer(
        type=question.type,
        status="abstained" if reason else "answered",
        value=None if reason else value,
        probabilities=dict(zip(compiled.keys, probabilities, strict=True)),
        probability_semantics=semantics,
        support=support,
        label_mass=label_mass,
        calibration_status="calibrated" if calibration else "uncalibrated",
        abstained=reason is not None,
        reason=reason,
        levels=levels,
    )
