"""Strict cohort accounting shared by the GPU benchmark and soak runners."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from jev_runtime.schema import Contract
from jev_runtime.telemetry import STAGES, Stage


class Measurement(Contract):
    request_id: str
    started_ms: float = Field(ge=0)
    finished_ms: float = Field(ge=0)
    outcome: Literal["completed", "partial", "failed"]
    questions: int = Field(ge=1)
    successful_questions: int = Field(ge=0)
    scoring_sequences: int = Field(ge=1)
    logical_prompt_tokens: int = Field(ge=0)
    engine_prompt_tokens: int | None = Field(default=None, ge=0)
    engine_completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    cached_prompt_tokens: int | None = Field(default=None, ge=0)
    response_content_bytes: int | None = Field(default=None, ge=0, strict=True)
    finish_reason: str | None = Field(default=None, max_length=64)
    selected_candidate: str | None = Field(default=None, min_length=1, max_length=128)
    abstained: bool | None = Field(default=None, strict=True)
    selected_candidate_matches_reference: bool | None = Field(default=None, strict=True)
    runtime_timings_ms: dict[Stage, Annotated[float, Field(ge=0, allow_inf_nan=False)]] | None = (
        None
    )
    input_fixture_index: int | None = Field(default=None, ge=0, strict=True)
    error_code: str | None = None
    http_status: int | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.finished_ms < self.started_ms:
            raise ValueError("Completion precedes dispatch")
        if self.successful_questions > self.questions:
            raise ValueError("Successful questions exceed the request size")
        complete = self.successful_questions == self.questions
        partial = 0 < self.successful_questions < self.questions
        if (self.outcome == "completed") != complete or (self.outcome == "partial") != partial:
            raise ValueError("Outcome disagrees with strict question accounting")
        if self.abstained is True and self.selected_candidate is not None:
            raise ValueError("Abstentions cannot contain a selected candidate")
        if self.selected_candidate_matches_reference is not None and (
            self.selected_candidate is None or self.outcome != "completed"
        ):
            raise ValueError("Reference agreement requires a completed selected candidate")
        if (
            self.cached_prompt_tokens is not None
            and self.engine_prompt_tokens is not None
            and self.cached_prompt_tokens > self.engine_prompt_tokens
        ):
            raise ValueError("Cached tokens exceed observed prompt tokens")
        return self


def _latencies(rows: list[Measurement]) -> dict:
    values = [row.finished_ms - row.started_ms for row in rows]
    return {
        "samples": len(values),
        "p50_ms": float(np.percentile(values, 50)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
        # Predeclare a denominator requirement; do not advertise a stable tail
        # using a few dozen requests. Even a large descriptive P99 is not a CI.
        "p99_ms": float(np.percentile(values, 99)) if len(values) >= 10000 else None,
        "p99_minimum_samples": 10000,
        "statistics": "descriptive, not confidence intervals",
    }


def _generation_cost(rows: list[Measurement]) -> dict:
    observed = [
        row.engine_completion_tokens for row in rows if row.engine_completion_tokens is not None
    ]
    content = [row.response_content_bytes for row in rows if row.response_content_bytes is not None]
    return {
        "request_denominator": len(rows),
        "requests_with_observed_completion_tokens": len(observed),
        "observed_completion_tokens": sum(observed),
        "full_cohort_completion_tokens": sum(observed) if len(observed) == len(rows) else None,
        "requests_with_observed_content_bytes": len(content),
        "observed_content_bytes": sum(content),
        "finish_reasons": dict(Counter(row.finish_reason or "unknown" for row in rows)),
    }


def _runtime_timings(rows: list[Measurement]) -> dict:
    stages = {}
    for stage in STAGES:
        values = [
            row.runtime_timings_ms[stage]
            for row in rows
            if row.runtime_timings_ms is not None and stage in row.runtime_timings_ms
        ]
        stages[stage] = {
            "observed_requests": len(values),
            "mean_ms": float(np.mean(values)) if values else None,
            "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
            "p99_ms": float(np.percentile(values, 99)) if len(values) >= 10000 else None,
        }
    return {
        "request_denominator": len(rows),
        "stages": stages,
        "qualification": (
            "Server-observed runtime wall time; excludes HTTP parsing/serialization/network. "
            "Serial phases partition each observed total; percentiles must not be added. "
            "Missing phases/headers remain unobserved."
        ),
    }


def summarize_cohort(
    rows: list[Measurement], window_seconds: float, makespan_seconds: float
) -> dict:
    """Warmup is excluded. Dispatch closes at window end; drain is counted separately."""
    if not 0 < window_seconds <= makespan_seconds:
        raise ValueError("Measurement window and makespan must be positive and ordered")
    if len({row.request_id for row in rows}) != len(rows):
        raise ValueError("Each attempt needs a unique request ID; retries are separate attempts")
    if any(row.started_ms >= window_seconds * 1000 for row in rows):
        raise ValueError("A cohort contains requests dispatched outside its measurement window")
    if any(row.finished_ms > makespan_seconds * 1000 + 1e-6 for row in rows):
        raise ValueError("Makespan omits outstanding request drain time")
    completed = [row for row in rows if row.outcome == "completed"]
    in_window = [row for row in completed if row.finished_ms <= window_seconds * 1000]
    observed_cache = [
        row
        for row in completed
        if row.engine_prompt_tokens is not None and row.cached_prompt_tokens is not None
    ]
    # observed_cache is pre-filtered for observed tokens; the filters only narrow.
    observed_prompt = sum(
        tokens for row in observed_cache if (tokens := row.engine_prompt_tokens) is not None
    )
    observed_cached = sum(
        tokens for row in observed_cache if (tokens := row.cached_prompt_tokens) is not None
    )
    observed_ratio = observed_cached / observed_prompt if observed_prompt else None
    compared = [row for row in completed if row.selected_candidate_matches_reference is not None]
    return {
        "dispatched_requests": len(rows),
        "strict_success_requests": len(completed),
        "partial_requests": sum(row.outcome == "partial" for row in rows),
        "failed_requests": sum(row.outcome == "failed" for row in rows),
        "successful_questions_including_partial": sum(row.successful_questions for row in rows),
        "window_seconds": window_seconds,
        "drain_seconds": makespan_seconds - window_seconds,
        "makespan_seconds": makespan_seconds,
        "strict_success_rps_over_full_cohort": len(completed) / makespan_seconds,
        "strict_success_rps_completed_in_window": len(in_window) / window_seconds,
        "successful_questions_per_second_over_full_cohort": (
            sum(row.successful_questions for row in rows) / makespan_seconds
        ),
        "latency_successful_requests": _latencies(completed),
        "latency_all_attempts": _latencies(rows),
        "input_fixtures": {
            "observed_attempts": sum(row.input_fixture_index is not None for row in rows),
            "counts": dict(
                Counter(
                    str(row.input_fixture_index)
                    for row in rows
                    if row.input_fixture_index is not None
                )
            ),
            "qualification": (
                "Per-dispatch round-robin selection; unequal completed rates "
                "can produce unequal fixture frequencies"
            ),
        },
        "outcomes": dict(Counter(row.outcome for row in rows)),
        "errors": dict(
            Counter(row.error_code or "unspecified" for row in rows if row.outcome != "completed")
        ),
        "http_statuses": dict(Counter(str(row.http_status) for row in rows)),
        "runtime_timings": {
            "all_attempts": _runtime_timings(rows),
            "strict_successes": _runtime_timings(completed),
        },
        "output_cost": {
            "all_attempts": _generation_cost(rows),
            "strict_successes": _generation_cost(completed),
        },
        "decision_outputs": {
            "strict_success_denominator": len(completed),
            "requests_with_selected_candidate": sum(
                row.selected_candidate is not None for row in completed
            ),
            "candidate_counts": dict(
                Counter(
                    row.selected_candidate
                    for row in completed
                    if row.selected_candidate is not None
                )
            ),
            "abstained_requests": sum(row.abstained is True for row in completed),
            "requests_with_abstention_status": sum(row.abstained is not None for row in completed),
            "reference_comparison_denominator": len(compared),
            "matches_label_reference": sum(
                row.selected_candidate_matches_reference is True for row in compared
            ),
            "qualification": "agreement with a readout reference, not accuracy against labels",
        },
        "cache": {
            "scope": "strictly successful requests in this cohort",
            "requests_with_observed_tokens": len(observed_cache),
            "request_denominator": len(completed),
            "observed_prompt_tokens": observed_prompt,
            "observed_cached_tokens": observed_cached,
            "observed_token_weighted_ratio": observed_ratio,
            "full_cohort_token_weighted_ratio": (
                observed_ratio if len(observed_cache) == len(completed) else None
            ),
        },
    }
