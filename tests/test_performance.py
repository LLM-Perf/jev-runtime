import pytest
from pydantic import ValidationError

from jev_runtime.performance import Measurement, summarize_cohort


def row(name, start, end, outcome="completed", successes=1, questions=1, prompt=100, cached=50):
    return Measurement(
        request_id=name,
        started_ms=start,
        finished_ms=end,
        outcome=outcome,
        questions=questions,
        successful_questions=successes,
        scoring_sequences=questions,
        logical_prompt_tokens=100 * questions,
        engine_prompt_tokens=prompt,
        cached_prompt_tokens=cached,
        http_status=200,
    )


def test_strict_success_drain_and_token_weighting_have_explicit_denominators():
    rows = [
        row("a", 0, 2),
        row("b", 1, 5, prompt=200),
        row("partial", 1, 3, outcome="partial", questions=2),
        row("bad", 2, 4, outcome="failed", successes=0),
    ]
    summary = summarize_cohort(rows, window_seconds=0.003, makespan_seconds=0.005)
    assert summary["dispatched_requests"] == 4 and summary["strict_success_requests"] == 2
    assert summary["strict_success_rps_over_full_cohort"] == 400
    assert summary["strict_success_rps_completed_in_window"] == pytest.approx(1 / 0.003)
    assert summary["successful_questions_including_partial"] == 3
    assert summary["cache"]["full_cohort_token_weighted_ratio"] == pytest.approx(100 / 300)
    assert summary["latency_successful_requests"]["p99_ms"] is None


def test_missing_cache_observation_does_not_become_zero_or_full_coverage():
    summary = summarize_cohort([row("a", 0, 1), row("b", 0, 2, cached=None)], 0.003, 0.003)
    assert summary["cache"]["requests_with_observed_tokens"] == 1
    assert summary["cache"]["observed_token_weighted_ratio"] == 0.5
    assert summary["cache"]["full_cohort_token_weighted_ratio"] is None


def test_http_200_does_not_override_partial_outcome():
    with pytest.raises(ValidationError, match="strict question"):
        row("false-success", 0, 1, questions=2)


def test_cohort_rejects_omitted_drain_duplicate_ids_and_late_dispatch():
    with pytest.raises(ValueError, match="drain"):
        summarize_cohort([row("a", 0, 5)], 0.003, 0.004)
    with pytest.raises(ValueError, match="unique"):
        summarize_cohort([row("a", 0, 1), row("a", 0, 2)], 0.003, 0.003)
    with pytest.raises(ValueError, match="outside"):
        summarize_cohort([row("late", 3, 4)], 0.003, 0.004)


def test_fixture_coverage_keeps_failed_and_unobserved_attempts_distinct():
    rows = [
        row("native-0", 0, 1).model_copy(update={"input_fixture_index": 0}),
        row("native-1", 1, 2).model_copy(update={"input_fixture_index": 1}),
        row("error", 2, 3, outcome="failed", successes=0),
    ]
    coverage = summarize_cohort(rows, 0.004, 0.004)["input_fixtures"]
    assert coverage["observed_attempts"] == 2
    assert coverage["counts"] == {"0": 1, "1": 1}
