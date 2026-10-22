import json

import pytest
from pydantic import ValidationError

from jev_runtime import evaluation
from jev_runtime.config import Settings
from jev_runtime.schema import Bundle, Option, Question, TextInput


@pytest.mark.parametrize("fail_abort", [False, True])
async def test_collection_journals_before_dispatch_and_retains_unconfirmed_work(
    runtime, bundle, question, monkeypatch, fail_abort
):
    fixed = Bundle.model_validate(
        {
            **bundle.model_dump(),
            "id": "evaluation",
            "candidate_policy": "fixed",
            "questions": [question.model_dump()],
        }
    )

    async def build(settings):
        return runtime

    async def start():
        pass  # The fixture is already started.

    original = runtime.backend.score
    observed = []

    async def score(sequence):
        if sequence.question_id == question.id:
            rows = runtime.registry.recovery_candidates()
            assert any(sequence.request_id in row["engine_request_ids"] for row in rows)
            observed.append(sequence.request_id)
            if fail_abort:
                runtime.backend.fail_cancel = True
                raise RuntimeError("injected inference failure")
        return await original(sequence)

    monkeypatch.setattr(evaluation, "build_runtime", build)
    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr(runtime.backend, "score", score)
    settings = Settings(backend="sglang", model_id="fixture", model_revision="a" * 40)
    sample = evaluation.EvaluationSample(
        sample_id="public-id",
        group_id="group",
        input=TextInput(text="refund"),
        labels={question.id: question.options[0].id},
    )
    if fail_abort:
        with pytest.raises(RuntimeError, match="injected"):
            await evaluation.collect_scores(settings, fixed, [sample])
        row = runtime.registry.recovery_candidates()[0]
        assert row["phase"] == "abort_pending" and row["engine_request_ids"] == observed
    else:
        result = await evaluation.collect_scores(settings, fixed, [sample])
        assert len(result["rows"]) == 1 and not runtime.registry.list()["leases"]
    assert observed and all("public-id" not in value for value in observed)


def _settings():
    return Settings(backend="sglang", model_id="fixture", model_revision="a" * 40)


def _fixed_bundle(bundle, *questions):
    return Bundle.model_validate(
        {
            **bundle.model_dump(),
            "id": "evaluation",
            "candidate_policy": "fixed",
            "questions": [q.model_dump() for q in questions],
        }
    )


def _sample(question_id, label):
    return evaluation.EvaluationSample(
        sample_id="public-id",
        group_id="group",
        input=TextInput(text="refund"),
        labels={question_id: label},
    )


def _serve(monkeypatch, runtime):
    async def build(settings):
        return runtime

    async def start():
        pass  # The fixture is already started.

    monkeypatch.setattr(evaluation, "build_runtime", build)
    monkeypatch.setattr(runtime, "start", start)
    calls = []
    original = runtime.backend.score

    async def score(sequence):
        calls.append(sequence)
        return await original(sequence)

    monkeypatch.setattr(runtime.backend, "score", score)
    return calls


def _read_samples(tmp_path, labels):
    row = {
        "sample_id": "public-id",
        "group_id": "group",
        "input": {"text": "refund"},
        "labels": labels,
    }
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return evaluation.read_samples(path)


# Numeric 0/1 and 0.0/1.0 must not silently coerce to false/true ground truth.
_INVALID_LABELS = [0, 1, 0.0, 1.0, None, 2, ["true"], {"id": "true"}]


@pytest.mark.parametrize("value", _INVALID_LABELS)
def test_sample_labels_reject_non_string_non_boolean_values(value):
    with pytest.raises(ValidationError):
        evaluation.EvaluationSample(
            sample_id="s", group_id="g", input=TextInput(text="t"), labels={"q": value}
        )


@pytest.mark.parametrize("value", _INVALID_LABELS)
def test_read_samples_rejects_non_string_non_boolean_labels(tmp_path, value):
    with pytest.raises(ValidationError):
        _read_samples(tmp_path, {"q": value})


@pytest.mark.parametrize("value", [True, False])
def test_sample_labels_preserve_booleans(tmp_path, value):
    sample = _sample("q", value)
    assert type(sample.labels["q"]) is bool and sample.labels["q"] is value
    (loaded,) = _read_samples(tmp_path, {"q": value})
    assert type(loaded.labels["q"]) is bool and loaded.labels["q"] is value


@pytest.mark.parametrize("value", ["true", "false", "1", "0"])
def test_sample_labels_preserve_choice_id_strings(tmp_path, value):
    sample = _sample("q", value)
    assert type(sample.labels["q"]) is str and sample.labels["q"] == value
    (loaded,) = _read_samples(tmp_path, {"q": value})
    assert type(loaded.labels["q"]) is str and loaded.labels["q"] == value


@pytest.mark.parametrize("label,index", [(True, 0), (False, 1)])
async def test_collector_scores_boolean_targets(
    runtime, bundle, monkeypatch, tmp_path, label, index
):
    question = Question(id="flag", type="boolean", instruction="Is the refund approved?")
    fixed = _fixed_bundle(bundle, question)
    calls = _serve(monkeypatch, runtime)
    samples = _read_samples(tmp_path, {question.id: label})
    result = await evaluation.collect_scores(_settings(), fixed, samples)
    (row,) = result["rows"]
    assert row["task_id"] == question.id and row["label"] == index
    assert len(row["logprobs"]) == 2
    # Warmup calls from prepare share the question ID; sample dispatch uses eval- IDs.
    dispatched = [c for c in calls if c.request_id.startswith("eval-")]
    assert [c.question_id for c in dispatched] == [question.id]
    assert any(c.request_id.startswith("prepare-") for c in calls)


@pytest.mark.parametrize(
    "options,label,index",
    [
        (("true", "false"), "true", 0),
        (("true", "false"), "false", 1),
        (("1", "0"), "1", 0),
        (("1", "0"), "0", 1),
    ],
)
async def test_collector_accepts_choice_ids_matching_boolean_spellings(
    runtime, bundle, monkeypatch, tmp_path, options, label, index
):
    question = Question(
        id="intent",
        type="choice",
        instruction="Choose the relevant category",
        options=tuple(Option(id=oid, description=f"Answers about {oid}") for oid in options),
    )
    fixed = _fixed_bundle(bundle, question)
    calls = _serve(monkeypatch, runtime)
    (sample,) = _read_samples(tmp_path, {question.id: label})
    assert type(sample.labels[question.id]) is str
    result = await evaluation.collect_scores(_settings(), fixed, [sample])
    (row,) = result["rows"]
    assert row["task_id"] == question.id and row["label"] == index
    assert len(row["logprobs"]) == len(options)
    dispatched = [c for c in calls if c.request_id.startswith("eval-")]
    assert [c.question_id for c in dispatched] == [question.id]


@pytest.mark.parametrize("label", ["true", "false", "1", "0"])
async def test_collector_rejects_string_labels_for_boolean_questions(
    runtime, bundle, monkeypatch, tmp_path, label
):
    question = Question(id="flag", type="boolean", instruction="Is the refund approved?")
    fixed = _fixed_bundle(bundle, question)
    calls = _serve(monkeypatch, runtime)
    (sample,) = _read_samples(tmp_path, {question.id: label})
    assert type(sample.labels[question.id]) is str
    # This check fires before target-in-keys dispatch: "1"/"0" are not valid keys and
    # would otherwise fail with "Target must be one of the fixed business answer IDs".
    with pytest.raises(ValueError, match="Boolean evaluation requires JSON boolean targets"):
        await evaluation.collect_scores(_settings(), fixed, [sample])
    assert not any(c.request_id.startswith("eval-") for c in calls)
    assert not runtime.registry.list()["leases"]
