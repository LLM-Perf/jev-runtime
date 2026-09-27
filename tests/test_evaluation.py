import pytest

from jev_runtime import evaluation
from jev_runtime.config import Settings
from jev_runtime.schema import Bundle, TextInput


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
