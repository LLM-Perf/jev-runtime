import asyncio
import math

from jev_runtime.backends.base import ScoreResult
from jev_runtime.schema import DecisionRequest, Option, Question, TemplateSpec


def test_branch_ids_cannot_prefix_another_branch(compiler, bundle):
    bundle = bundle.model_copy(update={"template": TemplateSpec(mode="independent-candidate")})
    ids = []
    for name in ("a", "a.1", "a.1.0", "a.10", "nested.question"):
        question = Question(
            id=name,
            type="choice",
            instruction="Choose",
            options=tuple(Option(id=f"c{i}", description=str(i)) for i in range(12)),
        )
        compiled = compiler.compile("input", question, bundle, "jev-" + "a" * 32)
        ids.extend(sequence.request_id for sequence in compiled.sequences)
    assert len(ids) == len(set(ids))
    assert all(not right.startswith(left) for left in ids for right in ids if left != right)


async def test_partial_failure_does_not_abort_a_dotted_sibling_question(
    runtime, bundle, monkeypatch
):
    independent = bundle.model_copy(
        update={"version": 2, "template": TemplateSpec(mode="independent-candidate")}
    )
    runtime.registry.upload(independent)
    await runtime.prepare(independent.reference)
    runtime.activate("model", independent.reference, 1)
    all_started, question_drained = asyncio.Event(), asyncio.Event()
    seen, cancelled, aborted = [], [], set()

    async def score(sequence):
        seen.append(sequence)
        if len(seen) == 3:
            all_started.set()
        await all_started.wait()
        if sequence.question_id == "a":
            if sequence.candidate_id == "y":
                raise RuntimeError("Injected failure in the second independent candidate")
            await asyncio.Event().wait()  # Parent question cleanup cancels this branch.
        await question_drained.wait()
        if sequence.request_id in aborted:
            raise RuntimeError("Engine prefix cancellation affected the sibling")
        return ScoreResult(sequence.request_id, (math.log(0.8), math.log(0.2)), 10, 1, 0)

    async def cancel(request_id):
        # This is the matching rule in the installed SGLang scheduler.
        aborted.update(item.request_id for item in seen if item.request_id.startswith(request_id))
        cancelled.append(request_id)
        if len(cancelled) >= 2:
            question_drained.set()

    monkeypatch.setattr(runtime.backend, "score", score)
    monkeypatch.setattr(runtime.backend, "cancel", cancel)
    request = DecisionRequest.model_validate(
        {
            "model": "model",
            "input": {"text": "sample"},
            "execution": {"allow_partial": True},
            "questions": [
                {
                    "id": "a",
                    "type": "choice",
                    "instruction": "Choose",
                    "options": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}],
                },
                {"id": "a.1", "type": "boolean", "instruction": "True?"},
            ],
        }
    )
    async with asyncio.timeout(3):
        result = await runtime.decide(request)
    assert result.status == "partial"
    assert result.answers["a"].status == "failed"
    assert result.answers["a.1"].status == "answered"
    assert result.answers["a.1"].value is True
    assert not runtime.registry.list()["leases"]
