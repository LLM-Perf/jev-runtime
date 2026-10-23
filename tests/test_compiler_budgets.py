from unittest.mock import patch

import pytest

from jev_runtime.errors import JevError
from jev_runtime.schema import DecisionRequest, Option, Policy, Question, TemplateSpec, TextInput


@pytest.mark.parametrize("kind", ["choice", "rank", "score"])
async def test_impossible_candidate_budget_rejects_before_rendering(runtime, bundle, kind):
    bounded = bundle.model_copy(
        update={
            "version": 2,
            "template": TemplateSpec(mode="independent-candidate"),
            "policy": Policy(max_scoring_sequences=1),
        }
    )
    runtime.registry.upload(bounded)
    await runtime.prepare(bounded.reference)
    runtime.registry.activate("model", bounded.reference, 1)
    runtime.backend.calls.clear()
    question = Question(
        id="many",
        type=kind,
        instruction="Select a matching candidate",
        options=tuple(Option(id=f"o{i}", description=f"Candidate {i}") for i in range(64)),
    )
    request = DecisionRequest(
        model="model", input=TextInput(text="Example data"), questions=(question,)
    )
    tokenizer = runtime.compiler.tokenizer
    with (
        patch.object(
            tokenizer, "apply_chat_template", wraps=tokenizer.apply_chat_template
        ) as render,
        patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode,
        pytest.raises(JevError) as error,
    ):
        await runtime.decide(request)
    assert error.value.code == "branch_budget" and error.value.status_code == 413
    assert render.call_count == encode.call_count == 0
    assert runtime.backend.calls == []
    assert runtime.registry.list()["leases"] == []
    assert runtime.admission.snapshot()["expanded_tokens"] == 0
    assert runtime.is_prepared(bounded.reference)


@pytest.mark.parametrize(
    ("mode", "kind", "limit", "expected_sequences"),
    [
        ("joint-label", "choice", 1, 1),
        ("independent-candidate", "boolean", 1, 1),
        ("independent-candidate", "choice", 2, 2),
    ],
)
def test_compiler_budget_counts_sequences_not_labels(
    compiler, bundle, question, mode, kind, limit, expected_sequences
):
    bounded = bundle.model_copy(
        update={"template": TemplateSpec(mode=mode), "policy": Policy(max_scoring_sequences=limit)}
    )
    if kind == "boolean":
        question = Question(id="boolean", type="boolean", instruction="Is this relevant?")
    compiled = compiler.compile("Example", question, bounded, "request")
    assert len(compiled.sequences) == expected_sequences
    assert all(len(sequence.label_ids) == 2 for sequence in compiled.sequences)


@pytest.mark.parametrize("single_label", [False, True])
async def test_runtime_still_checks_combined_and_single_label_budgets(
    runtime, bundle, question, single_label
):
    bounded = bundle.model_copy(update={"policy": Policy(max_scoring_sequences=1)})
    if single_label:
        runtime.capabilities = runtime.capabilities.model_copy(update={"label_scoring": "single"})
        questions = (question,)
    else:
        questions = (question, question.model_copy(update={"id": "other"}))
    request = DecisionRequest(model="model", input=TextInput(text="Example"), questions=questions)
    with pytest.raises(JevError) as error:
        runtime._compile_request(request, bounded, "request")
    assert error.value.code == "branch_budget" and error.value.status_code == 413
    assert runtime.backend.calls == []
