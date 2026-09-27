from __future__ import annotations

from typing import Any

from pydantic import Field

from jev_runtime.errors import JevError
from jev_runtime.schema import (
    Contract,
    DecisionRequest,
    DecisionResponse,
    ExecutionOptions,
    Option,
    Question,
    TextInput,
)


class SystemOneRequest(Contract):
    model: str
    state: str
    questions: dict[str, dict[str, Any]]
    bundle: str | None = None
    timeout_ms: int = Field(default=30000, gt=0, le=300000)

    def to_decision(self) -> DecisionRequest:
        questions = []
        for qid, spec in self.questions.items():
            allowed = {"type", "instructions", "criteria", "levels", "values"}
            unknown = set(spec) - allowed
            if unknown:
                raise JevError("unsupported_field", "Unsupported System One question fields")
            kind = str(spec.get("type", "")).lower()
            instruction = spec.get("instructions")
            if not isinstance(instruction, str) or not instruction.strip():
                raise JevError("invalid_question", "Every question requires instructions")
            if kind in {"noul", "boolean"}:
                if spec.get("criteria") or spec.get("levels"):
                    raise JevError(
                        "unsupported_field",
                        "Noul criteria are not supported by this compatibility version",
                    )
                questions.append(Question(id=qid, type="boolean", instruction=instruction))
            elif kind in {"choice", "rank"}:
                criteria = spec.get("criteria")
                if not isinstance(criteria, dict) or not all(
                    isinstance(v, str) for v in criteria.values()
                ):
                    raise JevError(
                        "invalid_criteria", "Choice and Rank criteria must map IDs to descriptions"
                    )
                questions.append(
                    Question(
                        id=qid,
                        type=kind,
                        instruction=instruction,
                        options=tuple(Option(id=k, description=v) for k, v in criteria.items()),
                    )
                )
            elif kind == "score":
                levels = spec.get("levels")
                if not isinstance(levels, list) or not all(isinstance(v, str) for v in levels):
                    raise JevError(
                        "invalid_levels", "Score levels must be an ordered list of descriptions"
                    )
                values = spec.get("values")
                if values is not None and (
                    not isinstance(values, list) or len(values) != len(levels)
                ):
                    raise JevError("invalid_values", "Score values must match the levels")
                questions.append(
                    Question(
                        id=qid,
                        type="score",
                        instruction=instruction,
                        options=tuple(
                            Option(id=str(i), description=v, value=values[i] if values else None)
                            for i, v in enumerate(levels)
                        ),
                    )
                )
            else:
                raise JevError(
                    "unsupported_type", "Supported types are choice, noul, score and rank"
                )
        return DecisionRequest(
            model=self.model,
            bundle=self.bundle,
            input=TextInput(text=self.state),
            questions=tuple(questions),
            execution=ExecutionOptions(timeout_ms=self.timeout_ms),
        )


def from_decision(response: DecisionResponse) -> dict:
    answers = {}
    for key, answer in response.answers.items():
        item = answer.model_dump(mode="json", exclude_none=True)
        if answer.type == "choice":
            item["choice"] = answer.value
        elif answer.type == "score":
            if isinstance(answer.value, float):
                item["score"] = answer.value
            else:
                item["level"] = answer.value
        elif answer.type == "boolean":
            item["type"] = "noul"
            item["p_true"] = (answer.probabilities or {}).get("true")
        answers[key] = item
    return {
        **response.model_dump(mode="json", exclude={"answers"}),
        "answers": answers,
        "compatibility": "jev-runtime-systemone-v1",
    }
