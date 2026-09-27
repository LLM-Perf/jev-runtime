from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

from pydantic import Field

from jev_runtime.calibration import LabeledScores
from jev_runtime.config import Settings, build_runtime
from jev_runtime.schema import Bundle, Contract, TextInput, content_digest


class EvaluationSample(Contract):
    sample_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    input: TextInput
    labels: dict[str, str | bool]


def read_samples(path: Path) -> list[EvaluationSample]:
    rows = [
        EvaluationSample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows or len({r.sample_id for r in rows}) != len(rows):
        raise ValueError("Dataset requires non-empty unique sample IDs")
    return rows


async def collect_scores(
    settings: Settings, bundle: Bundle, samples: list[EvaluationSample]
) -> dict:
    if bundle.candidate_policy != "fixed" or bundle.template.mode != "joint-label":
        raise ValueError("This collector requires a fixed joint-label task bundle")
    if any(q.type == "rank" for q in bundle.questions):
        raise ValueError("Rank requires relevance judgments and ranking metrics")
    runtime = await build_runtime(settings)
    rows = []
    try:
        await runtime.start()
        runtime._validate_bundle(bundle)
        questions = {q.id: q for q in bundle.questions}
        for sample in samples:
            if set(sample.labels) != set(questions):
                raise ValueError("Each sample needs labels for every fixed question")
            for qid, question in questions.items():
                compiled = runtime.compiler.compile(
                    sample.input.text, question, bundle, "eval-" + content_digest(sample)[7:39]
                )
                target = sample.labels[qid]
                if question.type == "boolean":
                    if not isinstance(target, bool):
                        raise ValueError("Boolean evaluation requires JSON boolean targets")
                    target = "true" if target else "false"
                if target not in compiled.keys:
                    raise ValueError("Target must be one of the fixed business answer IDs")
                runtime._validate_sequences(compiled.sequences, bundle)
                seq = compiled.sequences[0]
                try:
                    async with asyncio.timeout(120):
                        result = await runtime.backend.score(seq)
                except BaseException:
                    await runtime.backend.cancel(seq.request_id)
                    raise
                row = LabeledScores(
                    content_digest([sample.sample_id, qid]),
                    sample.group_id,
                    result.logprobs,
                    compiled.keys.index(target),
                    qid,
                )
                row.validate()
                rows.append(asdict(row))
    finally:
        await runtime.close()
    return {
        "scoring_contract_digest": bundle.scoring_contract_digest,
        "source_dataset_digest": content_digest([s.model_dump(mode="json") for s in samples]),
        "model": bundle.model.model_dump(mode="json"),
        "rows": rows,
    }


def read_scores(document: dict, bundle: Bundle) -> list[LabeledScores]:
    if document["scoring_contract_digest"] != bundle.scoring_contract_digest:
        raise ValueError("Scores are bound to a different model/template/task contract")
    tasks = {q.id: 2 if q.type == "boolean" else len(q.options) for q in bundle.questions}
    rows = [
        LabeledScores(**{**row, "logprobs": tuple(row["logprobs"])}) for row in document["rows"]
    ]
    for row in rows:
        row.validate()
        if row.task_id not in tasks or len(row.logprobs) != tasks[row.task_id]:
            raise ValueError("Scored task or class dimension does not match the bundle")
    if {row.task_id for row in rows} != set(tasks):
        raise ValueError("Scores do not cover every fixed task")
    return rows
