"""Public fixed-task HTTP evaluation and live calibration lifecycle validation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

from jev_runtime.calibration import (
    LabeledScores,
    _probabilities,
    bind_calibration,
    fit_platt,
    fit_temperature,
)
from jev_runtime.schema import Bundle, DecisionResponse, Option, Question


def question_for(name):
    if name == "ag_news":
        return Question(
            id="topic",
            type="choice",
            instruction="Classify the topic of this news text.",
            options=tuple(
                Option(id=key, description=description)
                for key, description in zip(
                    ("world", "sports", "business", "science"),
                    (
                        "World news and politics",
                        "Sports",
                        "Business and finance",
                        "Science and technology",
                    ),
                    strict=True,
                )
            ),
        ), ("world", "sports", "business", "science")
    if name == "sst2":
        return Question(
            id="positive",
            type="boolean",
            instruction="Does this movie review express positive sentiment?",
        ), ("true", "false")
    raise ValueError("Unknown frozen task")


def load_rows(root, name, split, manifest):
    data = (root / name / f"{split}.jsonl").read_bytes()
    expected = manifest["datasets"][name]["derived"][split]
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("Dataset hash differs from the frozen manifest")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != expected["samples"]:
        raise ValueError("Dataset denominator differs from manifest")
    return rows


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve previous evidence; choose a new output directory")
    args.output.mkdir(parents=True)
    config = json.loads((args.run_dir / "config.json").read_text())
    process = json.loads((args.run_dir / "process.json").read_text())
    keys = json.loads((args.run_dir / "keys.json").read_text())
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    prefix = "" if process["mode"] == "gateway" else "/plugins/jev-runtime"
    url = f"http://127.0.0.1:{process['port']}" + prefix + "/"
    report = {
        "source_commit": args.source_commit,
        "provided_runtime_source_commit": args.runtime_source_commit,
        "started_at": time.time(),
        "qualification": "public engineering checks; not business acceptance",
        "release_gate_passed": False,
        "pretraining_disjointness_verified": False,
        "dataset_manifest": manifest,
        "engine": config["backend"],
        "mode": process["mode"],
        "process_identity": process["identity"],
        "tasks": {},
        "score_representation": "log conditional label probabilities; common logit offset removed",
        "metric_definition": (
            "probability argmax metrics; actual API abstentions reported separately"
        ),
    }
    async with (
        httpx.AsyncClient(
            base_url=url, timeout=130, headers={"Authorization": "Bearer " + keys["api"]}
        ) as client,
        httpx.AsyncClient(
            base_url=url,
            timeout=130,
            headers={"Authorization": "Bearer " + keys["admin"]},
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as admin,
    ):

        async def manage(path, body=None):
            response = await (admin.get(path) if body is None else admin.post(path, json=body))
            response.raise_for_status()
            return response.json()

        async def prepare(bundle):
            await manage("admin/bundles", bundle.model_dump(mode="json"))
            async with asyncio.timeout(180):
                while True:
                    await manage("admin/bundles/prepare", {"reference": bundle.reference})
                    workers = (await manage("admin/workers"))["workers"]
                    serving = [
                        w
                        for w in workers
                        if w["state"] == "SERVING" and w["owner_status"] != "dead"
                    ]
                    if serving and all(bundle.reference in w["prepared"] for w in serving):
                        return

        profile = await manage("admin/profile")
        report["serving_profile"] = profile
        try:
            for name in manifest["datasets"]:
                result = {"splits": {}, "execution_complete": False}
                report["tasks"][name] = result
                question, order = question_for(name)
                bundle = Bundle(
                    id="quality-" + name + "-" + uuid.uuid4().hex,
                    version=1,
                    model=profile["model"],
                    candidate_policy="fixed",
                    questions=(question,),
                )
                result["uncalibrated_bundle"] = bundle.model_dump(mode="json")
                data = {
                    split: load_rows(args.data_root, name, split, manifest)
                    for split in ("fit", "heldout")
                }
                if {r["group_id"] for r in data["fit"]} & {r["group_id"] for r in data["heldout"]}:
                    raise ValueError("Normalized group leakage")
                activated, generation, versions = False, 0, [bundle]
                scores = {}

                async def decide(sample, current_bundle, name=name, order=order, question=question):
                    response = await client.post(
                        "v1/decisions",
                        json={
                            "model": current_bundle.id,
                            "request_id": "quality-" + uuid.uuid4().hex,
                            "input": {"text": sample["text"]},
                            "execution": {"timeout_ms": 120000},
                        },
                    )
                    response.raise_for_status()
                    decision = DecisionResponse.model_validate(response.json())
                    answer = decision.answers[question.id]
                    probabilities = answer.probabilities or {}
                    assert (
                        decision.status == "completed"
                        and decision.bundle_digest == current_bundle.digest
                    )
                    assert set(probabilities) == set(order)
                    assert all(0 < p <= 1 and math.isfinite(p) for p in probabilities.values())
                    assert abs(math.fsum(probabilities.values()) - 1) < 1e-6
                    target = sample["label"] if name == "ag_news" else 1 - sample["label"]
                    row = LabeledScores(
                        sample["sample_id"],
                        sample["group_id"],
                        tuple(math.log(probabilities[key]) for key in order),
                        target,
                        question.id,
                    )
                    row.validate()
                    return row, answer

                try:
                    await prepare(bundle)
                    await manage(
                        "admin/bundles/activate",
                        {
                            "alias": bundle.id,
                            "reference": bundle.reference,
                            "expected_generation": 0,
                        },
                    )
                    activated, generation = True, 1
                    for split, samples in data.items():
                        scores[split] = []
                        counts = {
                            "attempted": 0,
                            "completed": 0,
                            "failed": 0,
                            "abstained": 0,
                            "correct_api_decisions": 0,
                        }
                        result["splits"][split] = counts
                        with (args.output / f"{name}-{split}.jsonl").open("x") as output:
                            for sample in samples:
                                counts["attempted"] += 1
                                record = {
                                    "sample_id": sample["sample_id"],
                                    "group_id": sample["group_id"],
                                    "label": sample["label"],
                                }
                                try:
                                    row, answer = await decide(sample, bundle)
                                    scores[split].append(row)
                                    counts["completed"] += 1
                                    counts["abstained"] += int(answer.abstained)
                                    expected = (
                                        order[row.label] if name == "ag_news" else row.label == 0
                                    )
                                    counts["correct_api_decisions"] += int(
                                        not answer.abstained and answer.value == expected
                                    )
                                    record.update(
                                        status="completed",
                                        scores=asdict(row),
                                        answer=answer.model_dump(mode="json"),
                                    )
                                except Exception as exc:
                                    counts["failed"] += 1
                                    record.update(
                                        status="failed",
                                        error={
                                            "type": type(exc).__name__,
                                            "message": str(exc)[:500],
                                        },
                                    )
                                output.write(json.dumps(record) + "\n")
                                output.flush()
                                if counts["attempted"] % 64 == 0:
                                    print(
                                        json.dumps({"task": name, "split": split, **counts}),
                                        flush=True,
                                    )
                        (args.output / "report.json").write_text(
                            json.dumps(report, indent=2) + "\n"
                        )
                    if any(counts["failed"] for counts in result["splits"].values()):
                        result["calibration_skipped"] = "Incomplete scoring cohort"
                        continue
                    fitter = fit_temperature if name == "ag_news" else fit_platt
                    artifact, metrics = fitter(
                        scores["fit"], scores["heldout"], bundle.scoring_contract_digest
                    )
                    result["calibration_report"] = metrics
                    calibrated = bind_calibration(bundle, artifact).model_copy(
                        update={"version": 2}
                    )
                    versions.append(calibrated)
                    await prepare(calibrated)
                    # Warm the exact same example before and after activation;
                    # do not compare cold-prefill and cached-kernel numerics.
                    sample = data["heldout"][0]
                    for _ in range(3):
                        reference, _ = await decide(sample, bundle)
                    await manage(
                        "admin/bundles/activate",
                        {
                            "alias": bundle.id,
                            "reference": calibrated.reference,
                            "expected_generation": generation,
                        },
                    )
                    generation += 1
                    for _ in range(3):
                        _, actual = await decide(sample, calibrated)
                    expected = _probabilities(reference, artifact.temperature, artifact.bias)
                    error = max(
                        abs(actual.probabilities[key] - float(expected[i]))
                        for i, key in enumerate(order)
                    )
                    result["live_calibration_parity"] = {
                        "sample_id": sample["sample_id"],
                        "max_absolute_error": error,
                        "tolerance": 1e-4,
                        "passed": error <= 1e-4,
                        "status": actual.calibration_status,
                    }
                    assert error <= 1e-4 and actual.calibration_status == "calibrated"
                    (args.output / f"{name}-calibrated-bundle.json").write_text(
                        calibrated.model_dump_json(indent=2) + "\n"
                    )
                    result["execution_complete"] = True
                finally:
                    if activated:
                        try:
                            await manage(
                                "admin/bundles/disable",
                                {"alias": bundle.id, "expected_generation": generation},
                            )
                            for version in versions:
                                await manage(
                                    "admin/bundles/retire", {"reference": version.reference}
                                )
                            result["retired_after_drain"] = True
                        except Exception as exc:
                            result["cleanup_error"] = type(exc).__name__
                            result["execution_complete"] = False
        except BaseException as exc:
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
            raise
        finally:
            report["execution_complete"] = len(report["tasks"]) == len(
                manifest["datasets"]
            ) and all(task["execution_complete"] for task in report["tasks"].values())
            report["finished_at"] = time.time()
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "execution_complete": report["execution_complete"],
                        "release_gate_passed": False,
                    }
                ),
                flush=True,
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    result = asyncio.run(run(parser.parse_args()))
    if not result["execution_complete"]:
        raise SystemExit(1)
