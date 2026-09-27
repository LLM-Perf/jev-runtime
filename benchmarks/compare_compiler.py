"""Compare exact encodings and CPU compile time with a frozen source implementation.

Use retained serving fixtures and a local tokenizer whose implementation digest
matches those fixtures. No model weights or GPU engine are loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from jev_runtime.compiler import Compiler
from jev_runtime.schema import Bundle, Question


def run(args):
    if args.output.exists():
        raise ValueError("Output exists")
    fixtures = json.loads(args.fixtures.read_text())
    if not isinstance(fixtures, list):
        fixtures = [fixtures]
    spec = importlib.util.spec_from_file_location("jev_reference_compiler", args.reference_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    implementations = {
        "reference": module.Compiler(tokenizer, cache_entries=0),
        "candidate": Compiler(tokenizer, cache_entries=0),
    }
    report = {
        "source_commit": args.source_commit,
        "reference_commit": args.reference_commit,
        "reference_file_sha256": sha256(args.reference_source.read_bytes()).hexdigest(),
        "fixture_file_sha256": sha256(args.fixtures.read_bytes()).hexdigest(),
        "fixture_count": len(fixtures),
        "encoding_cache_entries": 0,
        "environment": {
            k: os.environ.get(k) for k in ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS")
        },
        "cpu_affinity_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "repeats": args.repeats,
        "passed": False,
        "measurements": [],
        "qualification": (
            "CPU full task compilation, no encoding cache; exact serving-token "
            "identity required; no GPU/throughput claim"
        ),
    }
    cases = []
    try:
        for f in fixtures:
            bundle = Bundle.model_validate(f["bundle"])
            questions = [Question.model_validate(q) for q in f["request"]["questions"]]
            for compiler in implementations.values():
                assert (
                    compiler.tokenizer_implementation_digest == f["tokenizer_implementation_digest"]
                )
                compiler.verify_bundle(bundle)
            cases.append((f, bundle, questions))

        def compile_case(compiler, case):
            fixture, bundle, questions = case
            return [
                sequence
                for q in questions
                for sequence in compiler.compile(
                    fixture["request"]["input"]["text"], q, bundle, "cpu-check"
                ).sequences
            ]

        for case in cases:
            expected = case[0]["sequences"]
            for name, compiler in implementations.items():
                actual = compile_case(compiler, case)
                assert len(actual) == len(expected), name
                for left, right in zip(actual, expected, strict=True):
                    assert list(left.input_ids) == right["input_ids"], name
                    assert list(left.label_ids) == right["label_ids"], name
                    assert left.candidate_id == right["candidate_id"], name
        report["exact_serving_encoding_matches"] = len(fixtures)
        for repeat in range(args.repeats):
            order = list(implementations)
            if repeat % 2:
                order.reverse()
            for name in order:
                timings = []
                for case in cases:
                    started = time.perf_counter()
                    compile_case(implementations[name], case)
                    timings.append((time.perf_counter() - started) * 1000)
                report["measurements"].append(
                    {
                        "repeat": repeat + 1,
                        "implementation": name,
                        "samples": len(timings),
                        "mean_ms": float(np.mean(timings)),
                        "p50_ms": float(np.percentile(timings, 50)),
                        "p95_ms": float(np.percentile(timings, 95)),
                        "per_fixture_ms": timings,
                    }
                )
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "encodings": report.get("exact_serving_encoding_matches"),
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be between 1 and 20")
    run(args)
