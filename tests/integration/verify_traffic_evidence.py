"""Recount every saved request and activation; never trust summary counts alone."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from jev_runtime.schema import Bundle
from tests.integration.hot_switch_traffic import verify_response


def verify(path: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    report = contract["checks"]["hot_switch_under_traffic"]
    assert contract["passed"] and report["passed"]
    assert path.name == report["evidence_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["evidence_sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    first, last = rows[0], rows[-1]
    assert first["kind"] == "configuration" and last["kind"] == "summary"
    assert {k: v for k, v in last.items() if k != "kind"} == {
        k: v for k, v in report.items() if k not in ("evidence_file", "evidence_sha256")
    }
    bundles = {b.reference: b for b in map(Bundle.model_validate, first["bundles"])}
    generation = first["starting_generation"]
    generations = {generation: first["starting_bundle"]}
    requests = []
    switches = 0
    for row in rows[1:-1]:
        if row["kind"] == "activation":
            assert row["generation"] == generation + 1
            assert row["bundle"] in bundles and row["bundle"] != generations[generation]
            generation = row["generation"]
            generations[generation] = row["bundle"]
            switches += 1
        else:
            assert row["kind"] == "request"
            requests.append(row)
    identifiers = set()
    inputs = set()
    indices = set()
    types = Counter()
    versions = Counter()
    latencies = []
    for row in requests:
        assert row["outcome"] == "response_validated_pending_generation_audit"
        request = row["request"]
        assert row["index"] not in indices
        indices.add(row["index"])
        assert request["model"] == first["payload"]["model"]
        assert request["questions"] == [
            first["payload"]["questions"][row["index"] % len(first["payload"]["questions"])]
        ]
        assert request["input"]["text"] not in inputs
        inputs.add(request["input"]["text"])
        result = verify_response(row["response"], request, bundles, first["engine"])
        assert result.request_id not in identifiers
        identifiers.add(result.request_id)
        assert generations[result.generation] == result.bundle
        types[request["questions"][0]["type"]] += 1
        versions[result.bundle] += 1
        assert row["latency_ms"] >= 0
        latencies.append(row["latency_ms"])
    assert indices == set(range(len(requests)))
    assert len(requests) == report["attempted_requests"] == report["strict_success_requests"]
    assert len(requests) == report["response_validated_requests"] >= report["minimum_requests"]
    assert switches == report["switches"] >= report["minimum_switches"]
    assert report["failed_requests"] == report["mixed_bundle_responses"] == 0
    assert report["unaccounted_requests"] == report["retries"] == 0
    assert dict(types) == report["requests_by_type"]
    assert dict(versions) == report["requests_by_bundle"]
    assert set(types) == {"choice", "boolean", "score", "rank"} and len(versions) == 2
    assert first["requirements"]["minimum_requests"] == report["minimum_requests"]
    assert first["requirements"]["minimum_switches"] == report["minimum_switches"]
    latencies.sort()
    return {
        "engine": first["engine"],
        "strict_success_requests": len(requests),
        "unique_request_ids": len(identifiers),
        "unique_inputs": len(inputs),
        "switches": switches,
        "requests_by_type": dict(types),
        "requests_by_bundle": dict(versions),
        "elapsed_seconds": report["elapsed_seconds"],
        "latency_ms": {
            "p50": latencies[len(latencies) // 2],
            "p99": latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)],
            "max": latencies[-1],
        },
        "qualification": "observed colocated functional traffic; not controlled performance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.responses, args.contract), indent=2))
