"""Revalidate complete saved DSW decisions with both SDKs; never contacts an engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from jev_runtime.schema import DecisionResponse

REQUIRED = {"request_id", "bundle_digest", "generation", "answers", "usage", "engine", "latency_ms"}


def collect(value, source: str, pointer=""):
    if isinstance(value, dict):
        if REQUIRED <= value.keys():
            yield {"source": source, "pointer": pointer, "response": value}
        else:
            for key, item in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                yield from collect(item, source, pointer + "/" + escaped)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from collect(item, source, pointer + "/" + str(index))


def replay(repo: Path, node: Path, output: Path):
    if output.exists():
        raise ValueError("Choose a new report path to retain previous results")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=repo).strip():
        raise ValueError("Commit source changes before producing revision-bound evidence")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    records = []
    sources = {}
    for path in sorted((repo / "evidence/dsw").rglob("*.json")):
        source = str(path.relative_to(repo))
        raw = path.read_bytes()
        found = list(collect(json.loads(raw), source))
        if found:
            records.extend(found)
            sources[source] = {"sha256": hashlib.sha256(raw).hexdigest(), "responses": len(found)}
    if not records:
        raise ValueError("No complete recorded decisions found")
    report = {
        "schema_version": 1,
        "source_commit": commit,
        "checked_at": time.time(),
        "qualification": "offline replay of saved response payloads; no new GPU execution",
        "selection_required_keys": sorted(REQUIRED),
        "sources": sources,
        "responses": len(records),
        "python": {"passed": 0, "failures": []},
        "typescript": {"status": "not_run"},
        "passed": False,
    }
    try:
        for item in records:
            try:
                DecisionResponse.model_validate(item["response"])
                report["python"]["passed"] += 1
            except ValueError as exc:
                report["python"]["failures"].append(
                    {"source": item["source"], "pointer": item["pointer"], "error": str(exc)}
                )
        package = repo / "packages/typescript"
        subprocess.run(
            [str(node), str(package / "node_modules/typescript/bin/tsc"), "-p", str(package)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        program = """
import {readFileSync} from 'node:fs';
const {parseDecision} = await import(process.argv[1]);
const records=JSON.parse(readFileSync(0,'utf8'));
const failures=[];
for(const {source,pointer,response} of records) {
  try { parseDecision(response); }
  catch(error) { failures.push({source,pointer,error:String(error)}); }
}
console.log(JSON.stringify({passed:records.length-failures.length,failures}));
"""
        result = subprocess.run(
            [
                str(node),
                "--input-type=module",
                "--eval",
                program,
                (package / "dist/index.js").as_uri(),
            ],
            input=json.dumps(records),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        report["typescript"] = json.loads(result.stdout)
        report["passed"] = not report["python"]["failures"] and not report["typescript"]["failures"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps({key: report[key] for key in ("source_commit", "responses", "passed")}))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        0 if replay(Path(__file__).resolve().parents[2], args.node, args.output) else 1
    )
