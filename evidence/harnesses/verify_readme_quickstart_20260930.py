"""Audit the README smoke artifacts without starting an engine."""

import hashlib
import json
import re
from pathlib import Path

from jev_runtime.schema import DecisionResponse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/readme-quickstart-20260930"
manifest = json.loads((BASE / "manifest.json").read_text())
for item in manifest["files"]:
    payload = (BASE / item["path"]).read_bytes()
    assert len(payload) == item["bytes"]
    assert hashlib.sha256(payload).hexdigest() == item["sha256"]

report = json.loads((BASE / "report.json").read_text())
assert report["passed"]
assert report["protected_before"] == report["protected_after"]
assert (BASE / "prepare_quickstart.py").read_bytes() == (
    ROOT / "examples/prepare_quickstart.py"
).read_bytes()
count = 0
for engine, row in report["engines"].items():
    assert row["passed"] and row["existing_output_rejected"]
    assert row["exit_code"] == 0 and not row["remaining_group_members"]
    assert row["quiescence"]["drained"]
    assert not any(row["quiescence"]["outstanding"].values())
    assert row["gpu_before"]["free_mib"] == row["gpu_after"]["free_mib"]
    assert row["registry_integrity"] == "ok"
    assert [r["bundle"] for r in row["decisions"]] == ["default@1", "demo@2", "default@1"]
    assert [r["generation"] for r in row["decisions"]] == [1, 2, 3]
    for response in row["decisions"] + [row["direct_http_decision"]]:
        parsed = DecisionResponse.model_validate(response)
        assert parsed.status == "completed" and parsed.engine["name"] == engine
        assert set(parsed.answers) == {"intent", "refund_requested"}
        assert parsed.answers["intent"].type == "choice"
        assert set(parsed.answers["intent"].probabilities) == {"billing", "technical", "other"}
        assert parsed.answers["refund_requested"].type == "boolean"
        assert set(parsed.answers["refund_requested"].probabilities) == {"true", "false"}
        count += 1
    doc = ROOT / ("README.md" if engine == "vllm" else "docs/quickstart-sglang.md")
    blocks = re.findall(r"```bash\n(.*?)\n```", doc.read_text(), re.S)
    prefix = "vllm serve" if engine == "vllm" else "python -m sglang.launch_server"
    command = next(block for block in blocks if block.startswith(prefix))
    flag = "--gpu-memory-utilization" if engine == "vllm" else "--mem-fraction-static"
    command = command.replace(f"{flag} 0.8", f"{flag} {row['memory_fraction']}")
    assert command == row["start_command_from_document"]

assert count == 8
print(json.dumps({"artifacts_verified": len(manifest["files"]), "responses_validated": count}))
