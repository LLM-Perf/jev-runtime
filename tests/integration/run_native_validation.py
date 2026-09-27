"""One isolated native validation attempt, with evidence and owned-group cleanup.

Run with the engine environment's Python. A fresh run directory is mandatory.
This is a colocated functional check, never a release performance certificate.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deployment import dsw_service  # noqa: E402


def save(path: Path, value: dict) -> None:
    with path.open("x") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def require_owner(record: dict) -> None:
    if dsw_service.process_identity(record["identity"]["pid"]) != record["identity"]:
        raise RuntimeError("Recorded engine exited or its process identity changed")


def group_members(pgid: int) -> list[int]:
    result = []
    for item in Path("/proc").iterdir():
        if not item.name.isdecimal():
            continue
        try:
            fields = (item / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            result.append(int(item.name))
    return sorted(result)


def wait_ready(record: dict, key: str, timeout: float) -> dict:
    started = time.monotonic()
    last = None
    with httpx.Client(timeout=2, headers={"Authorization": "Bearer " + key}) as client:
        while time.monotonic() - started < timeout:
            require_owner(record)
            try:
                response = client.get(
                    f"http://127.0.0.1:{record['port']}/plugins/jev-runtime/ready"
                )
                last = response.status_code
                if response.status_code == 200 and response.json().get("ready") is True:
                    require_owner(record)
                    return {"ready": True, "elapsed_seconds": time.monotonic() - started}
            except httpx.TransportError as exc:
                last = type(exc).__name__
            time.sleep(1)
    raise TimeoutError(f"Engine readiness deadline exceeded; last observation: {last}")


def worker_ranks(log: str, tp: int) -> dict[int, int]:
    if tp == 1:
        matches = re.findall(r"EngineCore(?:_DP0)? pid=(\d+)\).*world_size=1 rank=0", log)
        if not matches or len(set(matches)) != 1:
            raise RuntimeError("Expected exactly one rank-zero EngineCore identity")
        return {0: int(matches[0])}
    pairs = {(int(rank), int(pid)) for rank, pid in re.findall(r"Worker_TP(\d+) pid=(\d+)", log)}
    ranks = dict(pairs)
    if len(pairs) != tp or set(ranks) != set(range(tp)) or len(set(ranks.values())) != tp:
        raise RuntimeError("Expected exactly one distinct live worker identity per TP rank")
    return ranks


def topology(run: Path, record: dict) -> dict:
    require_owner(record)
    tp = record["tensor_parallel_size"]
    gpus = record["gpus_before"]
    if len(gpus) != tp or len({gpu["uuid"] for gpu in gpus}) != tp:
        raise RuntimeError("Recorded GPU allocation does not match TP")
    selection = ",".join(str(gpu["index"]) for gpu in gpus)
    environment = Path(f"/proc/{record['identity']['pid']}/environ").read_bytes().split(b"\0")
    mapping = next(
        x.split(b"=", 1)[1].decode() for x in environment if x.startswith(b"CUDA_VISIBLE_DEVICES=")
    )
    if mapping != selection:
        raise RuntimeError("Live parent CUDA mapping differs from the allocation")
    observed = {
        "tensor_parallel_size": tp,
        "parent_cuda_visible_devices": mapping,
        "selected_gpus": gpus,
    }
    if record["engine"] == "vllm":
        ranks = worker_ranks((run / "engine.log").read_text(), tp)
        members = group_members(record["identity"]["pid"])
        if not all(pid in members for pid in ranks.values()):
            raise RuntimeError("Rank-labelled workers are not live in the owned process group")
        observed["owned_worker_rank_pids"] = ranks
        observed["proof"] = (
            "Live rank-labelled workers and parent CUDA mapping; no NVML PID mapping claimed"
        )
    else:
        response = httpx.get(f"http://127.0.0.1:{record['port']}/server_info", timeout=5)
        response.raise_for_status()
        data = response.json()
        observed["server_parallel_config"] = {
            key: data.get(key)
            for key in ("tp_size", "pp_size", "dp_size", "dtype", "model_path", "context_length")
        }
        if data.get("tp_size") != tp or data.get("pp_size") != 1 or data.get("dp_size") != 1:
            raise RuntimeError("Live SGLang parallel configuration differs from requested profile")
        observed["proof"] = "Live server parallel configuration and parent CUDA mapping"
    return observed


def postcheck(run: Path, record: dict, key: str, source: str) -> dict:
    require_owner(record)
    with httpx.Client(
        base_url=f"http://127.0.0.1:{record['port']}/plugins/jev-runtime",
        headers={"Authorization": "Bearer " + key},
        timeout=5,
    ) as client:
        deadline = time.monotonic() + 10
        while True:
            response = client.get("/admin/bundles")
            response.raise_for_status()
            listing = response.json()
            if not listing["leases"]:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Bundle leases remain after validation")
            time.sleep(0.05)
        response = client.get("/admin/profile")
        response.raise_for_status()
        profile = response.json()
    save(run / "profile.json", profile)
    if any(
        profile["admission"][k] != 0
        for k in ("requests", "expanded_tokens", "expanded_branches", "queued_requests")
    ):
        raise RuntimeError("Admission counters did not drain")
    if not profile["health"]["monitor_running"] or not all(
        v["ready"] and v["prepared"] for v in profile["health"]["bundles"].values()
    ):
        raise RuntimeError("Active bundles are not healthy")
    modules = ("jev_runtime", "jev_runtime.health", "jev_" + record["engine"])
    paths = {name: importlib.import_module(name).__file__ for name in modules}
    if not all(source in path for path in paths.values()):
        raise RuntimeError("Installed runtime/plugin does not match the declared source checkout")
    save(
        run / "environment.json",
        {
            "runtime_source_commit": source,
            "python": sys.version,
            "import_paths": paths,
            "versions": {
                name: metadata.version(name) for name in (record["engine"], "torch", "transformers")
            },
        },
    )
    return {
        "zero_leases": True,
        "zero_admission": True,
        "all_active_bundles_healthy": True,
        "active_routes": listing["routes"],
    }


def cleanup(run: Path, record: dict, timeout: float = 60) -> dict:
    result = {"process_record": record, "checked_at": time.time()}
    try:
        dsw_service.stop(SimpleNamespace(run_dir=run))
    except BaseException as exc:
        result["stop_failure"] = {"type": type(exc).__name__, "message": str(exc)}
    deadline = time.monotonic() + timeout
    while True:
        remaining = group_members(record["identity"]["pid"])
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(1)
    result["remaining_process_group_members"] = remaining
    result["gpus_after"] = [
        {"index": gpu["index"], **dsw_service.get_gpu(gpu["index"])}
        for gpu in record["gpus_before"]
    ]
    result["gpu_uuids_match"] = all(
        before["uuid"] == after["uuid"]
        for before, after in zip(record["gpus_before"], result["gpus_after"], strict=True)
    )
    result["passed"] = not remaining and result["gpu_uuids_match"] and "stop_failure" not in result
    save(run / "cleanup.json", result)
    return result


def execute(script: Path, args: list[str], log: Path, timeout: float) -> int:
    with log.open("x") as file:
        return subprocess.run(
            [sys.executable, str(script), *map(str, args)],
            stdout=file,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        ).returncode


def validate(args) -> dict:
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    report = {
        "harness_source_commit": args.source_commit,
        "runtime_source_commit": args.runtime_source_commit,
        "started_at": time.time(),
        "qualification": (
            "colocated functional validation; single-position numerical development check"
        ),
        "stages": {},
    }
    record = None
    stage = "launch"
    try:
        launch = [
            "launch",
            "--run-dir",
            run,
            "--engine",
            args.engine,
            "--model-path",
            args.model_path,
            "--gpus",
            args.gpus,
            "--port",
            str(args.port),
            "--memory-fraction",
            str(args.memory_fraction),
            "--reserve-mib",
            str(args.reserve_mib),
            "--readout-dtype",
            args.readout_dtype,
        ]
        code = execute(ROOT / "deployment/dsw_service.py", launch, run / "launch.log", 60)
        if (run / "process.json").exists():
            record = json.loads((run / "process.json").read_text())
        if code or record is None:
            raise RuntimeError("Launch failed; inspect launch.log and engine.log")
        keys = json.loads((run / "keys.json").read_text())
        stage = "readiness"
        report["stages"][stage] = wait_ready(record, keys["api"], args.startup_timeout)
        stage = "topology"
        save(run / "topology.json", topology(run, record))
        report["stages"][stage] = {"passed": True}
        for stage, script, output, extra in (
            ("contract", "live_contract.py", "contract.json", ["--switches", str(args.switches)]),
            ("precision_binding", "live_precision_binding.py", "precision-binding.json", []),
        ):
            require_owner(record)
            options = [
                "--run-dir",
                run,
                "--output",
                run / output,
                "--source-commit",
                args.source_commit,
                "--runtime-source-commit",
                args.runtime_source_commit,
                *extra,
            ]
            code = execute(
                Path(__file__).with_name(script),
                options,
                run / (stage + ".log"),
                args.check_timeout,
            )
            result = json.loads((run / output).read_text())
            report["stages"][stage] = {"returncode": code, "passed": result["passed"]}
            if code or not result["passed"]:
                raise RuntimeError(f"{stage} failed; inspect retained report")
        stage = "postcheck"
        save(
            run / "postcheck.json",
            postcheck(run, record, keys["admin"], args.runtime_source_commit),
        )
        report["stages"][stage] = {"passed": True}
        report["functional_passed"] = True
        stage = "cpu_reference"
        code = execute(
            Path(__file__).with_name("reference_logits.py"),
            [
                "--model-path",
                args.model_path,
                "--contract-report",
                run / "contract.json",
                "--output",
                run / "reference-cpu.json",
                "--device",
                "cpu",
                "--atol",
                str(args.reference_atol),
                "--readout-dtype",
                args.readout_dtype,
            ],
            run / "reference-cpu.log",
            args.reference_timeout,
        )
        result = json.loads((run / "reference-cpu.json").read_text())
        if code != (0 if result["passed"] else 1):
            raise RuntimeError("CPU reference exit status disagrees with its report")
        report["stages"][stage] = {
            "returncode": code,
            "passed": result["passed"],
            "max_absolute_error": max(result["absolute_errors"]),
        }
        report["numerical_position_passed"] = result["passed"]
    except BaseException as exc:
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:3000]}
    finally:
        # Recover the launch record even if the launcher command was interrupted.
        if record is None and (run / "process.json").exists():
            record = json.loads((run / "process.json").read_text())
        if record is not None:
            try:
                report["cleanup_passed"] = cleanup(run, record)["passed"]
            except BaseException as exc:
                report["cleanup_passed"] = False
                report["cleanup_failure"] = {"type": type(exc).__name__, "message": str(exc)[:3000]}
        else:
            # No process identity means ownership/cleanup cannot be certified.
            report["cleanup_passed"] = False
        report["passed"] = bool(
            report.get("functional_passed")
            and report.get("numerical_position_passed")
            and report["cleanup_passed"]
            and "failure" not in report
        )
        report["finished_at"] = time.time()
        save(run / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--reserve-mib", type=int, default=3072)
    parser.add_argument("--readout-dtype", choices=["model", "float32"], default="model")
    parser.add_argument("--switches", type=int, default=1000)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--check-timeout", type=float, default=600)
    parser.add_argument("--reference-timeout", type=float, default=900)
    parser.add_argument("--reference-atol", type=float, default=0.15)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    args = parser.parse_args()
    if not all(
        re.fullmatch(r"[0-9a-f]{40}", value)
        for value in (args.source_commit, args.runtime_source_commit)
    ):
        parser.error("Full immutable source commits are required")
    if (
        min(args.startup_timeout, args.check_timeout, args.reference_timeout, args.switches) <= 0
        or args.reference_atol < 0
    ):
        parser.error("Timeouts/switches must be positive and tolerance nonnegative")
    report = validate(args)
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
