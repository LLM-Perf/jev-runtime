#!/usr/bin/env python3
"""Launch/status/stop an isolated test engine without modifying existing services.

Run with the intended engine environment's Python. Process identity includes the
boot ID and Linux start ticks, so a reused PID can never be stopped by this tool.
This is a single-GPU functional-test launcher, not a performance certification.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def process_identity(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": stat[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except FileNotFoundError:
        return None


def get_gpu(index: int) -> dict:
    value = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,name,memory.total,memory.free,driver_version,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    return dict(
        zip(("uuid", "name", "total_mib", "free_mib", "driver", "utilization"), value, strict=True)
    )


def launch(args):
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = root / "process.json"
    if record.exists():
        previous = json.loads(record.read_text())
        if process_identity(previous["identity"]["pid"]) == previous["identity"]:
            raise SystemExit("The recorded engine is still alive; refusing a duplicate launch")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    gpu = get_gpu(args.gpu)
    budget = float(gpu["total_mib"]) * args.memory_fraction
    if budget + args.reserve_mib > float(gpu["free_mib"]):
        raise SystemExit("Insufficient free GPU memory for the explicit budget and reserve")
    model = args.model_path.resolve()
    source = json.loads((model / "jev-source.json").read_text())
    config = {
        "backend": args.engine,
        "engine_url": f"http://127.0.0.1:{args.port}",
        "model_id": source["model_id"],
        "model_revision": source["revision"],
        "tokenizer": str(model),
        "dtype": "bfloat16",
        "registry_path": str(root / "registry.db"),
        "bootstrap_alias": "decision-model",
        "bootstrap_bundle_id": "default",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    keys = root / "keys.json"
    if not keys.exists():
        descriptor = os.open(keys, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump({"api": secrets.token_urlsafe(32), "admin": secrets.token_urlsafe(32)}, file)
    credentials = json.loads(keys.read_text())
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "JEV_API_KEY": credentials["api"],
            "JEV_ADMIN_KEY": credentials["admin"],
            "JEV_CONFIG": str(root / "config.json"),
        }
    )
    if args.engine == "vllm":
        env["VLLM_PLUGINS"] = "jev_runtime_api"
        command = [
            str(Path(sys.executable).with_name("vllm")),
            "serve",
            str(model),
            "--served-model-name",
            source["model_id"],
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "512",
            "--max-logprobs",
            "128",
            "--gpu-memory-utilization",
            str(args.memory_fraction),
            "--enforce-eager",
        ]
    else:
        env["SGLANG_PLUGINS"] = "jev_runtime"
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model),
            "--served-model-name",
            source["model_id"],
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--dtype",
            "bfloat16",
            "--context-length",
            "2048",
            "--max-running-requests",
            "4",
            "--chunked-prefill-size",
            "512",
            "--max-total-tokens",
            "2048",
            "--mem-fraction-static",
            str(args.memory_fraction),
            "--disable-cuda-graph",
            "--attention-backend",
            "triton",
        ]
    with (root / "engine.log").open("ab") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
    identity = process_identity(child.pid)
    if identity is None:
        raise SystemExit("Engine process exited before identity capture; inspect engine.log")
    manifest = {
        "identity": identity,
        "engine": args.engine,
        "command": command,
        "model": source,
        "gpu_before": gpu,
        "created": time.time(),
        "port": args.port,
        "qualification": "colocated-functional-test",
    }
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"started": identity, "port": args.port, "run_dir": str(root)}))


def status(args):
    record = json.loads((args.run_dir / "process.json").read_text())
    actual = process_identity(record["identity"]["pid"])
    print(
        json.dumps(
            {
                "recorded_identity": record["identity"],
                "actual_identity": actual,
                "alive": actual == record["identity"],
            }
        )
    )


def stop(args):
    record = json.loads((args.run_dir / "process.json").read_text())
    identity = record["identity"]
    actual = process_identity(identity["pid"])
    if actual is None:
        print(json.dumps({"stopped": True, "already_exited": True}))
        return
    if actual != identity:
        raise SystemExit("PID identity changed; refusing to signal an unrelated process")
    os.killpg(identity["pid"], signal.SIGTERM)
    print(json.dumps({"signal_sent": "SIGTERM", "identity": identity, "exit_confirmed": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "status", "stop"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=["sglang", "vllm"])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--port", type=int, default=18795)
    parser.add_argument("--memory-fraction", type=float, default=0.07)
    parser.add_argument("--reserve-mib", type=int, default=3072)
    args = parser.parse_args()
    if args.action == "launch":
        if not args.engine or args.model_path is None or not 0 < args.memory_fraction < 1:
            parser.error("launch requires engine, model path and a valid memory fraction")
    {"launch": launch, "status": status, "stop": stop}[args.action](args)


if __name__ == "__main__":
    main()
