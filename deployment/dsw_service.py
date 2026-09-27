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
        if stat[0] == "Z":
            return None
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
        # Match uvicorn's bind semantics. A previous gateway's accepted
        # connections can leave TIME_WAIT after SIGKILL without a live listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    gpu = None if args.gateway else get_gpu(args.gpu)
    # vLLM 0.30 uses total device memory. SGLang 0.5.19's configurator uses
    # pre-load available memory, which matters on shared devices.
    denominator = None
    if gpu is not None:
        denominator = "total_mib" if args.engine == "vllm" else "free_mib"
        budget = float(gpu[denominator]) * args.memory_fraction
        if budget + args.reserve_mib > float(gpu["free_mib"]):
            raise SystemExit("Insufficient free GPU memory for the explicit budget and reserve")
    model = args.model_path.resolve()
    source = json.loads((model / "jev-source.json").read_text())
    config = {
        "backend": args.engine,
        "engine_url": args.engine_url or f"http://127.0.0.1:{args.port}",
        "model_id": source["model_id"],
        "model_revision": source["revision"],
        "tokenizer": str(model),
        "dtype": "bfloat16",
        "registry_path": str(root / "registry.db"),
        "bootstrap_alias": "decision-model",
        "bootstrap_bundle_id": "default",
        "host": "127.0.0.1",
        "port": args.port,
        "workers": args.api_workers,
    }
    if args.admission_config:
        from jev_runtime.config import AdmissionSettings

        config["admission"] = AdmissionSettings.model_validate_json(
            args.admission_config.read_text()
        ).model_dump()
    if args.health_config:
        from jev_runtime.health import HealthSettings

        config["health"] = HealthSettings.model_validate_json(
            args.health_config.read_text()
        ).model_dump()
    config["tenant_key_envs"] = {
        tenant: f"JEV_TEST_TENANT_{index}" for index, tenant in enumerate(args.tenant)
    }
    if args.adapters_root:
        config["adapters"] = {
            "enabled": True,
            "store_path": str(root / "adapters"),
            "allowed_roots": [str(args.adapters_root.resolve(strict=True))],
        }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    keys = root / "keys.json"
    if not keys.exists():
        descriptor = os.open(keys, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump(
                {
                    "api": secrets.token_urlsafe(32),
                    "admin": secrets.token_urlsafe(32),
                    "tenants": {tenant: secrets.token_urlsafe(32) for tenant in args.tenant},
                },
                file,
            )
    credentials = json.loads(keys.read_text())
    if set(credentials.get("tenants", {})) != set(args.tenant):
        raise SystemExit(
            "Existing credentials have a different tenant set; use a new run directory"
        )
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "JEV_API_KEY": credentials["api"],
            "JEV_ADMIN_KEY": credentials["admin"],
            "JEV_CONFIG": str(root / "config.json"),
        }
    )
    for tenant, env_name in config["tenant_key_envs"].items():
        env[env_name] = credentials["tenants"][tenant]
    if args.engine_run_dir:
        engine_record = json.loads((args.engine_run_dir / "process.json").read_text())
        expected_url = f"http://127.0.0.1:{engine_record['port']}"
        if args.engine_url != expected_url or engine_record["engine"] != args.engine:
            raise SystemExit("Engine credential source does not match the configured target")
        env["JEV_ENGINE_API_KEY"] = json.loads((args.engine_run_dir / "keys.json").read_text())[
            "api"
        ]
    if args.gateway:
        command = [
            str(Path(sys.executable).with_name("jevctl")),
            "serve",
            str(root / "config.json"),
        ]
    elif args.engine == "vllm":
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
        if args.api_workers > 1:
            command.extend(["--api-server-count", str(args.api_workers)])
        if args.adapters_root:
            command.extend(
                [
                    "--enable-lora",
                    "--max-loras",
                    "2",
                    "--max-cpu-loras",
                    "4",
                    "--max-lora-rank",
                    "64",
                    "--worker-extension-cls",
                    "jev_vllm.worker.LoRAWorkerExtension",
                ]
            )
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
        if args.api_workers > 1:
            command.extend(["--tokenizer-worker-num", str(args.api_workers)])
        if args.adapters_root:
            command.extend(
                [
                    "--enable-lora",
                    "--max-lora-rank",
                    "64",
                    "--max-loaded-loras",
                    "4",
                    "--max-loras-per-batch",
                    "4",
                    "--lora-target-modules",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
    with (root / "engine.log").open("ab") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
    identity = process_identity(child.pid)
    if identity is None:
        raise SystemExit("Engine process exited before identity capture; inspect engine.log")
    manifest = {
        "identity": identity,
        "engine": args.engine,
        "mode": "gateway" if args.gateway else "native-plugin",
        "command": command,
        "model": source,
        "gpu_before": gpu,
        "created": time.time(),
        "port": args.port,
        "qualification": "colocated-functional-test",
        "memory_budget_denominator": denominator,
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
    parser.add_argument("--gateway", action="store_true")
    parser.add_argument("--engine-url")
    parser.add_argument("--engine-run-dir", type=Path)
    parser.add_argument("--api-workers", type=int, default=1)
    parser.add_argument("--adapters-root", type=Path)
    parser.add_argument("--admission-config", type=Path)
    parser.add_argument("--health-config", type=Path)
    parser.add_argument("--tenant", action="append", default=[])
    args = parser.parse_args()
    if args.action == "launch":
        if not args.engine or args.model_path is None or not 0 < args.memory_fraction < 1:
            parser.error("launch requires engine, model path and a valid memory fraction")
        if args.gateway and not args.engine_url:
            parser.error("gateway requires an explicit existing engine URL")
        if args.engine_run_dir and not args.gateway:
            parser.error("engine-run-dir only supplies credentials for a gateway")
        if not 1 <= args.api_workers <= 128:
            parser.error("api-workers must be between 1 and 128")
        if args.adapters_root and (args.gateway or args.api_workers != 1):
            parser.error("managed adapters require the native single-worker profile")
        if len(set(args.tenant)) != len(args.tenant) or any(
            not name or name == "default" for name in args.tenant
        ):
            parser.error("tenant names must be distinct, non-empty and non-default")
    {"launch": launch, "status": status, "stop": stop}[args.action](args)


if __name__ == "__main__":
    main()
