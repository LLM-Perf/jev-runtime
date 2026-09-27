"""Gateway image PID-1 entrypoint and one-worker readiness probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from jev_runtime.config import load_settings


def configuration(config: Path, descriptor: Path):
    image = json.loads(descriptor.read_text())
    settings = load_settings(config)
    if settings.release_id != image["source_commit"]:
        raise ValueError("Set release_id to the image's immutable source commit")
    if not Path(settings.registry_path).is_absolute():
        raise ValueError(
            "Container registry_path must be absolute and on a local persistent volume"
        )
    if settings.tokenizer is None or not Path(settings.tokenizer).is_absolute():
        raise ValueError("Mount an explicit absolute tokenizer path; image startup is offline")
    if not Path(settings.tokenizer).is_dir():
        raise ValueError("Configured tokenizer mount is missing")
    if settings.host not in {"0.0.0.0", "127.0.0.1", "::", "::1"}:
        raise ValueError("Use a wildcard or loopback listen address in the gateway image")
    return settings


def serve(config: Path, descriptor: Path):
    settings = configuration(config, descriptor)
    grace = int(os.environ.get("JEV_GRACEFUL_TIMEOUT_SECONDS", "180"))
    if not 1 <= grace <= 3600:
        raise ValueError("JEV_GRACEFUL_TIMEOUT_SECONDS must be between 1 and 3600")
    os.environ["JEV_CONFIG"] = str(config.resolve())
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-m",
            "uvicorn",
            "jev_runtime.api:create_app_from_env",
            "--factory",
            "--host",
            settings.host,
            "--port",
            str(settings.port),
            "--workers",
            str(settings.workers),
            "--timeout-graceful-shutdown",
            str(grace),
        ],
    )


def healthcheck(config: Path, descriptor: Path):
    settings = configuration(config, descriptor)
    key = os.environ.get(settings.api_key_env)
    if not key:
        raise ValueError("Gateway data-plane credential is missing")
    host = "[::1]" if ":" in settings.host else "127.0.0.1"
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.get(
            f"http://{host}:{settings.port}/ready", headers={"Authorization": "Bearer " + key}
        )
    response.raise_for_status()
    data = response.json()
    if (
        data.get("ready") is not True
        or data.get("engine") != settings.backend
        or not data.get("prepared_bundles")
        or not data.get("worker_id")
    ):
        raise ValueError("Worker does not report active prepared bundles and current readiness")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "healthcheck"))
    parser.add_argument(
        "--config", type=Path, default=Path(os.environ.get("JEV_CONFIG", "/config/gateway.yaml"))
    )
    parser.add_argument("--descriptor", type=Path, default=Path("/opt/jev/image.json"))
    args = parser.parse_args()
    try:
        (serve if args.command == "serve" else healthcheck)(args.config, args.descriptor)
    except Exception as exc:
        # Container engines persist probe output; omit bodies, paths and credentials.
        print(f"Gateway {args.command} failed ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
