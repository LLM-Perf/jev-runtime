import argparse
import json
import os
from pathlib import Path

import uvicorn

from jev_runtime.config import load_settings
from jev_runtime.errors import JevError
from jev_tokenspeed.plugin import create_app, verify_source


def main():
    parser = argparse.ArgumentParser(
        description="Launch a pinned TokenSpeed engine with Jev routes"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--engine-config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Preflight without loading a model")
    mode.add_argument(
        "--check-source", type=Path, help="CPU-only source check: path to tokenspeed/"
    )
    args = parser.parse_args()
    if args.check_source:
        try:
            print(
                json.dumps(
                    {
                        "upstream_revision": verify_source(args.check_source),
                        "scope": "source only",
                        "gpu_validated": False,
                    }
                )
            )
        except JevError as exc:
            parser.error(str(exc))
        return
    if args.config is None or args.engine_config is None:
        parser.error("--config and --engine-config are required for launch or --check")
    os.environ["JEV_CONFIG"] = str(args.config.resolve())
    settings = load_settings(args.config)
    options = json.loads(args.engine_config.read_text())
    if args.check:
        from jev_tokenspeed.preflight import check

        try:
            # Validate the same plugin settings as an actual launch.
            create_app(settings, options)
            print(json.dumps(check(settings, options), indent=2))
        except (JevError, ValueError, ImportError) as exc:
            parser.error(str(exc))
        return
    app = create_app(settings, options)
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
