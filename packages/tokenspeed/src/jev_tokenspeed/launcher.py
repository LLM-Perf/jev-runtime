import argparse
import json
import os
from pathlib import Path

import uvicorn

from jev_runtime.config import load_settings
from jev_tokenspeed.plugin import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Launch a pinned TokenSpeed engine with Jev routes"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    args = parser.parse_args()
    os.environ["JEV_CONFIG"] = str(args.config.resolve())
    settings = load_settings(args.config)
    app = create_app(settings, json.loads(args.engine_config.read_text()))
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
