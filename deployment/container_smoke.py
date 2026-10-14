"""Build the gateway image and test non-root offline installation and persistence.

Requires a dedicated Linux Docker builder. No model serving or GPU is certified.
Only the uniquely named test volume and containers created here are removed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

from deployment import image_context, release


def run(repo: Path, ref: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source = release.build(repo, ref, output / "wheels")
    locked = release.resolve(
        output / "wheels", source["manifest_sha256"], output / "locked", include_plugins=True
    )
    subprocess.run(["docker", "pull", "python:3.12-slim-bookworm"], check=True)
    base = json.loads(
        subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "python:3.12-slim-bookworm",
                "--format",
                "{{json .RepoDigests}}",
            ],
            text=True,
        )
    )[0]
    context = image_context.prepare(
        output / "locked", locked["manifest_sha256"], base, output / "context"
    )
    image_context.verify(output / "context", context["manifest_sha256"])
    tag = "jev-gateway-test:" + uuid.uuid4().hex
    subprocess.run(
        [
            "docker",
            "build",
            "--network=none",
            "--tag",
            tag,
            "--iidfile",
            str(output / "image-id.txt"),
            str(output / "context"),
        ],
        check=True,
    )
    volume = "jev-test-" + uuid.uuid4().hex
    created = False
    try:
        subprocess.run(["docker", "volume", "create", volume], check=True)
        created = True
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=volume,src={volume},dst=/data/jev",
            "--entrypoint",
            "/opt/jev/venv/bin/python",
            tag,
            "-I",
            "-c",
        ]
        subprocess.run(
            command
            + [
                "import os,sqlite3; from jev_runtime.backends.discovery import backend_inventory; "
                "from transformers import PreTrainedTokenizerFast; "
                "from tokenizers import Tokenizer,models; "
                "assert os.getuid()==10001; "
                "assert {x['name'] for x in backend_inventory()}=={'vllm','sglang','tokenspeed'}; "
                "t=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'x':0},'x')),"
                "chat_template='{{ messages[0].content }}'); "
                "assert t.apply_chat_template([{'role':'user','content':'x'}],"
                "return_dict=False)==[0]; "
                "db=sqlite3.connect('/data/jev/probe.db'); "
                "db.execute('CREATE TABLE probe(value INTEGER)'); "
                "db.execute('INSERT INTO probe VALUES (42)'); db.commit(); db.close()"
            ],
            check=True,
        )
        subprocess.run(
            command
            + [
                "import sqlite3; db=sqlite3.connect('/data/jev/probe.db'); "
                "assert db.execute('SELECT value FROM probe').fetchall()==[(42,)]; db.close()"
            ],
            check=True,
        )
        report = {
            "passed": True,
            "scope": "CPU gateway container packaging only; no GPU serving",
            "source_commit": source["source_commit"],
            "base_image": base,
            "local_image_config_digest": (output / "image-id.txt").read_text().strip(),
            "context_manifest_sha256": context["manifest_sha256"],
            "wheel_manifest_sha256": source["manifest_sha256"],
            "non_root": True,
            "network_disabled": True,
            "read_only_root": True,
            "chat_template": True,
            "persistent_volume_across_containers": True,
        }
        release.save(output / "container-smoke.json", report)
        return report
    finally:
        if created:
            subprocess.run(["docker", "volume", "rm", volume], check=True)
        subprocess.run(["docker", "image", "rm", tag], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.repo.resolve(), args.ref, args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
