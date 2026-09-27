"""Prepare and verify a digest-bound, offline gateway image build context.

This command does not contact Docker, pull a base image, build or publish an image.
It accepts an already resolved Linux wheelhouse and a trusted immutable base image.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from deployment import release

HERE = Path(__file__).resolve().parent
ARCHITECTURES = {"x86_64": "linux/amd64", "aarch64": "linux/arm64"}
# An OCI reference only; URLs, whitespace, credentials and Dockerfile syntax are excluded.
BASE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[a-f0-9]{64}")


def image_platform(manifest: dict) -> str:
    target = manifest["target"]
    if (
        target["platform"] != "linux"
        or target["implementation"] != "CPython"
        or target["machine"] not in ARCHITECTURES
        or not re.fullmatch(r"3\.(?:1[1-9]|[2-9][0-9])", target["python"])
    ):
        raise ValueError("Gateway image requires a Linux CPython 3.11+ amd64/arm64 lock")
    return ARCHITECTURES[target["machine"]]


def prepare(bundle: Path, digest: str, base_image: str, output: Path) -> dict:
    manifest = release.verify(bundle, digest, "jev-gateway-lock-v1")
    platform = image_platform(manifest)
    if not BASE.fullmatch(base_image) or "://" in base_image:
        raise ValueError("Supply a trusted base image reference pinned by @sha256:digest")
    commit = manifest["source_commit"]
    if not re.fullmatch(r"[a-f0-9]{40,64}", commit):
        raise ValueError("Image source must be an immutable Git commit")
    output = output.resolve()
    if output == bundle.resolve() or output.is_relative_to(bundle.resolve()):
        raise ValueError("Image context must be outside the source wheelhouse")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(bundle, output / "locked")
    release.verify(output / "locked", digest, "jev-gateway-lock-v1")
    for name in ("release.py", "container_gateway.py"):
        shutil.copyfile(HERE / name, output / name)
    dockerfile = (HERE / "Dockerfile.gateway").read_text()
    for name, value in {
        "__JEV_BASE_IMAGE__": base_image,
        "__JEV_LOCK_SHA256__": digest,
        "__JEV_SOURCE_COMMIT__": commit,
    }.items():
        dockerfile = dockerfile.replace(name, value)
    (output / "Dockerfile").write_text(dockerfile)
    (output / ".dockerignore").write_text(
        "*\n!Dockerfile\n!release.py\n!container_gateway.py\n!image.json\n!locked/\n!locked/**\n"
    )
    image = {
        "schema_version": 1,
        "kind": "jev-gateway-image-input-v1",
        "source_commit": commit,
        "gateway_manifest_sha256": digest,
        "base_image": base_image,
        "platform": platform,
        "target": manifest["target"],
        "preparer_sha256": release.sha(Path(__file__)),
        "helpers": {n: release.sha(output / n) for n in ("release.py", "container_gateway.py")},
        "template_sha256": release.sha(HERE / "Dockerfile.gateway"),
        "runtime_user": "10001:10001",
        "qualification": "Build inputs only; no container build or runtime validation implied",
    }
    release.save(output / "image.json", image)
    context = {
        "schema_version": 1,
        "kind": "jev-gateway-context-v1",
        "image": image,
        "files": release.inventory(output),
    }
    release.save(output / "manifest.json", context)
    return {
        "path": str(output),
        "manifest_sha256": release.sha(output / "manifest.json"),
        "source_commit": commit,
        "platform": platform,
        "image_built": False,
    }


def verify(context: Path, digest: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("An explicit context SHA256 is required")
    path = context / "manifest.json"
    if path.is_symlink() or release.sha(path) != digest:
        raise ValueError("Image context manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "jev-gateway-context-v1":
        raise ValueError("Unsupported image context")
    if release.inventory(context) != manifest["files"]:
        raise ValueError("Image context inventory, file hash or size mismatch")
    image = json.loads((context / "image.json").read_text())
    if image != manifest["image"] or not BASE.fullmatch(image["base_image"]):
        raise ValueError("Image descriptor mismatch")
    locked = release.verify(
        context / "locked", image["gateway_manifest_sha256"], "jev-gateway-lock-v1"
    )
    if image["source_commit"] != locked["source_commit"] or image["platform"] != image_platform(
        locked
    ):
        raise ValueError("Image descriptor differs from locked source or platform")
    if image["target"] != locked["target"]:
        raise ValueError("Image target differs from locked target")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--base-image", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.bundle, args.manifest_sha256, args.base_image, args.output)
    else:
        manifest = verify(args.context, args.manifest_sha256)
        result = {"verified": True, "image": manifest["image"], "image_built": False}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
