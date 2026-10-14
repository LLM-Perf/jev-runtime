"""Validate a hash-pinned qualification ZIP against the actual release wheels."""

from __future__ import annotations

import argparse
import json
import stat
import tempfile
import zipfile
from pathlib import Path

from jev_runtime.production import ProductionProfile, qualify, read_json, sha256


def unpack(archive: Path, expected: str, destination: Path):
    if sha256(archive) != expected:
        raise ValueError("Qualification archive hash mismatch")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 10000 or sum(e.file_size for e in entries) > 256 * 1024 * 1024:
            raise ValueError(
                "Qualification archive is too large; store large raw traces externally"
            )
        names = [e.filename for e in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        for entry in entries:
            path = Path(entry.filename)
            mode = entry.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or stat.S_ISLNK(mode)
                or not path.parts
            ):
                raise ValueError("Unsafe archive path")
        for entry in entries:
            path = destination / entry.filename
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bundle.read(entry))


def validate(archive: Path, expected: str, dist: Path, commit: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="jev-qualification-") as temporary:
        root = Path(temporary)
        unpack(archive, expected, root)
        profiles = sorted((root / "profiles").glob("*.json"))
        if not profiles:
            raise ValueError("Qualification requires production profiles")
        results, engines, ids = [], set(), set()
        for path in profiles:
            profile = ProductionProfile.model_validate(read_json(path))
            if profile.profile_id in ids:
                raise ValueError("Duplicate production profile")
            ids.add(profile.profile_id)
            engines.add(profile.engine)
            results.append(qualify(path, root / "evidence" / profile.profile_id, commit, dist))
        required = {"vllm", "sglang", "tokenspeed"}
        return {
            "passed": engines == required and all(r["passed"] for r in results),
            "source_commit": commit,
            "qualification_sha256": expected,
            "missing_engines": sorted(required - engines),
            "profiles": results,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        result = validate(args.bundle, args.bundle_sha256, args.dist, args.source_commit)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        result = {"passed": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
