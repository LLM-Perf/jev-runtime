"""Build immutable wheels and provision a fresh gateway environment from a hash lock.

Run with Python 3.11+. Build needs build/hatchling; resolve needs pip and network.
Install uses only the verified wheelhouse. Existing environments are never edited.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import zipfile
from pathlib import Path

PACKAGES = {
    "jev-runtime-core": (".", "src/jev_runtime"),
    "jev-sglang": ("packages/sglang", "packages/sglang/src/jev_sglang"),
    "jev-vllm": ("packages/vllm", "packages/vllm/src/jev_vllm"),
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    with path.open("x") as file:
        file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def target() -> dict:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "soabi": sysconfig.get_config_var("SOABI"),
    }


def wheel_info(path: Path) -> dict:
    with zipfile.ZipFile(path) as wheel:
        entries = wheel.namelist()
        if len(entries) != len(set(entries)):
            raise ValueError("Duplicate wheel entries")
        for name in entries:
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise ValueError("Unsafe wheel entry")
        candidates = [
            n for n in entries if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1:
            raise ValueError("Wheel must have one metadata record")
        info = email.parser.BytesParser().parsebytes(wheel.read(candidates[0]))
        name, version = canonical(info["Name"]), info["Version"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or not re.fullmatch(
            r"[a-zA-Z0-9.!+_-]+", version
        ):
            raise ValueError("Unsafe distribution identity")
        return {"name": name, "version": version, "requires": info.get_all("Requires-Dist", [])}


def gateway_extras(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as wheel:
        entry = next(
            n
            for n in wheel.namelist()
            if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")
        )
        info = email.parser.BytesParser().parsebytes(wheel.read(entry))
    supported = set(info.get_all("Provides-Extra", []))
    if "tokenizers" not in supported:
        raise ValueError("Core wheel does not define the gateway tokenizer extra")
    return [name for name in ("tokenizers", "tokenizer-conversion") if name in supported]


def inventory(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Release payload must not contain symlinks")
        if path.is_file() and path != directory / "manifest.json":
            result[path.relative_to(directory).as_posix()] = {
                "sha256": sha(path),
                "size_bytes": path.stat().st_size,
            }
    return result


def verify(directory: Path, expected: str, kind: str | None = None) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("An explicit lowercase SHA256 manifest digest is required")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or sha(manifest_path) != expected:
        raise ValueError("Release manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or (kind and manifest.get("kind") != kind):
        raise ValueError("Unsupported release manifest")
    if inventory(directory) != manifest["files"]:
        raise ValueError("Release file hash, size or inventory mismatch")
    for name, distribution in manifest["distributions"].items():
        wheel_path = distribution["wheel"]
        if wheel_path not in manifest["files"] or not wheel_path.endswith(".whl"):
            raise ValueError("Distribution wheel is outside the verified payload")
        if canonical(name) != name or wheel_info(directory / wheel_path) != {
            key: distribution[key] for key in ("name", "version", "requires")
        }:
            raise ValueError("Distribution metadata differs from verified wheel")
    return manifest


def run(command: list[str], log: Path, *, cwd: Path | None = None) -> None:
    # Do not inherit pip indexes, PYTHONPATH or user site configuration.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "PYTHON"))}
    env["PIP_CONFIG_FILE"] = os.devnull
    with log.open("ab") as file:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=file, stderr=file, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}; inspect {log}")


def build(repo: Path, ref: str, output: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=repo, text=True
    ).strip()
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "wheels"
    wheels.mkdir()
    paths = ["pyproject.toml", "README.md", "src", "packages/sglang", "packages/vllm"]
    with tempfile.TemporaryDirectory(prefix="jev-build-") as temporary:
        work = Path(temporary)
        archive = work / "source.tar"
        with archive.open("wb") as file:
            subprocess.run(["git", "archive", commit, *paths], cwd=repo, stdout=file, check=True)
        source = work / "source"
        source.mkdir()
        with tarfile.open(archive) as tar:
            # Git-owned source only; never follow archived symlinks or special files.
            for member in tar.getmembers():
                if (
                    (not member.isfile() and not member.isdir())
                    or member.name.startswith("/")
                    or ".." in Path(member.name).parts
                ):
                    raise ValueError("Unsafe source archive entry")
                if member.isfile():
                    destination = source / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(tar.extractfile(member).read())
        for directory, _ in PACKAGES.values():
            run(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(wheels),
                ],
                output / "build.log",
                cwd=source / directory,
            )
        distributions = {}
        for wheel_path in sorted(wheels.glob("*.whl")):
            info = wheel_info(wheel_path)
            name = info["name"]
            if name not in PACKAGES or name in distributions:
                raise ValueError("Unexpected or duplicate project wheel")
            prefix = PACKAGES[name][1]
            expected = {
                p.relative_to(source / prefix).as_posix(): p.read_bytes()
                for p in (source / prefix).rglob("*.py")
            }
            with zipfile.ZipFile(wheel_path) as wheel:
                root = Path(prefix).name + "/"
                actual = {
                    n[len(root) :]: wheel.read(n)
                    for n in wheel.namelist()
                    if n.startswith(root) and n.endswith(".py")
                }
            if actual != expected:
                raise ValueError("Wheel Python files differ from committed source")
            distributions[name] = {
                **info,
                "wheel": "wheels/" + wheel_path.name,
                "verified_python_files": len(actual),
            }
        if (
            set(distributions) != set(PACKAGES)
            or len({d["version"] for d in distributions.values()}) != 1
        ):
            raise ValueError("Release requires three matching package versions")
    manifest = {
        "schema_version": 1,
        "kind": "jev-wheels-v1",
        "source_commit": commit,
        "builder_sha256": sha(Path(__file__)),
        "builder_target": target(),
        "build_tools": {name: metadata.version(name) for name in ("build", "hatchling")},
        "distributions": distributions,
        "files": inventory(output),
    }
    save(output / "manifest.json", manifest)
    return {
        "path": str(output),
        "manifest_sha256": sha(output / "manifest.json"),
        "source_commit": commit,
    }


def resolve(bundle: Path, digest: str, output: Path, constraints: Path | None = None) -> dict:
    source = verify(bundle, digest, "jev-wheels-v1")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "wheels"
    wheels.mkdir()
    core = bundle.absolute() / source["distributions"]["jev-runtime-core"]["wheel"]
    extras = gateway_extras(core)
    pins = [str(core) + "[" + ",".join(extras) + "]", "pip==" + metadata.version("pip")]
    try:
        pins.append("setuptools==" + metadata.version("setuptools"))
    except metadata.PackageNotFoundError:
        pass
    command = [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "download",
        "--only-binary=:all:",
        "--index-url",
        "https://pypi.org/simple",
        "--dest",
        str(wheels),
        *pins,
    ]
    if constraints is not None:
        # Snapshot before resolution; do not place credential-bearing URLs in this file.
        text = constraints.read_text()
        if any(
            line.strip() and not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9.!+_-]+", line.strip())
            for line in text.splitlines()
        ):
            raise ValueError("Constraints must contain only exact name==version pins")
        (output / "constraints.txt").write_text(text)
        command.extend(["--constraint", str(output / "constraints.txt")])
    run(command, output / "resolve.log")
    verify(bundle, digest, "jev-wheels-v1")
    distributions = {}
    for path in sorted(wheels.iterdir()):
        if path.suffix != ".whl":
            raise ValueError("Only binary wheels are allowed")
        info = wheel_info(path)
        if info["name"] in distributions:
            raise ValueError("One wheel per locked distribution is required")
        distributions[info["name"]] = {**info, "wheel": "wheels/" + path.name, "sha256": sha(path)}
    copied_core = output / distributions["jev-runtime-core"]["wheel"]
    if sha(copied_core) != sha(core):
        raise ValueError("Resolved core wheel changed")
    lock = "".join(
        f"{name}=={d['version']} --hash=sha256:{d['sha256']}\n"
        for name, d in sorted(distributions.items())
    )
    (output / "requirements.txt").write_text(lock)
    manifest = {
        "schema_version": 1,
        "kind": "jev-gateway-lock-v1",
        "source_commit": source["source_commit"],
        "source_manifest_sha256": digest,
        "source_core_sha256": sha(core),
        "requested_core_extras": extras,
        "resolver_sha256": sha(Path(__file__)),
        "target": target(),
        "distributions": distributions,
        "files": inventory(output),
    }
    save(output / "manifest.json", manifest)
    return {
        "path": str(output),
        "manifest_sha256": sha(output / "manifest.json"),
        "packages": len(distributions),
        "target": target(),
    }


INSPECT = """
import importlib.metadata as m, importlib.util, json, platform, sys, sysconfig
print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'target':{'python':f'{sys.version_info.major}.{sys.version_info.minor}','implementation':platform.python_implementation(),'platform':sys.platform,'machine':platform.machine(),'soabi':sysconfig.get_config_var('SOABI')},
 'packages':{d.metadata['Name']:d.version for d in m.distributions()},
 'core_files':{str(p):str(m.distribution('jev-runtime-core').locate_file(p))
               for p in m.distribution('jev-runtime-core').files
               if str(p).startswith('jev_runtime/') and str(p).endswith('.py')},
 'core_path':importlib.util.find_spec('jev_runtime').origin}))
"""


def inspect_install(environment: Path, locked: Path, digest: str) -> dict:
    manifest = verify(locked, digest, "jev-gateway-lock-v1")
    python = environment / "bin/python"
    data = json.loads(subprocess.check_output([str(python), "-I", "-c", INSPECT], text=True))
    expected = {name: d["version"] for name, d in manifest["distributions"].items()}
    installed = {canonical(name): version for name, version in data["packages"].items()}
    if installed != expected or data["target"] != manifest["target"]:
        raise ValueError("Installed package inventory or target differs from lock")
    if (
        Path(data["prefix"]).resolve() != environment.resolve()
        or data["prefix"] == data["base_prefix"]
    ):
        raise ValueError("Install must be an isolated virtual environment")
    wheel_path = locked / manifest["distributions"]["jev-runtime-core"]["wheel"]
    with zipfile.ZipFile(wheel_path) as wheel:
        files = {
            n: wheel.read(n)
            for n in wheel.namelist()
            if n.startswith("jev_runtime/") and n.endswith(".py")
        }
    if set(files) != set(data["core_files"]):
        raise ValueError("Installed core source inventory differs from wheel")
    for name, expected_bytes in files.items():
        path = Path(data["core_files"][name])
        if (
            not path.resolve().is_relative_to(environment.resolve())
            or path.read_bytes() != expected_bytes
        ):
            raise ValueError("Installed core source differs from wheel")
    data.update(
        source_commit=manifest["source_commit"],
        manifest_sha256=digest,
        verified_core_files=len(files),
    )
    return data


def install(locked: Path, digest: str, destination: Path) -> dict:
    manifest = verify(locked, digest, "jev-gateway-lock-v1")
    if target() != manifest["target"]:
        raise ValueError(
            "Run install with the same Python ABI, platform and architecture as resolve"
        )
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=False)
    save(
        destination / "jev-install-started.json",
        {"manifest_sha256": digest, "source_commit": manifest["source_commit"]},
    )
    run(
        [sys.executable, "-I", "-m", "venv", "--copies", str(destination)],
        destination / "jev-install.log",
    )
    python = str(destination / "bin/python")
    run(
        [
            python,
            "-I",
            "-m",
            "pip",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "install",
            "--no-index",
            "--find-links",
            str(locked.absolute() / "wheels"),
            "--only-binary=:all:",
            "--require-hashes",
            "--no-deps",
            "-r",
            str(locked.absolute() / "requirements.txt"),
        ],
        destination / "jev-install.log",
    )
    run([python, "-I", "-m", "pip", "check"], destination / "jev-install.log")
    verified = inspect_install(destination, locked, digest)
    imports = "import jev_runtime.api; import transformers"
    if "tiktoken" in manifest["distributions"]:
        imports += "; import tiktoken"
    run(
        [python, "-I", "-c", imports],
        destination / "jev-install.log",
    )
    save(destination / "jev-install-complete.json", verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--ref", required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("verify", "resolve", "install", "inspect-install"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--manifest-sha256", required=True)
        if name in ("resolve", "install"):
            p.add_argument("--output", type=Path, required=True)
        if name == "resolve":
            p.add_argument("--constraints", type=Path)
        if name == "inspect-install":
            p.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.repo, args.ref, args.output)
    elif args.command == "resolve":
        result = resolve(args.bundle, args.manifest_sha256, args.output, args.constraints)
    elif args.command == "install":
        result = install(args.bundle, args.manifest_sha256, args.output)
    elif args.command == "inspect-install":
        result = inspect_install(args.environment, args.bundle, args.manifest_sha256)
    else:
        result = verify(args.bundle, args.manifest_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
