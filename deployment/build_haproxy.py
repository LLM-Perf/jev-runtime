"""Build the qualified HAProxy in a new task-owned directory; no package manager changes."""

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
import urllib.request
from pathlib import Path

VERSION = "3.2.24"
SHA256 = "f765638cc4819f25e20d974a4a1bc24ed54342467c88363d7fc34a1fa95b725a"
URL = f"https://www.haproxy.org/download/3.2/src/haproxy-{VERSION}.tar.gz"


def build(destination: Path):
    if platform.system() != "Linux":
        raise RuntimeError("The qualified proxy build profile targets Linux")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "source.tar.gz"
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(16_777_216)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Pinned HAProxy source digest mismatch")
    archive.write_bytes(data)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    tree = destination / f"haproxy-{VERSION}"
    command = ["make", "-C", str(tree), "-j4", "TARGET=linux-glibc", "USE_OPENSSL=1", "USE_ZLIB=1"]
    with (destination / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = tree / "haproxy"
    version = subprocess.check_output([str(binary), "-vv"], text=True)
    record = {
        "source_url": URL,
        "source_sha256": SHA256,
        "build_command": command,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "version": version,
    }
    (destination / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination)
