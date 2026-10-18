"""Narrow the pinned SGLang source install to Jev's text-only HTTP lane."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CUDA_SUBSTITUTIONS = (
    ("cuda-python>=13.0", "cuda-python>=12,<13"),
    ("flashinfer_python[cu13]", "flashinfer_python[cu12]"),
    ("nvidia-cutlass-dsl[cu13]", "nvidia-cutlass-dsl"),
)

# These upstream base dependencies serve clients, datasets, gRPC, audio, or
# multimodal model paths that the pinned Jev SGLang profile does not expose.
REMOVED_REQUIREMENTS = frozenset(
    {
        "anthropic",
        "datasets",
        "modelscope",
        "smg-grpc-servicer",
        "soundfile",
        "timm",
        "torchaudio",
        "torchcodec",
    }
)


def patch_pyproject(path: Path) -> tuple[str, ...]:
    text = path.read_text()
    for old, new in CUDA_SUBSTITUTIONS:
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            raise RuntimeError(f"Official CUDA dependency no longer contains {old!r}")

    project = text.index("[project]")
    dependencies = text.index("dependencies = [", project)
    end = text.index("\n]", dependencies) + 2
    block = text[dependencies:end]
    found: set[str] = set()
    kept: list[str] = []
    for line in block.splitlines(keepends=True):
        match = re.match(r'\s*"([A-Za-z0-9_.-]+)', line)
        name = match.group(1).lower().replace("_", "-") if match else None
        if name in REMOVED_REQUIREMENTS:
            found.add(name)
        else:
            kept.append(line)
    missing = REMOVED_REQUIREMENTS - found
    if missing:
        raise RuntimeError(
            f"Official text-only dependency filter no longer matches: {sorted(missing)}"
        )

    path.write_text(text[:dependencies] + "".join(kept) + text[end:])
    return tuple(sorted(found))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyproject", type=Path)
    args = parser.parse_args()
    removed = patch_pyproject(args.pyproject)
    print("Removed non-text SGLang dependencies: " + ", ".join(removed))


if __name__ == "__main__":
    main()
