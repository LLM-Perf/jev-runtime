#!/usr/bin/env bash
# Isolated text-only CUDA 12.9 environment for the inspected R550 DSW host.
# Sources: SGLang v0.5.19 docker/Dockerfile and python/setup.py.
# JEV_ROOT and JEV_RELEASE must refer to task-owned directories.
set -euo pipefail
: "${JEV_ROOT:?Set the task-owned environment root}"
: "${JEV_RELEASE:?Set the checkout containing this Jev release}"
: "${JEV_PYTHON:?Set an existing Python 3.12 executable}"
JEV_ENV="$JEV_ROOT/envs/sglang"
JEV_SOURCE="$JEV_ROOT/sources/sglang-v0.5.19"
mkdir -p "$JEV_ROOT/sources" "$JEV_ROOT/evidence"
if ! test -x "$JEV_ENV/bin/python"; then
  "$JEV_PYTHON" -m venv "$JEV_ENV"
fi
JEV_PIP_PY="$JEV_ENV/bin/python"
if ! test -d "$JEV_SOURCE/.git"; then
  git clone --depth 1 --branch v0.5.19 https://github.com/sgl-project/sglang.git "$JEV_SOURCE"
fi
git -C "$JEV_SOURCE" rev-parse HEAD > "$JEV_ROOT/evidence/sglang-source-revision.txt"
"$JEV_PIP_PY" -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  'torch==2.13.0+cu129' 'torchvision==0.28.0+cu129' \
  setuptools setuptools-rust setuptools-scm wheel
"$JEV_PIP_PY" -m pip install --no-deps \
  'https://github.com/sgl-project/whl/releases/download/v0.4.6.post1/sglang_kernel-0.4.6.post1+cu129-cp310-abi3-manylinux2014_x86_64.whl' \
  'https://github.com/sgl-project/whl/releases/download/v0.1.7/sgl_deep_gemm-0.1.7+cu129-py3-none-manylinux2014_x86_64.whl'
"$JEV_PIP_PY" -m pip install --index-url https://docs.sglang.ai/whl/cu129/ \
  --no-deps 'sgl-deep-ep==0.1.2+cu129'
# This dependency is a wheel stub with its own isolated build requirements.
"$JEV_PIP_PY" -m pip install --index-url https://pypi.org/simple 'cuda-tile==1.6.0rc5'
"$JEV_PYTHON" "$JEV_RELEASE/deployment/sglang_text_dependencies.py" \
  "$JEV_SOURCE/python/pyproject.toml"
git -C "$JEV_SOURCE" diff -- python/pyproject.toml \
  > "$JEV_ROOT/evidence/sglang-cu129-dependencies.patch"
# The official setup.py supports this flag. Optional gRPC/multimodal Rust
# extensions are excluded; this environment only certifies the text HTTP path.
SGLANG_BUILD_RUST_EXTS=none "$JEV_PIP_PY" -m pip install --no-build-isolation \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 \
  -e "$JEV_SOURCE/python"
"$JEV_PIP_PY" -m pip install -e "$JEV_RELEASE[tokenizers]" -e "$JEV_RELEASE/packages/sglang"
"$JEV_PIP_PY" -m pip check
"$JEV_PIP_PY" - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

unexpected = []
for name in (
    "anthropic",
    "datasets",
    "grpcio",
    "modelscope",
    "smg-grpc-servicer",
    "soundfile",
    "timm",
    "torchaudio",
    "torchcodec",
):
    try:
        distribution(name)
    except PackageNotFoundError:
        continue
    unexpected.append(name)
if unexpected:
    raise RuntimeError(f"Non-text dependencies were installed: {unexpected}")
print("Verified text-only dependency boundary")
PY
"$JEV_PIP_PY" -m pip freeze > "$JEV_ROOT/evidence/sglang-requirements.txt"
