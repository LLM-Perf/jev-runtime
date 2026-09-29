"""Check pinned upstream sample() ordering using Torch CPU primitive doubles.

This executes the unchanged upstream method body, not a TokenSpeed model engine.
It does NOT exercise native kernels, the scheduler, transport, CUDA or performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from jev_tokenspeed.plugin import verify_source


def method(path, cls_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name
    )
    node = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    node.decorator_list = []  # Remove the NVTX profiler decorator only.
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[method_name]


def validate(root: Path) -> dict:
    import torch

    revision = verify_source(root)
    namespace = {
        "torch": torch,
        "nan_guard_logits": lambda logits, enabled: logits,
        "selected_token_logprobs": lambda logits, tokens, out: out.copy_(
            torch.log_softmax(logits.float(), -1).gather(1, tokens.long()[:, None]).squeeze(1)
        ),
    }
    sample = method(
        root / "runtime/sampling/backends/triton_full.py",
        "TritonFullSamplingBackend",
        "sample",
        namespace,
    )
    write = method(
        root / "runtime/sampling/backends/triton.py",
        "TritonSamplingBackend",
        "_write_logprob_outputs",
        namespace,
    )
    reports = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for rows in (1, 3):
            logits = torch.linspace(-4, 4, rows * 521).reshape(rows, 521).to(dtype)
            target = torch.tensor([0, 2, 257][:rows], dtype=torch.int64)
            expected = torch.log_softmax(logits.float(), -1).gather(1, target[:, None]).squeeze(1)
            output = SimpleNamespace(next_token_logits=logits.clone())
            backend = SimpleNamespace(
                config=SimpleNamespace(enable_nan_detection=False, enable_output_logprobs=True),
                _selected_logprob_out=torch.empty(rows),
                _gumbel_out=torch.empty(rows, dtype=torch.int32),
                _ones_buf=torch.ones(rows, dtype=torch.int32),
                _zero_offsets_pool=torch.zeros(rows, dtype=torch.int32),
                _req_pool_indices_for_kernels=lambda indices, size: indices,
                _gumbel_sample_full_logits=lambda values, *args: values.argmax(-1),
                maybe_broadcast=lambda values: None,
                _accumulate_counts=lambda *args: None,
            )

            def bias(values, indices, rows=rows, target=target):
                values[torch.arange(rows), target] += 16384.0
                return values

            backend._apply_penalties_and_bias = bias
            backend._write_logprob_outputs = partial(write, backend)
            info = SimpleNamespace(
                vocab_mask=None, req_pool_indices=torch.arange(rows), valid_cache_lengths=None
            )
            tokens, lengths = sample(backend, output, info)
            error = float((expected - output.next_token_logprobs).abs().max())
            passed = bool(
                torch.equal(tokens.long(), target) and (lengths == 1).all() and error <= 1e-6
            )
            # A score calculated after forcing would be near zero, while these
            # deliberately low-probability labels have raw logprobs below -4.
            assert bool((expected < -4).all())
            reports.append(
                {"dtype": str(dtype), "rows": rows, "max_absolute_error": error, "passed": passed}
            )
    return {
        "upstream_revision": revision,
        "scope": "upstream Python sample/write bodies with Torch CPU primitive doubles",
        "native_tokenspeed_engine_executed": False,
        "native_tokenspeed_kernels_executed": False,
        "gpu_executed": False,
        "model_executed": False,
        "torch_version": torch.__version__,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_profile_sha256": hashlib.sha256(
            Path(__import__("jev_tokenspeed.plugin", fromlist=["x"]).__file__)
            .with_name("source-profile.json")
            .read_bytes()
        ).hexdigest(),
        "cases": reports,
        "passed": all(row["passed"] for row in reports),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_root)
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
