"""Reproduce host-row normalization against the installed SGLang source on CPU."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
from jev_sglang.plugin import _before_logprob_rows
from sglang.srt.layers.logprob_processor import LogprobStage, get_token_ids_logprobs_raw
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)


def run(args):
    assert version("sglang").split("+")[0] == "0.5.19"
    if args.output.exists():
        raise ValueError("Preserve previous evidence")
    batch = SimpleNamespace(
        return_logprob=True, spec_algorithm=SimpleNamespace(is_none=lambda: True)
    )
    expected = [[-0.12345678912345678, -2.125], []]

    def output():
        values, ids = get_token_ids_logprobs_raw(
            torch.tensor([[-10.0, *expected[0]], [-1.0, -2.0, -3.0]], dtype=torch.float64),
            [[1, 2], None],
            stage=LogprobStage.DECODE,
            no_copy_to_cpu=True,
        )
        assert torch.is_tensor(values[0]) and type(values[1]) is list and values[1] == []
        return SimpleNamespace(
            next_token_token_ids_logprobs_val=values,
            next_token_token_ids_logprobs_idx=ids,
            next_token_logprobs=torch.tensor([-1.0, -2.0]),
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
            input_token_logprobs=None,
        )

    results = []
    for name in ("move_logprobs_to_cpu", "_normalize_decode_outputs"):
        original = inspect.unwrap(getattr(SchedulerBatchResultProcessor, name))
        extra = (
            {}
            if name == "move_logprobs_to_cpu"
            else {"result": None, "next_token_ids": torch.tensor([1, 2])}
        )
        try:
            original(None, batch=batch, logits_output=output(), **extra)
        except AttributeError as error:
            assert "tolist" in str(error)
            failure = str(error)
        else:
            raise AssertionError("The installed upstream no longer reproduces the incompatibility")
        adapted = output()
        tensor = adapted.next_token_token_ids_logprobs_val[0]
        ids = adapted.next_token_token_ids_logprobs_idx
        _before_logprob_rows(batch=batch, logits_output=adapted)
        assert adapted.next_token_token_ids_logprobs_val[0] is tensor
        original(None, batch=batch, logits_output=adapted, **extra)
        assert adapted.next_token_token_ids_logprobs_val == expected
        assert adapted.next_token_token_ids_logprobs_idx is ids
        results.append(
            {
                "method": name,
                "unpatched_error": failure,
                "exact_rows_after_hook": adapted.next_token_token_ids_logprobs_val,
            }
        )
    report = {
        "source_commit": args.source_commit,
        "qualification": "Installed upstream CPU producer/consumer reproduction; not GPU load",
        "sglang": version("sglang"),
        "torch": version("torch"),
        "upstream_files_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in {
                inspect.getfile(get_token_ids_logprobs_raw),
                inspect.getfile(SchedulerBatchResultProcessor),
            }
        },
        "cases": results,
        "passed": True,
    }
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
