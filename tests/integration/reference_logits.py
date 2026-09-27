"""Compare engine readout with an independent Transformers forward on CPU.

This is numerical evidence for the saved answer position, not throughput or
quality evidence. The input IDs and checkpoint revision come from the live report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.15)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--readout-dtype", choices=["model", "float32"], default="model")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to retain previous reference attempts")
    source = json.loads((args.model_path / "jev-source.json").read_text())
    report = json.loads(args.contract_report.read_text())
    assert source["revision"] == report["model"]["revision"]
    assert source["model_id"] == report["model"]["id"]
    assert report["model"]["quantization"] is None
    assert report["model"]["dtype"] == "bfloat16"
    readout_dtype = "float32" if args.readout_dtype == "float32" else "bfloat16"
    if (report["model"].get("readout_dtype") or "bfloat16") != readout_dtype:
        parser.error("Reference readout dtype must match the saved serving model identity")
    torch.set_num_threads(2)
    if args.device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        weight_bytes = sum(p.stat().st_size for p in args.model_path.glob("*.safetensors"))
        if free < weight_bytes * 1.5 + 3 * 1024**3:
            raise RuntimeError("Insufficient GPU memory for reference weights and 3 GiB reserve")
    started = time.time()
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            dtype=torch.bfloat16,
            trust_remote_code=False,
            attn_implementation=args.attention,
        )
        .eval()
        .to(args.device)
    )
    example = report["reference_logits"]
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=args.device)
    hook = None
    observed_readout = None

    def project(module, inputs, output):
        nonlocal observed_readout
        if readout_dtype == "float32":
            # Recompute at the module boundary; changing a tied parameter's dtype
            # would also change the input embedding and the reference backbone.
            output = torch.nn.functional.linear(
                inputs[0].float(),
                module.weight.float(),
                module.bias.float() if module.bias is not None else None,
            )
        observed_readout = str(output.dtype).removeprefix("torch.")
        return output

    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("Reference requires an unquantized Linear output head")
    hook = head.register_forward_hook(project)
    try:
        with torch.inference_mode():
            logits = model(input_ids, use_cache=False).logits[0, -1].float()
            scores = torch.log_softmax(logits, dim=-1)[example["label_ids"]].tolist()
    finally:
        hook.remove()
    assert observed_readout == readout_dtype
    errors = [abs(a - b) for a, b in zip(scores, example["logprobs"], strict=True)]
    actual_top = max(range(len(scores)), key=scores.__getitem__)
    expected_top = max(range(len(scores)), key=example["logprobs"].__getitem__)
    result = {
        "model_id": source["model_id"],
        "revision": source["revision"],
        "reference": f"Transformers {args.device} BF16 {args.attention} forward; FP32 log_softmax",
        "readout_dtype": readout_dtype,
        "observed_readout_dtype": observed_readout,
        "contract_report": str(args.contract_report),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "engine_logprobs": example["logprobs"],
        "reference_logprobs": scores,
        "absolute_errors": errors,
        "atol": args.atol,
        "top_label_equal": actual_top == expected_top,
        "passed": max(errors) <= args.atol and actual_top == expected_top,
        "started_at": started,
        "finished_at": time.time(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
