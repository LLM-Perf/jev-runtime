"""Compare resident and offloaded CUDA references, including failure cleanup.

Run on one explicitly allocated GPU with a checkpoint that fits the allocator
budget. This validates the helper, not either serving engine or model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from reference_device import CudaLeafOffload, cuda_budget


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    if torch.cuda.device_count() != 1:
        parser.error("Expose exactly one allocated GPU")
    torch.set_num_threads(2)
    torch.manual_seed(917)
    free, total = torch.cuda.mem_get_info()
    budget = cuda_budget(free, total, 6144, 3072)
    weight_bytes = sum(p.stat().st_size for p in args.model_path.glob("*.safetensors"))
    if weight_bytes * 1.5 > budget:
        parser.error("Resident comparison model exceeds allocator budget")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    source = json.loads((args.model_path / "jev-source.json").read_text())
    contract = json.loads(args.contract_report.read_text())
    assert source["revision"] == contract["model"]["revision"]
    assert source["model_id"] == contract["model"]["id"]
    result = {
        "qualification": "CUDA reference helper equality and cleanup; not engine certification",
        "started_at": time.time(),
        "source": source,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path(__file__).with_name("reference_device.py").read_bytes()
        ).hexdigest(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "allocator_budget_bytes": budget,
        "free_bytes_before": free,
        "models": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for name in ("tiny_qwen2", "tiny_llama_tied", "checkpoint"):
            if name == "checkpoint":
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                    trust_remote_code=False,
                    local_files_only=True,
                ).eval()
                ids = contract["reference_logits"]["input_ids"]
            else:
                config_type = (
                    transformers.Qwen2Config if name == "tiny_qwen2" else transformers.LlamaConfig
                )
                config = config_type(
                    vocab_size=97,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    tie_word_embeddings=name == "tiny_llama_tied",
                )
                model = (
                    transformers.AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
                )
                ids = [1, 3, 57, 29, 87, 6, 32]
            state_before = {k: v.clone() for k, v in model.state_dict().items()}
            head = model.get_output_embeddings()
            tied = head.weight is model.get_input_embeddings().weight
            row = {"name": name, "tied_weights": tied, "comparisons": []}
            result["models"].append(row)
            for attention in ("eager", "sdpa"):
                model.set_attn_implementation(attention)
                for readout in ("bfloat16", "float32"):

                    def project(module, inputs, output, readout=readout):
                        if readout == "float32":
                            return torch.nn.functional.linear(
                                inputs[0].float(),
                                module.weight.float(),
                                None if module.bias is None else module.bias.float(),
                            )
                        return output

                    handle = head.register_forward_hook(project)
                    try:
                        inputs = torch.tensor([ids], device="cuda")
                        with torch.inference_mode():
                            model.to("cuda")
                            resident = model(inputs, use_cache=False).logits.cpu()
                            model.to("cpu")
                            torch.cuda.empty_cache()
                            with CudaLeafOffload(model) as offload:
                                offloaded = model(inputs, use_cache=False).logits.cpu()
                        comparison = {
                            "attention": attention,
                            "readout_dtype": readout,
                            "logits_shape": list(resident.shape),
                            "bitwise_equal": torch.equal(resident, offloaded),
                            "max_absolute_error": (resident.float() - offloaded.float())
                            .abs()
                            .max()
                            .item(),
                            "placement": offload.metadata(),
                        }
                        row["comparisons"].append(comparison)
                        assert comparison["bitwise_equal"], comparison
                        assert not comparison["placement"]["unvisited_modules"]
                    finally:
                        handle.remove()
                        model.to("cpu")
                    assert tied == (head.weight is model.get_input_embeddings().weight)
                    assert all(
                        torch.equal(v, state_before[k]) for k, v in model.state_dict().items()
                    )
                    save()

            # Exercise cleanup after an exception while a module's state is on CUDA.
            def fail(module, inputs, output):
                raise RuntimeError("injected_reference_failure")

            handle = head.register_forward_hook(fail)
            try:
                with CudaLeafOffload(model), torch.inference_mode():
                    model(inputs, use_cache=False)
            except RuntimeError as exc:
                assert str(exc) == "injected_reference_failure"
            else:
                raise AssertionError("Injected failure was not observed")
            finally:
                handle.remove()
            assert all(t.device.type == "cpu" for t in model.parameters())
            assert all(t.device.type == "cpu" for t in model.buffers())
            assert all(torch.equal(v, state_before[k]) for k, v in model.state_dict().items())
            assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
            row["state_unchanged"] = row["exception_cleanup_passed"] = True
            del model, head, state_before, resident, offloaded, inputs
            torch.cuda.empty_cache()
            save()
        result["passed"] = True
    except BaseException as exc:
        result["passed"] = False
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result["finished_at"] = time.time()
        result["free_bytes_after"] = torch.cuda.mem_get_info()[0]
        save()
        print(json.dumps(result))


if __name__ == "__main__":
    main()
