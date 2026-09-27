"""Pinned real-tokenizer coverage; explicitly not GPU or model-quality certification."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

from jev_runtime.config import Settings, TokenizerOptions, load_compiler, model_identity
from jev_runtime.schema import Bundle, Option, Policy, Question, TemplateSpec
from jev_runtime.templates import TemplateFile


def probe(
    item: dict,
    tokenizer: str | None = None,
    chat_template: TemplateFile | None = None,
    tokenizer_options: TokenizerOptions | None = None,
) -> dict:
    result = {
        "model": item["id"],
        "revision": item["revision"],
        "started_at": time.time(),
        "cases": [],
        "passed": False,
        "explicit_chat_template": chat_template.model_dump() if chat_template else None,
        "tokenizer_options": (tokenizer_options or TokenizerOptions()).model_dump(
            exclude_none=True
        ),
    }
    try:
        settings = Settings(
            backend="sglang",
            model_id=item["id"],
            model_revision=item["revision"],
            tokenizer=tokenizer,
            chat_template=chat_template,
            tokenizer_options=tokenizer_options or TokenizerOptions(),
        )
        compiler = load_compiler(settings)
        result["model_identity"] = model_identity(settings, compiler).model_dump(mode="json")
        # A system-role-free profile is explicit, never a silent rewrite of a
        # failing template. Report both independently for deployment selection.
        for system_role in (True, False):
            for mode in ("joint-label", "independent-candidate"):
                for count in (2, 8, 32, 64):
                    case = {"system_role": system_role, "mode": mode, "candidates": count}
                    try:
                        bundle = Bundle(
                            id="tokenizer-probe",
                            version=1,
                            model=model_identity(settings, compiler),
                            template=TemplateSpec(mode=mode, use_system_role=system_role),
                            policy=Policy(max_options=64),
                        )
                        question = Question(
                            id="choice",
                            type="choice",
                            instruction="Select the matching category.",
                            options=tuple(
                                Option(id=f"c{i}", description=f"Category {i}")
                                for i in range(count)
                            ),
                        )
                        compiled = compiler.compile("A refund is requested.", question, bundle, "p")
                        expected_sequences = count if mode == "independent-candidate" else 1
                        expected_labels = 2 if mode == "independent-candidate" else count
                        assert len(compiled.sequences) == expected_sequences
                        for seq in compiled.sequences:
                            assert len(seq.label_ids) == len(set(seq.label_ids)) == expected_labels
                            assert seq.input_ids
                        case.update(
                            passed=True,
                            sequences=len(compiled.sequences),
                            labels_per_sequence=expected_labels,
                            logical_prompt_tokens=sum(len(s.input_ids) for s in compiled.sequences),
                            first_label_ids=list(compiled.sequences[0].label_ids),
                        )
                    except Exception as exc:
                        case.update(
                            passed=False,
                            error={"type": type(exc).__name__, "message": str(exc)[:500]},
                        )
                    result["cases"].append(case)
        result["passed"] = all(case["passed"] for case in result["cases"])
        result["deployable_profiles"] = [
            {"system_role": role, "mode": mode}
            for role in (True, False)
            for mode in ("joint-label", "independent-candidate")
            if all(
                c["passed"]
                for c in result["cases"]
                if c["system_role"] == role and c["mode"] == mode
            )
        ]
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    result["finished_at"] = time.time()
    return result


def main(args):
    inventory = json.loads(args.inventory.read_text())["models"]
    selected = [item for item in inventory if not args.model or item["id"] in args.model]
    if not selected or (args.model and set(args.model) != {item["id"] for item in selected}):
        raise SystemExit("Requested model is not in the frozen inventory")
    if args.output.exists():
        raise SystemExit("Output exists; retain previous attempts and choose a new artifact")
    template = None
    options = TokenizerOptions(fix_mistral_regex=args.fix_mistral_regex)
    if args.fix_mistral_regex is not None and len(selected) != 1:
        raise SystemExit("An explicit tokenizer profile must select exactly one pinned model")
    if args.chat_template_path:
        if len(selected) != 1:
            raise SystemExit("An explicit template probe must select exactly one pinned model")
        template = TemplateFile(
            path=str(args.chat_template_path.resolve(strict=True)),
            sha256=args.chat_template_sha256,
            format=args.chat_template_format,
        )
    report = {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "qualification": "real tokenizer compilation only; no GPU or quality certification",
        "selected_count": len(selected),
        "transformers_version": version("transformers"),
        "models": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in selected:
        local_path = None
        if args.local_models:
            local_path = args.local_models / item["id"].rsplit("/", 1)[-1]
            manifest = json.loads((local_path / "jev-source.json").read_text())
            if (manifest["model_id"], manifest["revision"]) != (item["id"], item["revision"]):
                raise SystemExit("Local checkpoint identity differs from frozen inventory")
        result = probe(item, str(local_path) if local_path else None, template, options)
        report["models"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": item["id"],
                    "passed": result["passed"],
                    "cases_passed": sum(c["passed"] for c in result["cases"]),
                    "cases_total": len(result["cases"]),
                }
            ),
            flush=True,
        )
    report["complete"] = True
    report["passed_models"] = sum(item["passed"] for item in report["models"])
    report["models_with_deployable_profile"] = sum(
        bool(item.get("deployable_profiles")) for item in report["models"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--local-models", type=Path)
    parser.add_argument("--chat-template-path", type=Path)
    parser.add_argument("--chat-template-sha256")
    parser.add_argument("--chat-template-format", choices=["jinja", "json"], default="jinja")
    parser.add_argument("--fix-mistral-regex", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if bool(args.chat_template_path) != bool(args.chat_template_sha256):
        parser.error("chat-template-path and chat-template-sha256 must be supplied together")
    main(args)
