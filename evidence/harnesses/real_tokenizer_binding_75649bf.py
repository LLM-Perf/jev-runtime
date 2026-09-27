import asyncio
import hashlib
import importlib.metadata as metadata
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import jev_runtime.config
from jev_runtime.config import Settings, build_runtime, compiler_for_tokenizer, load_compiler
from jev_runtime.errors import JevError

repo = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
rev = "68faf511d618ef198fef186659617cfd2eb8e33a"
source = "75649bfb633ab5d6219e4b1c6bc4d7e9b9780bd5"


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    a = p.parse_args()
    root = Path("/root/jev-runtime")
    output = root / f"real-tokenizer-binding-{a.engine}-75649bf.json"
    assert not output.exists()
    report = {
        "runtime_source_commit": source,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "engine_environment": a.engine,
        "transformers_version": metadata.version("transformers"),
        "model_id": repo,
        "revision": rev,
        "qualification": (
            "Real tokenizer assets and runtime startup boundary; no engine/GPU scoring is performed"
        ),
        "started_at": time.time(),
    }
    try:
        template = Path(hf_hub_download(repo, "chat_template.json", revision=rev))
        digest = hashlib.sha256(template.read_bytes()).hexdigest()
        settings = Settings(
            backend=a.engine,
            model_id=repo,
            model_revision=rev,
            chat_template={"path": str(template), "sha256": digest, "format": "json"},
            tokenizer_options={"fix_mistral_regex": True},
            registry_path=str(root / f"rejected-tokenizer-{a.engine}-75649bf.db"),
        )
        assert not Path(settings.registry_path).exists()
        unfixed = AutoTokenizer.from_pretrained(
            repo, revision=rev, trust_remote_code=False, fix_mistral_regex=False
        )
        actual = compiler_for_tokenizer(settings, unfixed)
        expected = load_compiler(settings)
        report["vocabulary_equal"] = actual.tokenizer_digest == expected.tokenizer_digest
        report["implementation_equal"] = (
            actual.tokenizer_implementation_digest == expected.tokenizer_implementation_digest
        )
        report["actual_implementation_digest"] = actual.tokenizer_implementation_digest
        report["expected_implementation_digest"] = expected.tokenizer_implementation_digest
        assert report["vocabulary_equal"] and not report["implementation_equal"]
        try:
            asyncio.run(build_runtime(settings, native_backend=object(), compiler=actual))
        except JevError as e:
            report["rejection"] = {"code": e.code, "message": str(e)}
            assert e.code == "tokenizer_implementation_mismatch"
        else:
            raise AssertionError("Mismatched real tokenizer implementation was accepted")
        assert not Path(settings.registry_path).exists()
        assert source in jev_runtime.config.__file__
        report["runtime_path"] = jev_runtime.config.__file__
        report["registry_created"] = False
        report["passed"] = True
    except BaseException as e:
        report["passed"] = False
        report["failure"] = {"type": type(e).__name__, "message": str(e)[:2000]}
        raise
    finally:
        report["finished_at"] = time.time()
        with output.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
