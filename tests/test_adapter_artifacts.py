import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from jev_runtime.adapters import AdapterStore
from jev_runtime.errors import JevError


@pytest.fixture
def adapter_source(tmp_path):
    source = tmp_path / "approved" / "source"
    source.mkdir(parents=True)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "fixture",
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj"],
        "bias": "none",
    }
    (source / "adapter_config.json").write_text(json.dumps(config))
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones(
            (2, 4), dtype=np.float32
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones(
            (4, 2), dtype=np.float32
        ),
    }
    save_file(tensors, source / "adapter_model.safetensors")
    return source


def test_adapter_store_freezes_bytes_and_binds_base_revision(tmp_path, adapter_source):
    store = AdapterStore(tmp_path / "store", (str(adapter_source.parent),))
    artifact = store.register("task", str(adapter_source), "fixture", "a" * 40)
    store.verify(artifact)
    same = store.register("task", str(adapter_source), "fixture", "a" * 40)
    assert same == artifact
    different = store.register("task", str(adapter_source), "fixture", "b" * 40)
    assert different.revision != artifact.revision
    (adapter_source / "adapter_config.json").write_text("{}")
    store.verify(artifact)
    with pytest.raises(JevError) as exc:
        store.register("task", str(adapter_source), "fixture", "a" * 40)
    assert exc.value.code == "adapter_format"
    tensor_path = Path(artifact.path) / "adapter_model.safetensors"
    tensor_path.chmod(0o644)
    tensor_path.write_bytes(b"tampered")
    with pytest.raises(JevError) as exc:
        store.verify(artifact)
    assert exc.value.code == "adapter_corrupted"


def test_adapter_store_rejects_external_paths_and_symlink_files(tmp_path, adapter_source):
    store = AdapterStore(tmp_path / "store", (str(tmp_path / "approved"),))
    with pytest.raises(JevError) as exc:
        store.register("task", str(tmp_path), "fixture", "a" * 40)
    assert exc.value.code == "adapter_path"
    source = adapter_source / "adapter_model.safetensors"
    outside = tmp_path / "weights.safetensors"
    source.rename(outside)
    source.symlink_to(outside)
    with pytest.raises(JevError) as exc:
        store.register("task", str(adapter_source), "fixture", "a" * 40)
    assert exc.value.code == "adapter_path"


def test_adapter_store_refuses_budget_and_non_lora_tensors(tmp_path, adapter_source):
    small = AdapterStore(tmp_path / "small", (str(adapter_source.parent),), max_bytes=8)
    with pytest.raises(JevError) as exc:
        small.register("task", str(adapter_source), "fixture", "a" * 40)
    assert exc.value.code == "adapter_budget"
    save_file(
        {"lm_head.weight": np.ones((2, 4), dtype=np.float32)},
        adapter_source / "adapter_model.safetensors",
    )
    store = AdapterStore(tmp_path / "store", (str(adapter_source.parent),))
    with pytest.raises(JevError) as exc:
        store.register("task", str(adapter_source), "fixture", "a" * 40)
    assert exc.value.code == "adapter_tensor"
