from types import ModuleType, SimpleNamespace

import pytest
from jev_sglang import plugin


def test_mixed_host_and_tensor_logprobs_preserve_exact_values_and_empty_rows():
    # This is the documented producer contract: tensor rows for selected IDs,
    # an empty host list for the ordinary generation request in the same batch.
    tensor = object()
    precise = [-0.12345678912345678, -9007199254740991]
    rows = [tensor, [], precise]
    ids = [[7], [], [8, 9]]
    output = SimpleNamespace(
        next_token_token_ids_logprobs_val=rows, next_token_token_ids_logprobs_idx=ids
    )
    plugin._before_logprob_rows(batch=SimpleNamespace(return_logprob=True), logits_output=output)
    assert output.next_token_token_ids_logprobs_val is rows and rows[0] is tensor
    assert rows[1].tolist() == []
    assert rows[2].tolist() == precise and type(rows[2].tolist()[1]) is int
    assert output.next_token_token_ids_logprobs_idx is ids
    wrapped = rows[1]
    plugin._before_logprob_rows(batch=SimpleNamespace(return_logprob=True), logits_output=output)
    assert rows[1] is wrapped


def test_non_logprob_batches_and_absent_selected_rows_are_untouched():
    output = SimpleNamespace(next_token_token_ids_logprobs_val=[[]])
    row = output.next_token_token_ids_logprobs_val[0]
    plugin._before_logprob_rows(batch=SimpleNamespace(return_logprob=False), logits_output=output)
    assert output.next_token_token_ids_logprobs_val[0] is row and type(row) is list
    output.next_token_token_ids_logprobs_val = None
    plugin._before_logprob_rows(batch=SimpleNamespace(return_logprob=True), logits_output=output)
    assert output.next_token_token_ids_logprobs_val is None


@pytest.mark.parametrize("version,expected", [("0.5.19", 2), ("0.5.19+cu129", 2), ("0.5.20", 0)])
def test_compatibility_hooks_are_registered_only_for_the_affected_engine(
    monkeypatch, version, expected
):
    import sys

    registered = []
    host = ModuleType("sglang.srt.plugins.hook_registry")
    host.HookRegistry = SimpleNamespace(register=lambda *args: registered.append(args))
    host.HookType = SimpleNamespace(BEFORE="before", AFTER="after")
    monkeypatch.setitem(sys.modules, host.__name__, host)
    monkeypatch.setattr(plugin, "_registered", False)
    monkeypatch.setattr(plugin, "version", lambda name: version)
    monkeypatch.delenv("JEV_CONFIG", raising=False)
    plugin.register()
    matches = [row for row in registered if row[1] is plugin._before_logprob_rows]
    assert len(matches) == expected
    assert all(row[2] == "before" for row in matches)
    if matches:
        assert {row[0].rsplit(".", 1)[1] for row in matches} == {
            "move_logprobs_to_cpu",
            "_normalize_decode_outputs",
        }
