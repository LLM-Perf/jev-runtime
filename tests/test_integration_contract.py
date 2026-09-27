import json
from pathlib import Path

import pytest

from jev_runtime.schema import DecisionResponse
from tests.integration.live_contract import verify_four_types


def test_real_rank_tie_is_a_valid_default_policy_abstention():
    path = Path(__file__).parents[1] / "evidence/dsw/vllm-deepseek15b-ad9b417-abstention.json"
    response = DecisionResponse.model_validate(json.loads(path.read_text())["response"])
    assert response.answers["routing"].status == "abstained"
    assert response.answers["routing"].reason == "tie"
    verify_four_types(response)
    with pytest.raises(AssertionError):
        verify_four_types(response, require_selected=True)


def test_explicit_first_policy_can_return_deterministic_tied_rank():
    path = Path(__file__).parents[1] / "evidence/dsw/vllm-deepseek15b-ad9b417-abstention.json"
    payload = json.loads(path.read_text())["response"]
    payload["answers"]["routing"].update(
        status="answered", value=["payments", "sales", "engineering"], abstained=False, reason=None
    )
    verify_four_types(DecisionResponse.model_validate(payload), require_selected=True)
