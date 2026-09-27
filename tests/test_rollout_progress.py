import copy
from contextlib import ExitStack
from types import SimpleNamespace

import httpx
import pytest

from tests.integration.rollout_native_cancel import pin_workers, validate_progress


@pytest.mark.parametrize("failure", [False, True])
def test_worker_connections_are_not_reentered_and_all_close(monkeypatch, failure):
    original = httpx.Client
    made = []
    owners = iter(["a", "a", "b"])

    def factory(**kwargs):
        owner = next(owners)

        def handler(request):
            if failure and len(made) == 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"worker_id": owner})

        client = original(**kwargs, transport=httpx.MockTransport(handler))
        made.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", factory)
    probe = SimpleNamespace(snapshot=lambda _: {"workers": ["a", "b"]})
    if failure:
        with pytest.raises(httpx.HTTPStatusError), ExitStack() as stack:
            pin_workers(stack, probe, {"blue": 19000}, {})
    else:
        with ExitStack() as stack:
            clients, rosters = pin_workers(stack, probe, {"blue": 19000}, {})
            assert set(clients) == rosters["blue"] == {"a", "b"}
            assert made[1].is_closed
            assert clients["a"].get("/ready").json()["worker_id"] == "a"
            assert not clients["a"].is_closed and not clients["b"].is_closed
    assert all(client.is_closed for client in made)


@pytest.mark.parametrize(
    "change", ["none", "lease", "owner", "finished", "only_journaled", "failed", "no_active"]
)
def test_switch_evidence_rejects_unexecuted_finished_or_changed_requests(change):
    before = {
        "scope": "local_worker",
        "worker_id": "worker-a",
        "request": {
            "stage": "execute",
            "scoring_sequences": 128,
            "request_id": "request",
            "bundle": "b@1",
            "bundle_digest": "digest",
            "generation": 1,
            "lease_id": "lease-a",
            "work": {
                "engine_call": {"started": 3, "succeeded": 2, "active": 1, "failed_or_cancelled": 0}
            },
        },
    }
    after = copy.deepcopy(before)
    engine = after["request"]["work"]["engine_call"]
    if change == "lease":
        after["request"]["lease_id"] = "lease-b"
    elif change == "owner":
        after["worker_id"] = "worker-b"
    elif change == "finished":
        engine.update(started=128, succeeded=128, active=0)
    elif change == "only_journaled":
        engine.update(started=0, succeeded=0, active=0)
    elif change == "failed":
        engine.update(failed_or_cancelled=1, succeeded=1)
    elif change == "no_active":
        engine.update(started=2, active=0)
    else:
        engine.update(started=4, succeeded=3)
    if change == "none":
        validate_progress(before, after, 128)
    else:
        with pytest.raises(AssertionError):
            validate_progress(before, after, 128)
