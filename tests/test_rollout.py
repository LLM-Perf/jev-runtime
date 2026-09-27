import copy
import json

import httpx
import pytest

from jev_runtime.rollout import GatewayProbe, Rollout, RolloutError


class Proxy:
    selected = "blue"
    calls = 0
    sessions = 0
    unknown_reply = False

    def active(self):
        return self.selected

    def select(self, slot):
        self.selected = slot
        self.calls += 1
        if self.unknown_reply:
            raise TimeoutError("reply lost after map update")

    def identity(self):
        return {"Pid": "123", "Version": "3.2.24", "Nbthread": "1"}

    def outstanding(self, slot):
        return {
            "gateway": {"scur": self.sessions, "qcur": 0},
            "BACKEND": {"scur": self.sessions, "qcur": 0},
        }


class Probe:
    pending_leases = []
    fail_canary = False

    def __init__(self):
        self.profiles = {}
        for port, slot in ((19001, "blue"), (19002, "green")):
            self.profiles[port] = {
                "deployment": {
                    "protocol": 1,
                    "id": slot + "-1",
                    "release": "release-1",
                    "expected_workers": 2,
                    "registry": {"host": "host", "device": 1, "inode": 2},
                },
                "backend": "sglang:engine",
                "auth_policy": "same-auth-contract",
                "model": {"id": "fixture", "revision": "a" * 40},
                "capabilities": {"engine": "sglang"},
                "routes": [{"alias": "model", "ref": "default@1", "generation": 1}],
                "bindings": [["default@1", "sha256:digest", "sglang:engine"]],
                "workers": [slot + "-a", slot + "-b"],
                "all_workers": [slot + "-a", slot + "-b", slot + "-dead"],
            }

    def snapshot(self, port):
        return copy.deepcopy(self.profiles[port])

    def canary(self, port, payload, expected):
        if self.fail_canary:
            raise RolloutError("canary failed")
        return {"request_id": "canary"}

    def leases(self, port, owners):
        return [lease for lease in self.pending_leases if lease["owner"] in owners]


@pytest.fixture
def controller(tmp_path):
    # macOS pytest paths can exceed the AF_UNIX bound; inject a short owned path.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="jev-roll-", dir="/tmp") as root:
        instance = Rollout(Path(root) / "proxy", Probe(), Proxy())
        instance.initialize(19001, 19002, 19000)
        yield instance


def change(controller, **kwargs):
    args = dict(
        target="green",
        expected_generation=0,
        operation_id="op-1",
        deployment_id="green-1",
        release_id="release-1",
        canary={"fixture": True},
    )
    return controller.switch(**(args | kwargs))


@pytest.mark.parametrize(
    "point,observed", [("intent", "blue"), ("runtime", "green"), ("disk", "green")]
)
def test_reconcile_crash_windows_without_replay(controller, point, observed):
    def crash(stage):
        if stage == point:
            raise RuntimeError("simulated controller crash")

    with pytest.raises(RuntimeError, match="simulated"):
        change(controller, checkpoint=crash)
    with pytest.raises(RolloutError, match="Pending"):
        change(controller)
    calls = controller.proxy.calls
    receipt = controller.reconcile()
    assert receipt["observed"] == observed and receipt["generation"] == 1
    assert controller.proxy.calls == calls
    assert (controller.directory / "active.map").read_text() == f"active {observed}\n"
    assert controller.status()["state"]["pending"] is None
    assert change(controller) == receipt
    assert controller.proxy.calls == calls


def test_lost_runtime_ack_requires_observation(controller):
    controller.proxy.unknown_reply = True
    with pytest.raises(TimeoutError):
        change(controller)
    assert controller.reconcile()["observed"] == "green"
    assert controller.proxy.calls == 1


def test_stale_cas_id_collision_and_old_receipt_do_not_switch(controller):
    first = change(controller)
    with pytest.raises(RolloutError, match="reused"):
        change(controller, release_id="different")
    with pytest.raises(RolloutError, match="generation"):
        change(controller, target="blue", operation_id="stale")
    second = change(
        controller,
        target="blue",
        expected_generation=1,
        operation_id="back",
        deployment_id="blue-1",
    )
    assert second["generation"] == 2 and controller.proxy.active() == "blue"
    assert change(controller) == first
    assert controller.proxy.active() == "blue" and controller.proxy.calls == 2
    with pytest.raises(RolloutError, match="latest"):
        controller.drain("op-1", timeout=0.001)


@pytest.mark.parametrize(
    "mutation", ["registry", "model", "backend", "routes", "incarnation", "canary", "auth_policy"]
)
def test_incompatible_candidate_never_changes_traffic(controller, mutation):
    candidate = controller.probe.profiles[19002]
    if mutation == "registry":
        candidate["deployment"]["registry"]["inode"] = 3
    elif mutation == "incarnation":
        candidate["deployment"]["id"] = "green-unexpected"
    elif mutation == "canary":
        controller.probe.fail_canary = True
    else:
        candidate[mutation] = "different"
    with pytest.raises(RolloutError):
        change(controller)
    assert controller.proxy.active() == "blue" and controller.proxy.calls == 0
    assert controller.status()["state"]["pending"] is None


def test_drain_preserves_old_work_and_dead_owner_leases(controller):
    change(controller)
    controller.proxy.sessions = 1
    assert not controller.drain("op-1", timeout=0.001)["drained"]
    controller.proxy.sessions = 0
    controller.probe.pending_leases = [{"owner": "blue-dead"}]
    assert not controller.drain("op-1", timeout=0.001)["drained"]
    controller.probe.pending_leases = []
    assert controller.drain("op-1", timeout=0.001)["drained"]


def test_outside_map_changes_fail_closed(controller):
    controller.proxy.selected = "green"
    with pytest.raises(RolloutError, match="Runtime map differs"):
        change(controller)


def test_no_implicit_reinitialization_or_config_rewrite(controller):
    with pytest.raises(FileExistsError):
        controller.initialize(19001, 19002, 19000)
    (controller.directory / "haproxy.cfg").write_text("external replacement")
    with pytest.raises(RolloutError, match="configuration changed"):
        controller.status()


def test_competing_controller_cannot_mutate(controller):
    with controller.locked(), pytest.raises(RolloutError, match="Another rollout"):
        change(controller)


async def test_deployment_profile_lists_starting_and_missing_workers(runtime):
    registry = runtime.registry
    assert registry.deployment_profile() is None
    registry.tag_deployment("green-1", "source-abc", 2)
    profile = registry.deployment_profile()
    assert profile["expected_workers"] == 2
    assert len(profile["workers"]) == 1
    assert profile["workers"][0]["id"] == registry.owner
    assert profile["workers"][0]["state"] == "SERVING"
    assert json.loads(json.dumps(profile)) == profile


@pytest.mark.parametrize("failure", [None, "unseen", "unknown", "mixed", "incomplete", "unhealthy"])
def test_gateway_probe_requires_every_worker(monkeypatch, failure):
    template = Probe().profiles[19002]
    group = [
        {
            "id": worker,
            "release": "release-1",
            "expected_workers": 2,
            "state": "SERVING",
            "backend": template["backend"],
            "owner_status": "alive",
        }
        for worker in template["workers"]
    ]
    if failure == "unknown":
        group[1]["owner_status"] = "unknown"
    if failure == "mixed":
        group[1]["release"] = "other"
    if failure == "incomplete":
        group.pop()
    counter = 0
    original_client = httpx.Client

    def factory(**kwargs):
        nonlocal counter
        index = 0 if failure == "unseen" else counter % 2
        worker = template["workers"][index]
        counter += 1

        def handler(request):
            if request.url.path == "/ready":
                data = {"ready": True, "worker_id": worker}
            elif request.url.path == "/admin/profile":
                data = {
                    "worker_id": worker,
                    "control_healthy": True,
                    "deployment": template["deployment"] | {"workers": group},
                    "model": template["model"],
                    "capabilities": template["capabilities"],
                    "auth_policy": template["auth_policy"],
                    "health": {
                        "monitor_running": True,
                        "bundles": {
                            "default@1": {
                                "ready": failure != "unhealthy" or index == 0,
                                "prepared": True,
                            }
                        },
                    },
                }
            else:
                data = {
                    "routes": template["routes"],
                    "bundles": [
                        {"ref": row[0], "digest": row[1], "backend": row[2]}
                        for row in template["bindings"]
                    ],
                }
            return httpx.Response(200, json=data)

        return original_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", factory)
    probe = GatewayProbe("api", "admin", timeout=0.06)
    if failure:
        with pytest.raises(RolloutError):
            probe.snapshot(19002)
    else:
        assert probe.snapshot(19002)["workers"] == template["workers"]
        assert counter == 2
