"""Single-host gateway rollout over an explicitly owned HAProxy Runtime API.

The data plane never calls this controller. A pending operation stops further
mutations until an operator reconciles the observed map; it is never replayed.
Both slots must share one registry and one native engine. This is not an engine
failover or a registry migration protocol.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import socket
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import httpx

from jev_runtime.schema import DecisionRequest, DecisionResponse


class RolloutError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RolloutError(message)


def atomic_write(path: Path, data: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as file:
            os.chmod(temporary, 0o600)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2) + "\n")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class HAProxy:
    def __init__(self, directory: Path):
        self.directory = directory

    def command(self, command: str) -> str:
        require("\n" not in command and ";" not in command, "Invalid Runtime API command")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            connection.connect(str(self.directory / "control.sock"))
            connection.sendall((command + "\n").encode())
            connection.shutdown(socket.SHUT_WR)
            chunks, size = [], 0
            while data := connection.recv(65536):
                chunks.append(data)
                size += len(data)
                require(size < 16_777_216, "Runtime API response exceeds the bound")
        return b"".join(chunks).decode().strip()

    def active(self) -> str:
        lines = self.command(f"show map {self.directory}/active.map").splitlines()
        entries = [line.split() for line in lines if line and not line.startswith("#")]
        require(
            len(entries) == 1
            and len(entries[0]) == 3
            and entries[0][1] == "active"
            and entries[0][2] in {"blue", "green"},
            "Runtime map must have exactly one known active entry",
        )
        return entries[0][2]

    def select(self, slot: str) -> None:
        require(slot in {"blue", "green"}, "Unknown slot")
        reply = self.command(f"set map {self.directory}/active.map active {slot}")
        require(not reply, f"Runtime API rejected selection: {reply[:200]}")
        require(self.active() == slot, "Runtime API selection could not be confirmed")

    def identity(self) -> dict:
        info = dict(
            line.split(": ", 1) for line in self.command("show info").splitlines() if ": " in line
        )
        require(info.get("Nbthread") == "1", "Rollout drain requires one HAProxy event loop")
        require(info.get("Version", "").startswith("3.2.24"), "Use qualified HAProxy 3.2.24")
        return {key: info[key] for key in ("Pid", "Version", "Nbthread")}

    def outstanding(self, slot: str) -> dict:
        rows = list(csv.DictReader(io.StringIO(self.command("show stat").removeprefix("# "))))
        selected = [row for row in rows if row["pxname"] == slot]
        require(
            {row["svname"] for row in selected} == {"gateway", "BACKEND"},
            "Unexpected proxy topology; cannot certify drain",
        )
        return {
            row["svname"]: {field: int(row[field]) for field in ("scur", "qcur")}
            for row in selected
        }


def render_configuration(directory: Path, ports: dict, frontend_port: int) -> str:
    # One event loop serializes map selection and backend stream accounting with
    # Runtime API operations. Do not add yielding rules before backend assignment.
    return f"""global
    nbthread 1
    maxconn 8192
    stats socket {directory}/control.sock mode 600 level admin
    stats timeout 5s
defaults
    mode http
    retries 0
    timeout connect 2s
    timeout queue 5s
    timeout client 310s
    timeout server 310s
    timeout http-request 30s
    timeout http-keep-alive 30s
    option http-keep-alive
frontend jev
    bind 127.0.0.1:{frontend_port}
    use_backend %[str(active),map_str({directory}/active.map)]
    default_backend invalid_selection
    http-after-response set-header X-Jev-Deployment %[be_name]
backend invalid_selection
    http-request deny deny_status 503
backend blue
    http-reuse safe
    server gateway 127.0.0.1:{ports["blue"]}
backend green
    http-reuse safe
    server gateway 127.0.0.1:{ports["green"]}
"""


class GatewayProbe:
    def __init__(self, api_key: str, admin_key: str, timeout: float = 30):
        require(bool(api_key and admin_key), "Both API and admin keys are required")
        self.api_key, self.admin_key, self.timeout = api_key, admin_key, timeout

    def snapshot(self, port: int) -> dict:
        """Observe every advertised worker; a sampled worker is not a replica barrier."""
        deadline, seen, common, roster = time.monotonic() + self.timeout, {}, None, None
        while time.monotonic() < deadline:
            # One HTTP/1 connection stays on one uvicorn worker. A new connection
            # each iteration lets us discover the rest; starvation fails closed.
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=min(self.timeout, 10),
                trust_env=False,
            ) as client:
                ready = client.get("/ready", headers={"Authorization": f"Bearer {self.api_key}"})
                ready.raise_for_status()
                response = client.get(
                    "/admin/profile", headers={"Authorization": f"Bearer {self.admin_key}"}
                )
                response.raise_for_status()
                profile = response.json()
                require(
                    profile["worker_id"] == ready.json()["worker_id"], "Worker changed mid-probe"
                )
                require(profile.get("control_healthy") is True, "Cancellation control is unhealthy")
                deployment = profile.get("deployment")
                require(deployment and deployment.get("protocol") == 1, "Missing rollout metadata")
                workers = [row for row in deployment["workers"] if row["owner_status"] != "dead"]
                require(len(workers) == deployment["expected_workers"], "Incomplete worker group")
                require(
                    all(
                        row["owner_status"] == "alive"
                        and row["state"] == "SERVING"
                        and row["release"] == deployment["release"]
                        and row["expected_workers"] == deployment["expected_workers"]
                        for row in workers
                    ),
                    "Worker group contains unverified, unavailable, or mixed-release processes",
                )
                observed = {row["id"] for row in workers}
                require(profile["worker_id"] in observed, "Responder not in deployment roster")
                if roster is None:
                    roster = observed
                require(roster == observed, "Worker roster changed during qualification")
                health = profile["health"]
                require(
                    health["monitor_running"]
                    and health["bundles"]
                    and all(row["ready"] and row["prepared"] for row in health["bundles"].values()),
                    "Worker lacks current bundle health evidence",
                )
                response = client.get(
                    "/admin/bundles", headers={"Authorization": f"Bearer {self.admin_key}"}
                )
                response.raise_for_status()
                inventory = response.json()
                routes = inventory["routes"]
                refs = {row["ref"] for row in routes if row["ref"]}
                require(refs == set(health["bundles"]), "Health evidence and current routes differ")
                bindings = sorted(
                    [row["ref"], row["digest"], row["backend"]]
                    for row in inventory["bundles"]
                    if row["ref"] in refs
                )
                require(len(bindings) == len(refs), "Missing active bundle binding")
                require(len({row["backend"] for row in workers}) == 1, "Mixed engine group")
                current = {
                    "deployment": {
                        key: value for key, value in deployment.items() if key != "workers"
                    },
                    "backend": workers[0]["backend"],
                    "auth_policy": profile.get("auth_policy"),
                    "model": profile["model"],
                    "capabilities": profile["capabilities"],
                    "routes": routes,
                    "bindings": bindings,
                }
                require(current["model"] is not None, "Configured model identity required")
                require(current["auth_policy"] is not None, "Authenticated gateway required")
                if common is None:
                    common = current
                require(common == current, "Worker identity or routes changed during qualification")
                seen[profile["worker_id"]] = True
                if set(seen) == roster:
                    return {
                        **common,
                        "workers": sorted(roster),
                        "all_workers": sorted(row["id"] for row in deployment["workers"]),
                    }
            time.sleep(0.01)
        raise RolloutError("Did not observe every worker before qualification timeout")

    def canary(self, port: int, payload: dict, expected: dict) -> dict:
        request = DecisionRequest.model_validate(payload).model_copy(
            update={"request_id": "rollout-" + uuid.uuid4().hex}
        )
        with httpx.Client(trust_env=False, timeout=self.timeout) as client:
            result = client.post(
                f"http://127.0.0.1:{port}/v1/decisions",
                json=request.model_dump(mode="json"),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        result.raise_for_status()
        response = DecisionResponse.model_validate(result.json())
        route = next((row for row in expected["routes"] if row["alias"] == request.model), None)
        require(
            route
            and response.bundle == route["ref"]
            and response.generation == route["generation"],
            "Canary used a different route generation",
        )
        require(
            response.status == "completed"
            and set(response.answers) == {q.id for q in request.questions},
            "Canary did not complete every question",
        )
        require(result.headers.get("X-Jev-Worker") in expected["workers"], "Canary worker differs")
        binding = next(row for row in expected["bindings"] if row[0] == response.bundle)
        require(response.bundle_digest == binding[1], "Canary bundle digest differs")
        return {
            "request_id": response.request_id,
            "bundle": response.bundle,
            "generation": response.generation,
            "worker": result.headers["X-Jev-Worker"],
        }

    def leases(self, port: int, owners: list[str]) -> list:
        with httpx.Client(trust_env=False, timeout=10) as client:
            result = client.get(
                f"http://127.0.0.1:{port}/admin/bundles",
                headers={"Authorization": f"Bearer {self.admin_key}"},
            )
        result.raise_for_status()
        return [row for row in result.json()["leases"] if row["owner"] in owners]


def compatible(first: dict, second: dict) -> None:
    for field in ("backend", "model", "capabilities", "routes", "bindings", "auth_policy"):
        require(first[field] == second[field], f"Candidate differs in {field}")
    require(
        first["deployment"]["registry"] == second["deployment"]["registry"],
        "Both slots must share one local registry inode",
    )
    require(
        first["deployment"]["id"] != second["deployment"]["id"],
        "Slot incarnations must have distinct deployment IDs",
    )


class Rollout:
    def __init__(self, directory: Path, probe: GatewayProbe, proxy=None):
        self.directory = directory.resolve()
        require(
            re.fullmatch(r"/[A-Za-z0-9_./-]+", str(self.directory)), "Unsafe proxy directory path"
        )
        require(
            len(str(self.directory / "control.sock").encode()) < 104, "Unix socket path too long"
        )
        self.probe, self.proxy = probe, proxy or HAProxy(self.directory)

    @contextmanager
    def locked(self):
        with (self.directory / "controller.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RolloutError("Another rollout controller owns the deployment") from exc
            yield

    def initialize(
        self, blue_port: int, green_port: int, frontend_port: int, active="blue"
    ) -> dict:
        require(active in {"blue", "green"}, "Unknown slot")
        ports = {"blue": blue_port, "green": green_port}
        require(
            len({blue_port, green_port, frontend_port}) == 3
            and all(1024 <= port <= 65535 for port in (blue_port, green_port, frontend_port)),
            "Ports must be distinct unprivileged TCP ports",
        )
        # Qualification precedes creation. A partial directory after a filesystem
        # failure cannot be adopted silently by a second initialize.
        snapshot = self.probe.snapshot(ports[active])
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        config = render_configuration(self.directory, ports, frontend_port)
        atomic_write(self.directory / "haproxy.cfg", config)
        atomic_write(self.directory / "active.map", f"active {active}\n")
        state = {
            "format": 1,
            "generation": 0,
            "active": active,
            "ports": ports,
            "frontend_port": frontend_port,
            "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
            "current": snapshot,
            "pending": None,
            "operations": {},
        }
        write_json(self.directory / "state.json", state)
        return state

    def _load(self) -> dict:
        state = json.loads((self.directory / "state.json").read_text())
        require(state["format"] == 1, "Unknown rollout state format")
        require(
            hashlib.sha256((self.directory / "haproxy.cfg").read_bytes()).hexdigest()
            == state["config_sha256"],
            "Proxy configuration changed outside the controller",
        )
        return state

    def _consistent(self, state):
        require(state["pending"] is None, "Pending operation requires explicit reconcile")
        require(
            self.proxy.active() == state["active"], "Runtime map differs; do not guess ownership"
        )
        require(
            (self.directory / "active.map").read_text() == f"active {state['active']}\n",
            "Disk map differs from committed state",
        )
        self.proxy.identity()

    def status(self) -> dict:
        with self.locked():
            state = self._load()
            return {
                "state": state,
                "runtime_active": self.proxy.active(),
                "proxy": self.proxy.identity(),
            }

    def switch(
        self,
        target: str,
        expected_generation: int,
        operation_id: str,
        deployment_id: str,
        release_id: str,
        canary: dict,
        checkpoint: Callable[[str], None] = lambda _: None,
    ) -> dict:
        require(target in {"blue", "green"}, "Unknown slot")
        require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", operation_id), "Invalid operation ID")
        intent = {
            "target": target,
            "expected_generation": expected_generation,
            "deployment_id": deployment_id,
            "release_id": release_id,
            "canary_digest": digest(canary),
        }
        with self.locked():
            state = self._load()
            if operation_id in state["operations"]:
                receipt = state["operations"][operation_id]
                require(receipt["intent"] == intent, "Operation ID reused with different input")
                return receipt  # A receipt is historical; never repeat a completed switch.
            self._consistent(state)
            require(state["generation"] == expected_generation, "Rollout generation conflict")
            require(state["active"] != target, "Target is already selected")
            old = self.probe.snapshot(state["ports"][state["active"]])
            require(
                old["deployment"] == state["current"]["deployment"],
                "Active incarnation changed outside the rollout controller",
            )
            candidate = self.probe.snapshot(state["ports"][target])
            require(
                candidate["deployment"]["id"] == deployment_id
                and candidate["deployment"]["release"] == release_id,
                "Candidate deployment or release differs from requested incarnation",
            )
            compatible(old, candidate)
            evidence = self.probe.canary(state["ports"][target], canary, candidate)
            # Recheck both complete worker groups after the real canary.
            require(
                self.probe.snapshot(state["ports"][target]) == candidate,
                "Candidate changed during canary",
            )
            require(
                self.probe.snapshot(state["ports"][state["active"]]) == old,
                "Active worker group or routes changed during canary",
            )
            state["pending"] = {
                "id": operation_id,
                "intent": intent,
                "source": state["active"],
                "target": target,
                "source_snapshot": old,
                "target_snapshot": candidate,
                "canary": evidence,
                "created": time.time(),
                "proxy": self.proxy.identity(),
            }
            write_json(self.directory / "state.json", state)
            checkpoint("intent")
            self.proxy.select(target)
            checkpoint("runtime")
            atomic_write(self.directory / "active.map", f"active {target}\n")
            checkpoint("disk")
            return self._commit(state, target, "switched")

    def _commit(self, state: dict, observed: str, outcome: str) -> dict:
        pending = state["pending"]
        receipt = {
            **pending,
            "observed": observed,
            "outcome": outcome,
            "generation": state["generation"] + 1,
            "completed": time.time(),
        }
        state.update(
            active=observed,
            generation=receipt["generation"],
            pending=None,
            current=pending[f"{'target' if observed == pending['target'] else 'source'}_snapshot"],
        )
        state["operations"][pending["id"]] = receipt
        write_json(self.directory / "state.json", state)
        return receipt

    def reconcile(self) -> dict:
        """Adopt the observed source/target only, without retrying a mutation."""
        with self.locked():
            state = self._load()
            pending = state["pending"]
            require(pending is not None, "There is no pending operation")
            self.proxy.identity()
            observed = self.proxy.active()
            require(
                observed in {pending["source"], pending["target"]}, "Unexpected runtime selection"
            )
            snapshot = self.probe.snapshot(state["ports"][observed])
            role = "source" if observed == pending["source"] else "target"
            require(
                snapshot == pending[f"{role}_snapshot"],
                "Observed incarnation changed; manual diagnosis required",
            )
            atomic_write(self.directory / "active.map", f"active {observed}\n")
            return self._commit(state, observed, "reconciled_observation")

    def drain(self, operation_id: str, timeout: float = 60) -> dict:
        """Observe inactive HTTP streams and durable work. Never send a signal.

        Callers must prevent direct/new admin traffic to the inactive slot and
        hold their service-manager rollout lock through graceful termination.
        """
        require(0 < timeout <= 3600, "Drain timeout must be in (0, 3600]")
        with self.locked():
            state = self._load()
            self._consistent(state)
            require(operation_id in state["operations"], "Unknown operation")
            receipt = state["operations"][operation_id]
            source = receipt["source"]
            require(
                receipt["generation"] == state["generation"] and state["active"] != source,
                "Drain requires the latest successful switch away from this slot",
            )
            identity = self.proxy.identity()
            deadline = time.monotonic() + timeout
            owners = receipt["source_snapshot"]["all_workers"]
            while True:
                require(
                    self.proxy.identity() == identity and self.proxy.active() == state["active"],
                    "Proxy restarted or selection changed during drain",
                )
                stats = self.proxy.outstanding(source)
                leases = self.probe.leases(state["ports"][state["active"]], owners)
                if not leases and all(
                    value == 0 for row in stats.values() for value in row.values()
                ):
                    # Re-read after the registry RPC; no earlier stream may appear
                    # after this single-event-loop barrier on the owned proxy.
                    stats = self.proxy.outstanding(source)
                    if all(value == 0 for row in stats.values() for value in row.values()):
                        return {
                            "drained": True,
                            "slot": source,
                            "workers": owners,
                            "generation": state["generation"],
                            "proxy": identity,
                            "stats": stats,
                            "remaining_leases": 0,
                        }
                if time.monotonic() >= deadline:
                    return {
                        "drained": False,
                        "slot": source,
                        "stats": stats,
                        "remaining_leases": len(leases),
                        "generation": state["generation"],
                    }
                time.sleep(0.05)
