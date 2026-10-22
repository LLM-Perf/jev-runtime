"""Explicit fence for owners this host can no longer prove dead.

A changed boot ID, PID namespace or hostname makes the old owner ``unknown``.
Unknown is not dead: recovery, restore and admission keep failing closed until
an operator attests that the previous deployment is stopped. The fence records
that attestation and the exact owner identities. It does not delete leases,
journals or receipts, and it refuses while any owner is verifiably alive.

Processes that registered under an older fence generation fail their next
registry write and readiness check. A copied database cannot stop a process
that does not share this file or does not run this runtime version.
"""

from __future__ import annotations

import hashlib
import json
import time

from jev_runtime.registry_backup import connect

FENCE_ACK = "previous-deployment-stopped"
REASONS = ("lost_host", "container_namespace", "unverifiable_identity")


def gone(status: str) -> bool:
    """True when the owner cannot still be admitted or block maintenance."""
    return status in {"dead", "fenced"}


def observation() -> dict:
    from jev_runtime.registry import Registry

    identity = Registry._process_identity()
    if not identity.get("boot_id") or not identity.get("pid_namespace"):
        raise ValueError("Fence requires a readable Linux boot ID and PID namespace")
    return {
        "host": identity["host"],
        "boot_id": identity["boot_id"],
        "pid_namespace": identity["pid_namespace"],
    }


def identity_digest(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fences(db) -> list[dict]:
    rows = db.execute(
        "SELECT details FROM events WHERE action='deployment_fence' ORDER BY id"
    ).fetchall()
    parsed = []
    for (raw,) in rows:
        value = json.loads(raw)
        if not isinstance(value, dict) or type(value.get("generation")) is not int:
            raise ValueError("Deployment fence event is corrupt")
        parsed.append(value)
    return parsed


def current_fence_generation(db) -> int:
    return max((fence["generation"] for fence in _fences(db)), default=0)


def fenced_owner(db, identity: dict, observed: dict) -> bool:
    digest = identity_digest(identity)
    return any(
        fence.get("observation") == observed and digest in fence.get("owner_digests", [])
        for fence in _fences(db)
    )


def unknown_kind(identity: dict, observed: dict) -> str:
    if (
        not identity.get("boot_id")
        or not identity.get("pid_namespace")
        or not identity.get("start_ticks")
    ):
        return "unverifiable_identity"
    if identity.get("host") != observed["host"] or identity.get("boot_id") != observed["boot_id"]:
        return "lost_host"
    if identity.get("pid_namespace") != observed["pid_namespace"]:
        return "container_namespace"
    return "unverifiable_identity"


def maintenance_owner_blockers(db) -> dict[str, int]:
    from jev_runtime.registry import Registry

    try:
        observed = observation()
    except ValueError:
        observed = None
    result: dict[str, int] = {}
    for (raw,) in db.execute("SELECT identity FROM owners"):
        identity = json.loads(raw)
        state = Registry._owner_status(identity)
        if state == "unknown" and observed is not None and fenced_owner(db, identity, observed):
            state = "fenced"
        if not gone(state):
            key = "owners_" + state
            result[key] = result.get(key, 0) + 1
    return result


def fence_stale_owners(source, reason: str, acknowledgement: str) -> dict:
    if acknowledgement != FENCE_ACK:
        raise ValueError(f"Fence requires acknowledgement {FENCE_ACK}")
    if reason not in REASONS:
        raise ValueError(
            "Fence reason must be lost_host, container_namespace or unverifiable_identity"
        )
    observed = observation()
    from jev_runtime.registry import Registry
    from jev_runtime.registry_schema import identify

    db = connect(source, writable=True)
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            identify(db)
            rows = db.execute("SELECT owner, identity FROM owners").fetchall()
            alive = []
            candidates = []
            covered = []
            for owner, raw in rows:
                identity = json.loads(raw)
                status = Registry._owner_status(identity)
                if status == "alive":
                    alive.append(owner)
                elif status == "unknown" and fenced_owner(db, identity, observed):
                    covered.append(owner)
                elif status == "unknown":
                    candidates.append((owner, identity, unknown_kind(identity, observed)))
            if alive:
                raise ValueError("Refusing to fence while owners are alive: " + ",".join(alive))
            if not candidates:
                if not covered:
                    raise ValueError("No unknown owners to fence")
                recorded = _fences(db)[-1]
                if recorded.get("reason") != reason:
                    raise ValueError("Existing fence reason is " + str(recorded.get("reason")))
                result = {
                    "generation": recorded["generation"],
                    "reason": recorded.get("reason"),
                    "fenced_owners": len(covered),
                    "idempotent": True,
                    "observation": observed,
                }
                db.rollback()
                return result
            mismatched = sorted({kind for _, _, kind in candidates if kind != reason})
            if mismatched:
                raise ValueError(
                    "Unknown owners do not match " + reason + ": " + ",".join(mismatched)
                )
            generation = current_fence_generation(db) + 1
            digests = [identity_digest(identity) for _, identity, _ in candidates]
            event = {
                "generation": generation,
                "reason": reason,
                "acknowledgement": acknowledgement,
                "observation": observed,
                "owner_digests": sorted(digests),
            }
            db.execute(
                "INSERT INTO events(ts,action,details) VALUES(?,?,?)",
                (time.time(), "deployment_fence", json.dumps(event, sort_keys=True)),
            )
            result = {
                "generation": generation,
                "reason": reason,
                "fenced_owners": len(candidates),
                "idempotent": False,
                "observation": observed,
            }
            db.commit()
            return result
        except BaseException:
            db.rollback()
            raise
    finally:
        db.close()
