from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from pydantic import Field, model_validator

from jev_runtime.errors import JevError
from jev_runtime.schema import Contract


class HealthSettings(Contract):
    interval_seconds: float = Field(default=30, ge=1, le=300)
    timeout_seconds: float = Field(default=10, ge=0.1, le=120)
    max_age_seconds: float = Field(default=90, ge=1, le=900)

    @model_validator(mode="after")
    def allow_probe_and_abort(self):
        if self.max_age_seconds < self.interval_seconds + self.timeout_seconds + 5:
            raise ValueError("Health max age must allow the interval, probe and five-second abort")
        return self


@dataclass
class Observation:
    generation: int = 0
    last_success: float | None = None
    error: str | None = None


class ServingHealth:
    """Per-worker canary evidence; never substitutes for durable dispatch leases."""

    def __init__(self, settings: HealthSettings):
        self.settings = settings
        self.observations: dict[str, Observation] = {}

    def generation(self, reference: str) -> int:
        return self.observations.setdefault(reference, Observation()).generation

    def passed(self, reference: str, generation: int) -> None:
        observation = self.observations.setdefault(reference, Observation())
        # A probe started before a new serving error cannot close that circuit.
        if observation.generation == generation:
            observation.last_success = monotonic()
            observation.error = None

    def failed(self, reference: str, error: str) -> None:
        observation = self.observations.setdefault(reference, Observation())
        observation.generation += 1
        observation.error = error

    def status(self, reference: str) -> dict:
        observation = self.observations.get(reference, Observation())
        age = (
            None
            if observation.last_success is None
            else max(0, monotonic() - observation.last_success)
        )
        error = observation.error
        if not error and (age is None or age > self.settings.max_age_seconds):
            error = "health_stale"
        return {"ready": error is None, "last_success_age_seconds": age, "error": error}

    def require(self, reference: str) -> None:
        if not self.status(reference)["ready"]:
            raise JevError(
                "engine_unavailable",
                "Bundle has no current successful engine canary; wait for recovery",
                503,
            )
