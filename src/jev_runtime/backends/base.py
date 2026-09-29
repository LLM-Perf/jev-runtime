from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, StrictBool

from jev_runtime.schema import Contract, FloatingDType


def reported_dtype(value) -> str | None:
    """Normalize an observed engine dtype without guessing `auto` or quantized types."""
    name = str(value).removeprefix("torch.")
    return name if name in {"float16", "bfloat16", "float32"} else None


class Capabilities(Contract):
    engine: str
    version: str
    model_id: str
    max_label_tokens: int = 128
    max_context_tokens: int = 32768
    selected_logprobs: bool = True
    raw_logprobs: bool = True
    cancellation: bool = True
    lora: bool = False
    prefix_cache: bool | None = None
    api_workers: int | None = Field(default=None, ge=1)
    verified: bool = False
    model_dtype: FloatingDType | None = None
    readout_dtype: FloatingDType | None = None
    # A single-label backend must expose every native request to admission,
    # journaling and usage accounting; it must not hide K requests in score().
    label_scoring: Literal["joint", "single"] = "joint"
    # Observed engine configuration, not a numerical/kernel certificate.
    batch_invariant: StrictBool | None = None


@dataclass(frozen=True)
class ScoreInput:
    request_id: str
    question_id: str
    input_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    candidate_id: str | None = None
    adapter_id: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    request_id: str
    logprobs: tuple[float, ...]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    raw_logprobs: bool = True


class EngineAdapter(Protocol):
    async def probe(self) -> Capabilities: ...
    async def score(self, request: ScoreInput) -> ScoreResult: ...
    async def cancel(self, request_id: str) -> None: ...
    async def close(self) -> None: ...
