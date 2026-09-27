from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jev_runtime.schema import Contract


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
    verified: bool = False


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
