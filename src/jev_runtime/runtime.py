from __future__ import annotations

import asyncio
import logging
import time
import uuid

from jev_runtime.admission import Admission
from jev_runtime.backends.base import Capabilities, EngineAdapter, ScoreInput, ScoreResult
from jev_runtime.compiler import CompiledQuestion, Compiler
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.schema import (
    Answer,
    Bundle,
    DecisionRequest,
    DecisionResponse,
    ModelIdentity,
    Question,
    Usage,
)
from jev_runtime.scoring import assemble

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(
        self,
        backend: EngineAdapter,
        compiler: Compiler,
        registry: Registry,
        backend_identity: str,
        model_id: str,
        admission: Admission | None = None,
        expected_model: ModelIdentity | None = None,
    ):
        self.backend, self.compiler, self.registry = backend, compiler, registry
        self.backend_identity, self.model_id = backend_identity, model_id
        self.expected_model = expected_model
        self.admission = admission or Admission()
        self.capabilities: Capabilities | None = None
        self._management_lock = asyncio.Lock()
        self._active: dict[str, asyncio.Task] = {}
        self._tenants: dict[str, str] = {}
        self._unconfirmed_leases: dict[str, tuple[str, set[str]]] = {}

    async def start(self) -> None:
        self.capabilities = await self.backend.probe()
        if not self.capabilities.selected_logprobs or not self.capabilities.raw_logprobs:
            raise JevError(
                "unsupported_engine", "Engine must return complete raw selected logprobs", 503
            )

    async def close(self) -> None:
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.backend.close()

    def _validate_bundle(self, bundle: Bundle) -> None:
        self.compiler.verify_bundle(bundle)
        if bundle.model.id != self.model_id:
            raise JevError("model_mismatch", "Bundle is bound to a different model", 409)
        if self.expected_model is not None:
            for field in (
                "revision",
                "dtype",
                "quantization",
                "tokenizer_digest",
                "template_digest",
            ):
                if getattr(bundle.model, field) != getattr(self.expected_model, field):
                    raise JevError(
                        "model_mismatch", f"Bundle model {field} differs from loaded profile", 409
                    )
        if bundle.model.adapter_id and (not self.capabilities or not self.capabilities.lora):
            raise JevError("lora_unsupported", "Backend has not enabled adapter scoring", 409)

    async def prepare(self, reference: str) -> dict:
        async with self._management_lock:
            bundle = self.registry.begin_prepare(reference, self.backend_identity)
            request_id = "prepare-" + uuid.uuid4().hex
            lease_id = self.registry.pin_preparation(reference, request_id, self.backend_identity)
            self._active[request_id] = asyncio.current_task()
            unconfirmed: set[str] = set()
            try:
                self._validate_bundle(bundle)
                questions = bundle.questions or (
                    Question(
                        id="canary",
                        type="boolean",
                        instruction="Does the data contain the word ready?",
                    ),
                )
                async with asyncio.timeout(120):
                    for question in questions:
                        compiled = self.compiler.compile("ready", question, bundle, request_id)
                        self._validate_sequences(compiled.sequences, bundle)
                        results = [
                            await self._score(seq, unconfirmed) for seq in compiled.sequences
                        ]
                        assemble(compiled, results, bundle)
            except BaseException as exc:
                message = exc.message if isinstance(exc, JevError) else type(exc).__name__
                self.registry.finish_prepare(reference, message)
                raise
            finally:
                if unconfirmed:
                    self._unconfirmed_leases[request_id] = (lease_id, unconfirmed)
                else:
                    self.registry.release(lease_id)
                self._active.pop(request_id, None)
            self.registry.finish_prepare(reference)
            return self.registry.inspect(reference)

    def _validate_sequences(self, sequences: tuple[ScoreInput, ...], bundle: Bundle) -> int:
        if self.capabilities is None:
            raise JevError("not_started", "Engine capabilities have not been probed", 503)
        if len(sequences) > bundle.policy.max_scoring_sequences:
            raise JevError("branch_budget", "Request expands to too many scoring sequences", 413)
        total = 0
        for seq in sequences:
            if len(seq.label_ids) > self.capabilities.max_label_tokens:
                raise JevError("engine_label_limit", "Engine cannot return this many labels", 413)
            if len(seq.input_ids) + 1 > self.capabilities.max_context_tokens:
                raise JevError("engine_context_limit", "Request exceeds engine context limit", 413)
            total += len(seq.input_ids)
        if total > bundle.policy.max_expanded_tokens:
            raise JevError(
                "expanded_token_budget", "Expanded prompts exceed task token budget", 413
            )
        return total

    async def _score(self, seq: ScoreInput, unconfirmed: set[str]) -> ScoreResult:
        try:
            return await self.backend.score(seq)
        except BaseException:
            try:
                async with asyncio.timeout(5):
                    await self.backend.cancel(seq.request_id)
            except BaseException:
                unconfirmed.add(seq.request_id)
                logger.exception("Engine cancellation could not be confirmed")
            raise

    async def _question(
        self,
        compiled: CompiledQuestion,
        bundle: Bundle,
        semaphore: asyncio.Semaphore,
        unconfirmed: set[str],
    ) -> tuple[Answer, list[ScoreResult]]:
        async def score(seq: ScoreInput) -> ScoreResult:
            async with semaphore:
                return await self._score(seq, unconfirmed)

        tasks = [asyncio.create_task(score(seq)) for seq in compiled.sequences]
        try:
            results = await asyncio.gather(*tasks)
            return assemble(compiled, results, bundle), results
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def decide(
        self, request: DecisionRequest, request_id: str | None = None, tenant: str = "default"
    ) -> DecisionResponse:
        rid = request_id or request.request_id or "dec-" + uuid.uuid4().hex
        if rid in self._active:
            raise JevError("duplicate_request", "Request ID is already in flight", 409)
        current = asyncio.current_task()
        self._active[rid] = current
        self._tenants[rid] = tenant
        started = time.monotonic()
        snapshot = None
        unconfirmed: set[str] = set()
        try:
            async with asyncio.timeout(request.execution.timeout_ms / 1000):
                snapshot = self.registry.acquire(
                    request.model, rid, request.bundle, self.backend_identity
                )
                bundle = snapshot.bundle
                self._validate_bundle(bundle)
                questions = request.questions or bundle.questions
                if bundle.candidate_policy == "fixed" and questions != bundle.questions:
                    raise JevError(
                        "task_mismatch", "Fixed task bundle does not allow question changes", 409
                    )
                if not questions or len(questions) > bundle.policy.max_questions:
                    raise JevError(
                        "question_budget",
                        "Request must contain an allowed number of questions",
                        413,
                    )
                compiled = [
                    self.compiler.compile(request.input.text, q, bundle, rid) for q in questions
                ]
                sequences = tuple(s for q in compiled for s in q.sequences)
                total_tokens = self._validate_sequences(sequences, bundle)
                async with self.admission.acquire(total_tokens, tenant):
                    semaphore = asyncio.Semaphore(bundle.policy.max_parallel_branches)
                    tasks = [
                        asyncio.create_task(self._question(q, bundle, semaphore, unconfirmed))
                        for q in compiled
                    ]
                    try:
                        outcomes = await asyncio.gather(
                            *tasks, return_exceptions=request.execution.allow_partial
                        )
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                answers, results = {}, []
                failed = 0
                for question, outcome in zip(questions, outcomes, strict=True):
                    if isinstance(outcome, BaseException):
                        if not isinstance(outcome, Exception):
                            raise outcome
                        error = (
                            outcome
                            if isinstance(outcome, JevError)
                            else JevError("engine_error", "Backend scoring failed", 502)
                        )
                        answers[question.id] = Answer(
                            type=question.type, status="failed", error=error.as_dict()
                        )
                        failed += 1
                    else:
                        answer, scored = outcome
                        answers[question.id] = answer
                        results.extend(scored)

                def total(field: str) -> int | None:
                    values = [getattr(result, field) for result in results]
                    if failed or any(v is None for v in values):
                        return None
                    return sum(values)

                return DecisionResponse(
                    request_id=rid,
                    status="completed"
                    if not failed
                    else ("failed" if failed == len(questions) else "partial"),
                    bundle=bundle.reference,
                    bundle_digest=bundle.digest,
                    generation=snapshot.generation,
                    engine={"name": self.capabilities.engine, "version": self.capabilities.version},
                    answers=answers,
                    usage=Usage(
                        questions=len(questions),
                        successful_questions=len(questions) - failed,
                        scoring_sequences=len(sequences),
                        logical_prompt_tokens=total_tokens,
                        engine_prompt_tokens=total("prompt_tokens"),
                        engine_completion_tokens=total("completion_tokens"),
                        cached_prompt_tokens=total("cached_tokens"),
                    ),
                    latency_ms=(time.monotonic() - started) * 1000,
                )
        except TimeoutError as exc:
            raise JevError("deadline_exceeded", "Decision deadline exceeded", 504) from exc
        except asyncio.CancelledError as exc:
            raise JevError("request_cancelled", "Decision request was cancelled", 499) from exc
        finally:
            if snapshot is not None:
                if unconfirmed:
                    self._unconfirmed_leases[rid] = (snapshot.lease_id, unconfirmed)
                else:
                    self.registry.release(snapshot.lease_id)
            self._active.pop(rid, None)
            self._tenants.pop(rid, None)

    async def recover_cancelled(self, request_id: str) -> bool:
        pending = self._unconfirmed_leases.get(request_id)
        if pending is None:
            return False
        lease_id, branches = pending
        for branch in list(branches):
            async with asyncio.timeout(10):
                await self.backend.cancel(branch)
            branches.remove(branch)
        self.registry.release(lease_id)
        self._unconfirmed_leases.pop(request_id, None)
        return True

    async def cancel(self, request_id: str, tenant: str = "default") -> bool:
        if self._tenants.get(request_id) != tenant:
            return False
        task = self._active.get(request_id)
        if not task:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if request_id in self._unconfirmed_leases:
            raise JevError("cancellation_unconfirmed", "Engine abort is not yet confirmed", 503)
        return True
