from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict

from jev_runtime.admission import Admission
from jev_runtime.backends.base import Capabilities, EngineAdapter, ScoreInput, ScoreResult
from jev_runtime.compiler import CompiledQuestion, Compiler
from jev_runtime.errors import JevError
from jev_runtime.lifecycle import cancel_and_drain
from jev_runtime.registry import Registry
from jev_runtime.schema import (
    Answer,
    Bundle,
    DecisionRequest,
    DecisionResponse,
    ModelIdentity,
    Question,
    Usage,
    content_digest,
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
        self._prepared: set[str] = set()

    async def start(self) -> None:
        self._prepared.clear()
        self.registry.start_worker(self.backend_identity)
        self.capabilities = await self.backend.probe()
        if not self.capabilities.selected_logprobs or not self.capabilities.raw_logprobs:
            raise JevError(
                "unsupported_engine", "Engine must return complete raw selected logprobs", 503
            )
        async with asyncio.timeout(180):
            while True:
                listing = self.registry.list()
                active = {route["ref"] for route in listing["routes"] if route["ref"]}
                for bundle in listing["bundles"]:
                    if bundle["ref"] in active and bundle["backend"] == self.backend_identity:
                        if not self.is_prepared(bundle["ref"]):
                            await self.prepare(bundle["ref"])
                # Atomic against activation: a concurrently published route
                # sends startup back through validation before joining traffic.
                if self.registry.serve_worker(self.backend_identity):
                    break

    def is_prepared(self, reference: str) -> bool:
        return reference in self._prepared

    def activate(self, alias: str, reference: str, expected_generation: int) -> dict:
        if not self.is_prepared(reference):
            raise JevError(
                "replica_not_ready", "Prepare this version on this API worker first", 503
            )
        return self.registry.activate(alias, reference, expected_generation)

    async def close(self) -> None:
        self.registry.stop_worker()
        tasks = list(self._active.values())
        try:
            await cancel_and_drain(tasks)
        finally:
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
            request_id = "prepare-" + uuid.uuid4().hex
            fresh = self.registry.inspect(reference)["state"] in {"VALIDATED", "FAILED", "RETIRED"}
            if fresh:
                bundle, lease_id = self.registry.begin_prepare(
                    reference, self.backend_identity, request_id
                )
            else:
                bundle, lease_id = self.registry.pin_revalidation(
                    reference, request_id, self.backend_identity
                )
            self._active[request_id] = asyncio.current_task()
            unconfirmed: set[str] = set()
            error = None
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
                        self.registry.record_branches(
                            lease_id, [seq.request_id for seq in compiled.sequences]
                        )
                        results = [
                            await self._score(seq, unconfirmed) for seq in compiled.sequences
                        ]
                        assemble(compiled, results, bundle)
            except BaseException as exc:
                error = exc.message if isinstance(exc, JevError) else type(exc).__name__
                self._prepared.discard(reference)
                self.registry.forget_worker_prepared(reference)
                raise
            finally:
                if fresh:
                    self.registry.finish_prepare(reference, lease_id, error, unconfirmed)
                elif unconfirmed:
                    self.registry.mark_abort_pending(lease_id, unconfirmed)
                else:
                    self.registry.release(lease_id)
                self._active.pop(request_id, None)
            self._prepared.add(reference)
            self.registry.record_worker_prepared(reference)
            return {**self.registry.inspect(reference), "worker_id": self.registry.owner}

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

            async def abort():
                async with asyncio.timeout(5):
                    await self.backend.cancel(seq.request_id)

            cleanup = asyncio.create_task(abort())
            try:
                # Parent gather/finally blocks and a second client cancellation
                # can cancel this task again while abort performs network I/O.
                # Own the bounded abort task and wait for its result; shielding
                # without awaiting it to completion would leak detached cleanup.
                while True:
                    try:
                        await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError:
                        if cleanup.cancelled():
                            raise
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
            await cancel_and_drain(tasks)

    def _compile_request(self, request: DecisionRequest, bundle: Bundle, engine_rid: str):
        if not self.is_prepared(bundle.reference):
            raise JevError(
                "replica_not_ready", "This API worker has not validated the active version", 503
            )
        self._validate_bundle(bundle)
        questions = request.questions or bundle.questions
        if bundle.candidate_policy == "fixed" and questions != bundle.questions:
            raise JevError(
                "task_mismatch", "Fixed task bundle does not allow question changes", 409
            )
        if not questions or len(questions) > bundle.policy.max_questions:
            raise JevError(
                "question_budget", "Request must contain an allowed number of questions", 413
            )
        compiled = tuple(
            self.compiler.compile(request.input.text, question, bundle, engine_rid)
            for question in questions
        )
        sequences = tuple(sequence for question in compiled for sequence in question.sequences)
        return compiled, sequences, self._validate_sequences(sequences, bundle)

    def compile_preview(self, request: DecisionRequest) -> dict:
        """Export the actual worker's validated scoring inputs without GPU dispatch."""
        snapshot = self.registry.acquire(
            request.model, "compile-" + uuid.uuid4().hex, request.bundle, self.backend_identity
        )
        try:
            compiled, sequences, total_tokens = self._compile_request(
                request, snapshot.bundle, "preview"
            )
            prompts = [asdict(sequence) for sequence in sequences]
            return {
                "worker_id": self.registry.owner,
                "bundle": snapshot.bundle.model_dump(mode="json"),
                "bundle_digest": snapshot.bundle.digest,
                "generation": snapshot.generation,
                "logical_prompt_tokens": total_tokens,
                "tokenizer_implementation_digest": self.compiler.tokenizer_implementation_digest,
                "sequences": prompts,
                "input_digest": content_digest(prompts),
                "questions": [
                    {"id": q.question.id, "mode": q.mode, "keys": q.keys} for q in compiled
                ],
                "engine_dispatched": False,
            }
        finally:
            self.registry.release(snapshot.lease_id)

    async def decide(
        self, request: DecisionRequest, request_id: str | None = None, tenant: str = "default"
    ) -> DecisionResponse:
        rid = request_id or request.request_id or "dec-" + uuid.uuid4().hex
        if rid in self._active:
            raise JevError("duplicate_request", "Request ID is already in flight", 409)
        current = asyncio.current_task()
        self._active[rid] = current
        self._tenants[rid] = tenant
        # Engine IDs must not inherit caller-controlled prefixes: SGLang's abort
        # matches prefixes, so a caller ID must never overlap another request's
        # scoring branch namespace.
        engine_rid = "jev-" + uuid.uuid4().hex
        started = time.monotonic()
        snapshot = None
        unconfirmed: set[str] = set()
        try:
            async with asyncio.timeout(request.execution.timeout_ms / 1000):
                snapshot = self.registry.acquire(
                    request.model, rid, request.bundle, self.backend_identity
                )
                bundle = snapshot.bundle
                compiled, sequences, total_tokens = self._compile_request(
                    request, bundle, engine_rid
                )
                questions = tuple(item.question for item in compiled)
                self.registry.record_branches(
                    snapshot.lease_id, [seq.request_id for seq in sequences]
                )
                logger.info(
                    "jev_scoring %s",
                    json.dumps(
                        {
                            "request_id": rid,
                            "tenant": tenant,
                            "bundle": bundle.reference,
                            "bundle_digest": bundle.digest,
                            "generation": snapshot.generation,
                            "engine_request_ids": [seq.request_id for seq in sequences],
                        }
                    ),
                )
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
                        await cancel_and_drain(tasks)
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
                    self.registry.mark_abort_pending(snapshot.lease_id, unconfirmed)
                else:
                    self.registry.release(snapshot.lease_id)
            self._active.pop(rid, None)
            self._tenants.pop(rid, None)

    async def recover_cancelled(self, request_id: str) -> bool:
        snapshot = self.registry.recovery_snapshot(request_id, self.backend_identity)
        if snapshot is None:
            return False
        for branch in snapshot["engine_request_ids"]:
            async with asyncio.timeout(10):
                await self.backend.cancel(branch)
        self.registry.release_recovered(snapshot)
        return True

    def pending_cancellations(self) -> list[dict]:
        return self.registry.recovery_candidates()

    async def cancel(self, request_id: str, tenant: str = "default") -> bool:
        if self._tenants.get(request_id) != tenant:
            return False
        task = self._active.get(request_id)
        if not task:
            return False
        if not task.cancelling():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if any(row["request_id"] == request_id for row in self.registry.recovery_candidates()):
            raise JevError("cancellation_unconfirmed", "Engine abort is not yet confirmed", 503)
        return True
