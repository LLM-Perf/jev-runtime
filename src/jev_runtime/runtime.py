from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict

from jev_runtime.adapters import AdapterStore
from jev_runtime.admission import Admission
from jev_runtime.backends.base import Capabilities, EngineAdapter, ScoreInput, ScoreResult
from jev_runtime.compiler import CompiledQuestion, Compiler
from jev_runtime.errors import JevError
from jev_runtime.health import HealthSettings, ServingHealth
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
from jev_runtime.shared_admission import SharedAdmission
from jev_runtime.telemetry import DecisionTrace

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(
        self,
        backend: EngineAdapter,
        compiler: Compiler,
        registry: Registry,
        backend_identity: str,
        model_id: str,
        admission: Admission | SharedAdmission | None = None,
        expected_model: ModelIdentity | None = None,
        adapter_store: AdapterStore | None = None,
        adapter_timeout: int = 120,
        health_settings: HealthSettings | None = None,
    ):
        self.backend, self.compiler, self.registry = backend, compiler, registry
        self.backend_identity, self.model_id = backend_identity, model_id
        self.expected_model = expected_model
        self.adapter_store, self.adapter_timeout = adapter_store, adapter_timeout
        self._adapter_tasks: set[asyncio.Task] = set()
        self.admission = admission or SharedAdmission(registry, backend_identity)
        self.capabilities: Capabilities | None = None
        self._management_lock = asyncio.Lock()
        self._active: dict[str, asyncio.Task] = {}
        self._tenants: dict[str, str] = {}
        self._active_leases: dict[str, str] = {}
        self._prepared: set[str] = set()
        self._control_task: asyncio.Task | None = None
        self._cancel_jobs: set[asyncio.Task] = set()
        self.control_healthy = False
        self.health = ServingHealth(health_settings or HealthSettings())
        self._health_task: asyncio.Task | None = None
        self._health_lock = asyncio.Lock()
        self._health_pending: dict[str, str] = {}

    async def start(self) -> None:
        if self._control_task is not None and not self._control_task.done():
            raise RuntimeError("Runtime is already started")
        self._prepared.clear()
        self.admission.start(self.registry, self.backend_identity)
        self.capabilities = await self.backend.probe()
        if self.expected_model is not None and self.capabilities.engine in {"sglang", "vllm"}:
            for expected_field, observed_field in (
                ("dtype", "model_dtype"),
                ("readout_dtype", "readout_dtype"),
            ):
                expected = getattr(self.expected_model, expected_field)
                observed = getattr(self.capabilities, observed_field)
                if expected is not None and expected != observed:
                    raise JevError(
                        "engine_precision_mismatch",
                        f"Configured {expected_field}={expected} differs from engine "
                        f"report {observed}; use an explicit verified engine precision profile",
                        409,
                    )
        if self.adapter_store is not None:
            if not self.capabilities.lora:
                raise JevError(
                    "adapter_profile",
                    "Native engine configuration does not support managed LoRA",
                    409,
                )
            self.registry.start_adapter_session(self.backend_identity)
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
        self.control_healthy = True
        self._control_task = asyncio.create_task(self._watch_cancellations())
        self._health_task = asyncio.create_task(self._watch_health())

    def active_references(self) -> list[str]:
        listing = self.registry.list()
        active = {route["ref"] for route in listing["routes"] if route["ref"]}
        return sorted(
            bundle["ref"]
            for bundle in listing["bundles"]
            if bundle["ref"] in active and bundle["backend"] == self.backend_identity
        )

    def health_profile(self) -> dict:
        return {
            "scope": "local_api_worker",
            "settings": self.health.settings.model_dump(),
            "monitor_running": self._health_task is not None and not self._health_task.done(),
            "bundles": {
                ref: {**self.health.status(ref), "prepared": self.is_prepared(ref)}
                for ref in self.active_references()
            },
        }

    async def check_health(self) -> None:
        async with self._health_lock:
            for reference in self.active_references():
                try:
                    await self.prepare(
                        reference,
                        monitoring=True,
                        timeout_seconds=self.health.settings.timeout_seconds,
                    )
                except Exception as exc:
                    self.health.failed(
                        reference, exc.code if isinstance(exc, JevError) else type(exc).__name__
                    )
                    self.registry.forget_worker_prepared(reference)
                    logger.warning("Jev engine canary failed for bundle %s", reference)

    async def _watch_health(self) -> None:
        while True:
            await asyncio.sleep(self.health.settings.interval_seconds)
            try:
                await self.check_health()
            except Exception:
                # Expiring canary evidence fails closed even if registry access fails.
                logger.exception("Jev engine health polling failed")

    async def _watch_cancellations(self) -> None:
        while True:
            try:
                for command in self.registry.pending_cancel_commands():
                    if self.registry.claim_cancel(command["id"]):
                        task = asyncio.create_task(self._perform_remote_cancel(command))
                        self._cancel_jobs.add(task)
                        task.add_done_callback(self._cancel_job_done)
                self.control_healthy = True
            except Exception:
                self.control_healthy = False
                logger.exception("Jev cancellation control polling failed")
            await asyncio.sleep(0.05)

    def _cancel_job_done(self, task: asyncio.Task) -> None:
        self._cancel_jobs.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.control_healthy = False
            logger.error("Jev cancellation result could not be persisted: %s", task.exception())

    async def _perform_remote_cancel(self, command: dict) -> None:
        state = "UNCONFIRMED"
        try:
            # Caller IDs can be reused after completion. An old queued command
            # must never cancel a new request with the same public ID.
            if self._active_leases.get(command["request_id"]) != command["lease_id"]:
                state = (
                    "UNCONFIRMED" if self.registry.has_lease(command["lease_id"]) else "COMPLETED"
                )
            else:
                cancelled = await self._cancel_local(command["request_id"], command["tenant"])
                state = "CONFIRMED" if cancelled else "COMPLETED"
        except Exception:
            logger.exception("Jev owner could not confirm cancellation")
        finally:
            self.registry.finish_cancel(command["id"], state)

    def is_prepared(self, reference: str) -> bool:
        return reference in self._prepared

    def activate(self, alias: str, reference: str, expected_generation: int) -> dict:
        if not self.is_prepared(reference):
            raise JevError(
                "replica_not_ready", "Prepare this version on this API worker first", 503
            )
        self.health.require(reference)
        return self.registry.activate(alias, reference, expected_generation)

    async def close(self) -> None:
        self.registry.drain_worker()
        self.control_healthy = False
        tasks = [*self._active.values(), *self._cancel_jobs, *self._adapter_tasks]
        if self._control_task is not None:
            tasks.append(self._control_task)
        if self._health_task is not None:
            tasks.append(self._health_task)
        try:
            await cancel_and_drain(tasks)
        finally:
            try:
                await self.backend.close()
            finally:
                self.registry.stop_worker()
                self.registry.close()

    def _validate_bundle(self, bundle: Bundle) -> None:
        self.compiler.verify_bundle(bundle)
        if bundle.model.id != self.model_id:
            raise JevError("model_mismatch", "Bundle is bound to a different model", 409)
        if self.expected_model is not None:
            for field in (
                "revision",
                "dtype",
                "readout_dtype",
                "quantization",
                "tokenizer_digest",
                "tokenizer_implementation_digest",
                "template_digest",
            ):
                if getattr(bundle.model, field) != getattr(self.expected_model, field):
                    raise JevError(
                        "model_mismatch", f"Bundle model {field} differs from loaded profile", 409
                    )
        if bundle.model.adapter_id and (not self.capabilities or not self.capabilities.lora):
            raise JevError("lora_unsupported", "Backend has not enabled adapter scoring", 409)
        if bundle.model.adapter_id:
            self.registry.adapter_binding(
                f"{bundle.model.adapter_id}@{bundle.model.adapter_revision}", self.backend_identity
            )

    def _require_adapters(self) -> None:
        if self.adapter_store is None or self.capabilities is None or not self.capabilities.lora:
            raise JevError(
                "lora_unsupported", "Managed LoRA is not configured for this engine", 409
            )

    async def register_adapter(self, adapter_id: str, source: str) -> dict:
        self._require_adapters()
        if self.expected_model is None:
            raise JevError("adapter_profile", "A frozen base-model identity is required", 409)
        # Registration copies files but never dispatches GPU work. A cancelled
        # filesystem copy can leave only an unreferenced immutable artifact.
        try:
            artifact = await asyncio.to_thread(
                self.adapter_store.register,
                adapter_id,
                source,
                self.expected_model.id,
                self.expected_model.revision,
            )
        except JevError:
            raise
        except Exception as exc:
            raise JevError(
                "adapter_artifact_invalid",
                "Cannot read or validate the local adapter artifact",
                409,
            ) from exc
        return self.registry.register_adapter(artifact, self.backend_identity)

    async def change_adapter(self, reference: str, action: str, recover: bool = False) -> dict:
        self._require_adapters()
        task = asyncio.current_task()
        self._adapter_tasks.add(task)
        try:
            async with self._management_lock:
                operation, binding = self.registry.begin_adapter_operation(
                    reference, self.backend_identity, action, recover
                )
                error = None
                try:
                    async with asyncio.timeout(self.adapter_timeout):
                        if action == "load":
                            await asyncio.to_thread(self.adapter_store.verify, binding.artifact)
                            await self.backend.load_adapter(binding)
                        else:
                            await self.backend.unload_adapter(binding)
                except BaseException as exc:
                    error = exc.code if isinstance(exc, JevError) else type(exc).__name__
                    if isinstance(exc, Exception) and not isinstance(exc, JevError):
                        raise JevError(
                            "adapter_operation_failed",
                            "Adapter operation could not be confirmed; reconcile before reuse",
                            503,
                        ) from exc
                    raise
                finally:
                    self.registry.finish_adapter_operation(reference, operation, error)
                    for bundle in self.registry.inspect_adapter(reference)["bundles"]:
                        self._prepared.discard(bundle)
                        self.registry.forget_worker_prepared(bundle)
                return self.registry.inspect_adapter(reference)
        finally:
            self._adapter_tasks.discard(task)

    async def prepare(
        self, reference: str, *, monitoring: bool = False, timeout_seconds: float = 120
    ) -> dict:
        async with asyncio.timeout(timeout_seconds):
            return await self._prepare(reference, monitoring=monitoring)

    async def _prepare(self, reference: str, *, monitoring: bool) -> dict:
        async with self._management_lock:
            pending = self._health_pending.get(reference)
            if pending and self.registry.has_lease(pending):
                raise JevError(
                    "health_cleanup_pending", "Recover the prior canary before probing again", 503
                )
            self._health_pending.pop(reference, None)
            if monitoring and await self.backend.probe() != self.capabilities:
                raise JevError(
                    "engine_profile_changed",
                    "Engine capabilities changed; restart and validate the configured profile",
                    503,
                )
            generation = self.health.generation(reference)
            # Only an already fully validated immutable bundle gets a minimal
            # periodic canary. New worker/version preparation still checks all tasks.
            minimal = monitoring and self.is_prepared(reference)
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
                if minimal:
                    questions = questions[:1]
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
                self.health.failed(
                    reference, exc.code if isinstance(exc, JevError) else type(exc).__name__
                )
                if not monitoring:
                    self._prepared.discard(reference)
                self.registry.forget_worker_prepared(reference)
                raise
            finally:
                try:
                    if fresh:
                        self.registry.finish_prepare(reference, lease_id, error, unconfirmed)
                    elif unconfirmed:
                        self.registry.mark_abort_pending(lease_id, unconfirmed)
                    else:
                        self.registry.release(lease_id)
                finally:
                    self._active.pop(request_id, None)
                    if self.registry.has_lease(lease_id):
                        self._health_pending[reference] = lease_id
            if self.health.generation(reference) == generation:
                self.registry.record_worker_prepared(reference)
                self.health.passed(reference, generation)
            else:
                self.registry.forget_worker_prepared(reference)
            self._prepared.add(reference)
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

    async def _score(
        self, seq: ScoreInput, unconfirmed: set[str], trace: DecisionTrace | None = None
    ) -> ScoreResult:
        try:
            with trace.measure_work("engine_call") if trace else nullcontext():
                return await self.backend.score(seq)
        except BaseException:

            async def abort():
                with trace.measure_work("abort") if trace else nullcontext():
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
        trace: DecisionTrace | None = None,
    ) -> tuple[Answer, list[ScoreResult]]:
        trace = trace or DecisionTrace()

        async def score(seq: ScoreInput) -> ScoreResult:
            with trace.measure_work("branch_queue"):
                await semaphore.acquire()
            try:
                return await self._score(seq, unconfirmed, trace)
            except Exception as exc:
                if not isinstance(exc, JevError) or exc.status_code >= 500:
                    self.health.failed(bundle.reference, "engine_score_failed")
                    self.registry.forget_worker_prepared(bundle.reference)
                raise
            finally:
                semaphore.release()

        tasks = [asyncio.create_task(score(seq)) for seq in compiled.sequences]
        try:
            results = await asyncio.gather(*tasks)
            with trace.measure_work("assembly"):
                try:
                    return assemble(compiled, results, bundle), results
                except JevError as exc:
                    if exc.status_code >= 500:
                        self.health.failed(bundle.reference, "engine_score_contract")
                        self.registry.forget_worker_prepared(bundle.reference)
                    raise
        finally:
            await cancel_and_drain(tasks)

    def _compile_request(self, request: DecisionRequest, bundle: Bundle, engine_rid: str):
        if not self.is_prepared(bundle.reference):
            raise JevError(
                "replica_not_ready", "This API worker has not validated the active version", 503
            )
        self.health.require(bundle.reference)
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
                "compiler_profile": self.compiler.profile(),
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
        self,
        request: DecisionRequest,
        request_id: str | None = None,
        tenant: str = "default",
        *,
        trace: DecisionTrace | None = None,
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
        trace = trace or DecisionTrace()
        trace.switch("pin")
        try:
            async with asyncio.timeout(request.execution.timeout_ms / 1000):
                snapshot = self.registry.acquire(
                    request.model, rid, request.bundle, self.backend_identity, tenant
                )
                self._active_leases[rid] = snapshot.lease_id
                bundle = snapshot.bundle
                trace.switch("compile")
                compiled, sequences, total_tokens = self._compile_request(
                    request, bundle, engine_rid
                )
                questions = tuple(item.question for item in compiled)
                trace.switch("journal")
                branch_ids = [seq.request_id for seq in sequences]
                admission_args = dict(branches=len(sequences), lease_id=snapshot.lease_id)
                if isinstance(self.admission, SharedAdmission):
                    # The durable admission transaction also journals recovery IDs.
                    admission_args["branch_ids"] = branch_ids
                else:
                    self.registry.record_branches(snapshot.lease_id, branch_ids)
                logger.info(
                    "jev_scoring %s",
                    json.dumps(
                        {
                            "request_id": rid,
                            "tenant": tenant,
                            "bundle": bundle.reference,
                            "bundle_digest": bundle.digest,
                            "generation": snapshot.generation,
                            "engine_request_ids": branch_ids,
                        }
                    ),
                )
                trace.switch("queue")
                async with self.admission.acquire(total_tokens, tenant, **admission_args):
                    # Compilation and durable admission contain synchronous work.
                    # asyncio's timeout callback may not run until we next yield.
                    if time.monotonic() - started >= request.execution.timeout_ms / 1000:
                        raise TimeoutError("Deadline expired before engine dispatch")
                    self.health.require(bundle.reference)
                    trace.switch("execute")
                    semaphore = asyncio.Semaphore(bundle.policy.max_parallel_branches)
                    tasks = [
                        asyncio.create_task(
                            self._question(q, bundle, semaphore, unconfirmed, trace)
                        )
                        for q in compiled
                    ]
                    try:
                        outcomes = await asyncio.gather(
                            *tasks, return_exceptions=request.execution.allow_partial
                        )
                    finally:
                        await cancel_and_drain(tasks)
                trace.switch("finalize")
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
            trace.switch("release")
            try:
                if snapshot is not None:
                    if unconfirmed:
                        self.registry.mark_abort_pending(snapshot.lease_id, unconfirmed)
                    else:
                        self.registry.release(snapshot.lease_id)
            finally:
                self._active.pop(rid, None)
                self._tenants.pop(rid, None)
                self._active_leases.pop(rid, None)
                trace.finish()

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
        if request_id in self._active:
            return await self._cancel_local(request_id, tenant)
        command = self.registry.request_cancel(request_id, tenant, self.backend_identity)
        if command is None:
            return False
        try:
            async with asyncio.timeout(10):
                while True:
                    state = self.registry.cancel_status(command)
                    if state == "CONFIRMED":
                        return True
                    if state == "COMPLETED":
                        return False
                    if state in {None, "UNCONFIRMED"}:
                        break
                    await asyncio.sleep(0.025)
        except TimeoutError:
            pass
        raise JevError(
            "cancellation_unconfirmed", "The owning API worker has not confirmed cleanup", 503
        )

    async def _cancel_local(self, request_id: str, tenant: str) -> bool:
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
