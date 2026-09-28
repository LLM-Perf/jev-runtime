# Recovery-only startup: DSW validation

**The retained SGLang preparation lease is recovered through the administrative
API, and both engines resume serving after leaving recovery mode.** Runtime source
`95de3d5` supplies the explicit maintenance startup mode. No database row is
manually deleted or rewritten. The previous failed campaign and its original
SQLite snapshot remain unchanged as historical evidence.

## Profiles and fault scopes

The model is `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with each engine's preserved tokenizer.
Both profiles use TP1/API1, eager execution, context 2048 and maximum four running
sequences on the same colocated L20Z GPU. Existing services remain running.

| Engine | Serving profile | Fault under recovery | Evidence boundary |
|---|---|---|---|
| SGLang 0.5.19 | FP32 backbone/readout, explicit invariant mode, Triton attention | Original ordinary-FP32 startup/canary crash from `f2ccc2a`; one PREPARING bundle and one journaled branch | The old native group was already verified dead. Its unsupported ordinary mode is not repaired or relabelled as passing. Recovery uses the supported invariant configuration and a new serving bundle. |
| vLLM 0.30.0+cu129 | BF16 backbone/readout, ordinary mode | Task-owned preparation-journal process SIGKILL after durable branch recording, before native dispatch | The replacement engine and cancellation endpoint are real. This is a process-crash/before-dispatch test; no killed GPU request is claimed. |

SGLang uses Transformers 5.12.1, vLLM 5.17.0, and both use Torch 2.13.0+cu129.
Source imports are captured and checked for every native launch. New local
wheels are built and compared with source, but DSW runs these recorded source
checkouts, not the new wheels.

## Recovery and resume

Each engine completes three separate native process lifetimes:

1. **Maintenance startup:** explicit `--recovery-only --no-bootstrap`, the
   original registry and original backend target. The operator profile is
   available; Jev readiness returns 503 `recovery_only`. The recovery worker
   does not join admission/publication and no bundle is prepared or activated.
2. **Normal startup after recovery:** recovery mode is disabled. An immutable
   `recovered-profile@1` bundle is built from the actual worker profile, prepared
   with a real canary and activated as `decision-model` at generation 1. Ten
   decisions complete.
3. **Another normal restart:** the active bundle is revalidated during startup,
   readiness succeeds, the same route/digest/generation remain, and another ten
   decisions complete.

During maintenance, six authenticated writes are rejected per engine: typed
decisions, raw plugin scores, preparation, activation, disable and retirement.
Their attempts do not alter the bundle/route/lease listing. Recovery using the
data credential returns 401. The admin request verifies a dead owner and the
recorded branch ID, awaits native cancellation, releases the lease through the
existing compare-and-swap path and marks PREPARING as FAILED. Repeating recovery
returns `recovered: false`. Readiness stays closed until a normal process starts.

Both original bundle manifests, digests, backend bindings and creation records
remain unchanged. Each registry contains exactly one matching `recovered` event.
The old SGLang configuration, process record, credentials and logs retain their
original hashes. Enabling invariant execution is explicit; the old ordinary-mode
bundle is neither repurposed nor activated with a changed identity.

The resumed serving count is **40/40 completed responses**, twenty per engine,
with all answers marked `answered`. The verifier checks the response schema,
bundle digest/reference, generation and usage success counts. This repeated
one-question fixture validates recovery/resume, not business accuracy, numerical
equivalence, throughput or a latency SLO.

## Audit and limits

Four consistent private snapshots capture each registry before and after the
exercise. Their payload hashes, exact schema, integrity, content digests and raw
table rows verify. Before-state blockers are one lease, one work journal and one
PREPARING bundle per registry. After-state blockers are empty, all owned worker
groups have exited, and GPU7 returns to 11,990 MiB free.

The final audit checks **276 retained owned process identities/groups are
terminal**, seven checkpoint files against pinned Hub digests, both tokenizer
profiles, all source hashes and absence of the campaign's credential values from
the export. **44 exported artifacts** are hash-bound; the offline verifier adds
an independently recomputed summary. It links the prior failed audit rather than
changing that audit to a pass.

At the exact runtime source, **452 local tests pass** and lint/format checks pass.
All three wheels build, and their 31 core / 4 SGLang / 4 vLLM Python payload files
match source. TypeScript is unchanged and is not rerun.

Launcher-only follow-up `bfb8cde` omits a disabled `recovery_only` field from
ordinary configurations, preserving their prior shape. Its 33 launcher tests
pass; core/plugin sources and the enabled recovery setting are unchanged.
The GPU evidence remains at `95de3d5`, before this serialization follow-up.

This closes the observed retained SGLang preparation lease and establishes the
tested maintenance/resume paths. It does not certify other crash timings,
multi-worker recovery, LoRA or adapter residency recovery, a lost host/source,
PID-namespace replacement, schema migration, engine failover or a 24-hour soak.
Native chat/generation remains under the host engine's controls; recovery mode
only gates Jev endpoints. Functional coverage stays **24/40**, historical
numerical failures remain, and full release acceptance is false.

- [Operation procedure](recovery-mode.md)
- [Recomputed summary](../evidence/dsw/recovery-mode-95de3d5/verified-summary.json)
- [Raw campaign](../evidence/dsw/recovery-mode-95de3d5/campaign.json)
- [Final audit](../evidence/dsw/recovery-mode-95de3d5/audit.json)
- [Offline verifier](../evidence/harnesses/verify_recovery_mode_95de3d5.py)
- [Local checks and wheel payload hashes](../evidence/local-check-95de3d5.json)
- [Launcher compatibility follow-up](../evidence/local-check-bfb8cde.json)
