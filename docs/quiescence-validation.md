# Backend quiescence: implementation and DSW evidence

Date: 2026-09-28. Runtime protocol: `919c859598bbbd6bb885be280dd0b46360afd532`.
Corrected native-parent stop: `7fe8e723d4b74db22fff12320a3073c85734cce0`.

The corrected campaign passed five colocated GPU lifecycle runs: two vLLM and
three SGLang. The Jev gate rejects new work across both API workers, accepted work
drains, periodic canaries stop, and aliases survive explicit offline resume and
restart. Every owned process group is terminal at the final audit. This is a
scoped Jev drain/process-reclamation result, **not full graceful native shutdown
or a production release certificate**. Native shutdown warnings remain below.

Follow-up: [positive native vLLM shutdown budget](vllm-shutdown-validation.md)
records three later runs at `e3f82d7` without forced child cleanup, leaked-semaphore
warnings or tracebacks. It does not revise the historical results below or resolve
the SGLang native shutdown messages.

## Change and operator contract

The [operator guide](quiescence.md) specifies the authenticated HTTP/CLI handshake,
generation checks, timeout semantics, raw-work recovery, and restart procedure.
The implementation adds a persistent backend gate, worker acknowledgements, raw
score journals and serialized recovery claims. Decisions check the gate in their
existing admission transaction; their ordinary two-commit path remains intact.
Preparation, publication and adapter operations also check authoritative state.
Already queued decisions drain normally; cancellation control remains available.

Quiescence does not disable or rewrite routes. A timeout keeps admission closed.
Restart cannot silently reopen the gate: the local resume operation requires no
retained work and every old API worker stopped or verifiably dead. Recovery claims
also prevent an in-progress administrative abort from being mistaken for completion.

`dsw_service.py stop` performs the handshake, rechecks the owned native parent's
PID/start ticks/boot ID, then signals **that parent**. It does not broadcast SIGTERM
to the process group. Signal delivery and confirmed process-group exit remain
separate results.

## Frozen test scope

| Item | Actual profile |
|---|---|
| Checkpoint | SmolLM2-1.7B-Instruct, revision `31b70e2e869a7173562077fd711b654946d38674` |
| Engines | vLLM 0.30.0+cu129; SGLang 0.5.19 |
| Execution | BF16 backbone/readout, TP1, two API workers, eager mode |
| Device | Shared L20Z GPU 7, UUID `GPU-b57fb933-0e5a-dd28-7041-a177da03405e` |
| Health | Interval 1 s, timeout 10 s, maximum evidence age 30 s |
| Decision workload | Two warm decisions, then one 128-question decision with one parallel branch |
| Raw workload | One native-plugin selected-label score, with its journal observed before gate closure |
| Deployment | Source checkouts through explicit PYTHONPATH; local wheels were built but not installed here |

The long request's durable lease is observed before closing the gate. Both workers
then reject new decision, raw-score, prepare and publication requests with 503
`backend_quiescing`. Readiness also rejects traffic. The accepted long decision
finishes all 128 questions; the raw response completes. Both workers acknowledge
QUIESCED. After more than two health intervals, their monitors remain stopped and
no Jev work remains. This GPU check does not claim that a canary was necessarily
in flight at gate closure; that exact interleaving is covered by controlled tests.

The restart checks reopen admission with the CLI only after process exit, reuse
the same registry and backend identity, suppress bootstrap publication, and confirm
that both new workers serve both preserved aliases with unchanged generations.

## Original failure is retained

At `919c859`, Jev quiescence and serving/restart checks completed on both engines,
but the second SGLang shutdown failed its 60-second process-group exit check.
The stop receipt already showed zero leases, raw work, adapter operations and
recovery claims. One tokenizer worker remained QUIESCED; the native parent waited
for that child. Three processes remained, while GPU memory had returned to baseline.
The log records the group signal terminating the detokenizer during native shutdown.

After preserving the failed receipt, exact parent identity, `sglang.launch_server`
command, run configuration and zero retained work were rechecked. An explicit
SIGKILL cleaned only that owned test group. Its separate remediation receipt does
not change the original failure into a pass. No direct SQL deletion was used.

The follow-up changes the launcher to signal the parent after Jev drain. It also
allows the CLI transport timeout to cover the API's full 300-second drain window.

## Corrected results

| Engine | Runs | Preserved-state restart checks | Final group exit |
|---|---:|---:|---:|
| vLLM | Initial + restart | 1 | 2/2 |
| SGLang | Initial + restart + second restart | 2 | 3/3 |

Across these corrected runs, 18 typed responses completed all 272 requested
questions, and two raw responses completed. The artifacts retain 16 expected
admission/preparation/publication rejections across the two workers and engines.
These counts exclude health probes and do not measure prediction quality.

All five corrected runs retain native shutdown messages: vLLM reports forced child
cleanup and one leaked-semaphore warning per run; SGLang records SystemExit/
CancelledError shutdown tracebacks. The logs contain no Jev canary-failure or
unconfirmed-cancellation messages in these corrected runs. **No native-warning-free
shutdown, absence of all OS-resource leaks, or long-term stability is asserted.**

Final verification checks:

- 490 local Python tests passed; the 17 quiescence/stop tests include startup races,
  queued work, an active canary, a prepare waiting on a lock, raw abort uncertainty,
  duplicate recovery claims, adapter reconciliation and guarded signalling.
- 74 relevant CPU contract tests passed separately in each actual engine environment,
  for both the initial and corrected versions. These use engine doubles, not extra GPU runs.
- Ruff and formatting passed. All three wheels built; 32 core and 4+4 plugin Python
  files matched the committed sources. TypeScript was unchanged and not rerun.
- 121 exported artifact hashes and 314 Python source-file hashes across the two
  declared commits verified locally. The stopped SQLite snapshots pass integrity
  and foreign-key checks, with zero leases, work, tickets or recovery claims.
- Nine retained run snapshots represent **four distinct registries**, because each
  restart deliberately reuses its earlier registry. They are not nine independent databases.
- All 304 retained owned process identities/groups are terminal. GPU 7 free memory
  is 11,990 MiB, matching the baseline. Eight model-file hashes still match the
  previously Hub-verified immutable checkpoint audit; no new Hub query was made.

## Reproduce and inspect

- [Complete evidence and export manifest](../evidence/dsw/quiescence-7fe8e72/export-manifest.json)
- [Corrected campaign](../evidence/dsw/quiescence-7fe8e72/campaign.json)
- [Initial failed campaign](../evidence/dsw/quiescence-7fe8e72/initial/campaign.json)
- [Failed shutdown receipt](../evidence/dsw/quiescence-7fe8e72/initial/sglang-restart/cleanup.json)
- [Separate owned-process remediation](../evidence/dsw/quiescence-7fe8e72/initial/sglang-restart/cleanup-remediation.json)
- [Final audit](../evidence/dsw/quiescence-7fe8e72/audit.json)
- [Verifier](../evidence/harnesses/verify_quiescence_7fe8e72.py)
- [Local source/package checks](../evidence/local-check-7fe8e72.json)

Run `.venv/bin/python evidence/harnesses/verify_quiescence_7fe8e72.py` from this repo.
The initial full source archive was verified before extraction. The corrected
source was copied from that exact base and overlaid with a SHA-bound four-file
Git archive; the audit verifies every Python file against each declared commit.
The exported evidence archive SHA256 is
`36f60bf51d8ae0bc54feabc270b7fe35c7cd6fbf06487cc1ba8d79acd1ca74ba`.

## Gates unchanged

Functional model coverage remains 24/40 (12/20 per engine); numerical, business
quality, controlled 144-case performance, 24-hour stability and full release gates
are not newly accepted. The gate adds a SQL read and configured raw scoring adds
persistent writes; this version has not had a matched performance campaign. Prior
performance ratios are not measurements of this new version.

Mixed-version migration, old-schema snapshot conversion/downgrade, real container
preStop/PID1 behavior, broader host/native shutdown failures and operations handoff
remain incomplete. Raw-only plugins without a configured Runtime, native chat
traffic, other registries and other hosts are outside this quiescence handshake.
