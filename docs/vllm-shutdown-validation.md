# Native vLLM shutdown budget: DSW validation

Date: 2026-09-28. Runtime/launcher commit:
`e3f82d705b5db37504ec88f7c9ab89cf5a9c041c`.

Three real GPU runs completed with positive native shutdown budgets: one API
worker, two API workers, and a two-worker restart preserving the registry. Each
owned process group exited; none of these logs contains forced child cleanup,
leaked-semaphore warnings, `mode=abort`, or a traceback. This qualifies the tested
shutdown path only. SGLang's earlier shutdown tracebacks and the wider release
gates remain open.

## Cause and change

The previous [quiescence campaign](quiescence-validation.md) already completed
the Jev drain handshake before signalling the native parent. vLLM nevertheless
logged forced child cleanup and leaked semaphores. Inspection of the actual
installed vLLM 0.30.0+cu129 sources found a native shutdown timeout of zero:

- `engine/arg_utils.py` defaults `shutdown_timeout` to 0 and describes zero as abort.
- `entrypoints/cli/serve.py` passes that budget to the API process manager and the
  remaining deadline to the engine process manager.
- `v1/utils.py` signals processes, waits within the budget, then forcibly kills
  remaining children. An explicit zero provides no join interval.
- The API launcher and EngineCore explicitly distinguish abort and drain modes.

The [source excerpts](../evidence/dsw/vllm-shutdown-e3f82d7/native-shutdown-excerpts.json)
retain original paths, line numbers and SHA256 values. All six native source hashes
were unchanged before and after the campaign; no installed engine file was patched.

The DSW launcher now supplies `--shutdown-timeout 30` for native vLLM. An explicit
`--vllm-shutdown-timeout` accepts integer seconds from 1 through 300; it is rejected
for a gateway or SGLang. `process.json` records the effective native budget.
The Jev handshake and native resource teardown have separate deadlines. A service
manager must allow time for both, as explained in [the operator guide](quiescence.md).

## Profile and results

The checkpoint is SmolLM2-1.7B-Instruct revision
`31b70e2e869a7173562077fd711b654946d38674`, with BF16 backbone/readout,
TP1, eager execution, context 2048, four running sequences and chunk budget 512.
It uses shared L20Z GPU 7, UUID `GPU-b57fb933-0e5a-dd28-7041-a177da03405e`.
The memory fraction is 0.07 of total device memory. Health interval/timeout/max-age
are 1/10/30 seconds. These are colocated development checks, not performance runs.

| Run | API workers | Native budget | Main check | Stop-to-group-exit observation |
|---|---:|---:|---|---:|
| Single | 1, launcher default | 30 s, launcher default | Typed decision and raw score | 4.301 s |
| Dual | 2 | 30 s, launcher default | In-flight drain and cross-worker admission rejection | 6.319 s |
| Restart | 2, new identities | 45 s, explicit override | Same registry and unchanged routes/generations | 6.419 s |

The duration wraps the product stop helper, including its Jev handshake and
one-second group polling. It is not an exact native exit latency or an SLO.
The detached launcher does not capture process exit codes; terminal groups do
not establish an exit code of zero. No manual SIGKILL remediation was used here.

The dual-worker run reuses the committed lifecycle harness. It observes a long
decision lease and a raw-score journal before closing admission. The gate reports
an outstanding accepted decision; both workers then reject new typed/raw scores,
prepare and activation calls with 503 `backend_quiescing`. The long request
finishes all 128 questions. Readiness returns 503, both workers acknowledge
QUIESCED, and health monitors remain stopped beyond two intervals. Routes stay
unchanged. Raw work was observed before closure; no claim is made that its GPU
execution, or a health canary, was still in flight exactly at gate closure.

The restart uses the guarded offline resume CLI after the old group exits and
launches with `--no-bootstrap`. Each new worker serves both preserved aliases.
Across the three runs, eight typed responses answer 135 questions, two raw scores
complete, and eight expected gate rejections are retained. Health probes are
excluded from these response counts. Prediction quality was not evaluated.

The native logs show positive-timeout drain and EngineCore resource teardown in
all three runs. Ordinary startup/JIT warnings remain. The dual-worker shutdown
also logs that the shared port is held by its sibling API worker; both workers
then complete application shutdown and the whole group exits. **This is not a
claim of warning-free logs or absence of every possible OS resource leak.**

## Verification and evidence

- 497 local Python tests pass, including 43 launcher/guarded-stop tests. Invalid
  timeout values and applying the option to the wrong deployment mode fail before
  allocation. Ruff and formatting pass.
- 114 CPU contracts pass in the actual vLLM environment using isolated test tools.
  These are distinct from the three GPU runs and do not reinstall engine packages.
- Three local wheels build; 32 core and 4+4 plugin Python files match committed
  source. GPU runs use the source checkout, not these wheels. TypeScript is unchanged.
- 43 exported artifact hashes and 165 Python source hashes verify. Three stopped
  SQLite snapshots represent two distinct registries because restart reuses one.
  Integrity/foreign-key checks pass; leases, work, tickets and recovery claims are zero.
- All 307 retained owned process records/groups are terminal. GPU 7 free memory
  returns to 11,990 MiB. Eight model-file hashes match the earlier immutable Hub
  audit; no fresh Hub query is claimed.

Inspect the [campaign](../evidence/dsw/vllm-shutdown-e3f82d7/campaign.json),
[manifest](../evidence/dsw/vllm-shutdown-e3f82d7/export-manifest.json),
[verified summary](../evidence/dsw/vllm-shutdown-e3f82d7/verified-summary.json),
[final audit](../evidence/dsw/vllm-shutdown-e3f82d7/audit.json), and
[local checks](../evidence/local-check-e3f82d7.json).

Run `.venv/bin/python evidence/harnesses/verify_vllm_shutdown_e3f82d7.py` from the
repository to verify artifacts, committed source hashes, responses, native log
counts, route preservation and the SQLite snapshots. The evidence archive SHA256 is
`4e076509e41771f7a3d5a879445f3968ac1a37ea954986dbe6b0158c88a3dbe9`.

Functional model coverage remains 24/40, twelve per engine. Numerical certification,
business quality, controlled performance, 24-hour soak, container execution,
migration/downgrade and final release acceptance are unchanged. Native chat traffic,
other registries/hosts, larger/quantized/distributed models, API-process-only faults
and a native teardown exceeding the configured budget are outside this campaign.
