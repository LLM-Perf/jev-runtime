# Multi-input numerical diagnostics

`tests.integration.numerical_suite` extends the previous single-position checks
to a frozen synthetic corpus of 32 cases: eight English/Chinese inputs, each
with boolean, four-choice, eight-choice and identical-description four-choice
questions. The latter challenges label bias; the report counts an actual near
tie only when the reference's two largest selected logprobs differ by at most
`2 * 0.15`. Synthetic categories have no business ground-truth labels.

The collector uses the running plugin's compiler and retains exact input IDs,
ordered label IDs, model identity and compiler profile. It records 96 scoring
responses, covering each case in three execution states:

1. Serial submission after a confirmed local prefix-cache reset.
2. Immediate repetition of the same input.
3. A fresh reset, reversed corpus order, four concurrent client submissions.

These names describe interventions, not measured cache-hit ratios or realized
GPU batch sizes. Cached-token counters are retained when available. A periodic
health canary can affect cache contents; the campaign must record its cadence
and check whether collection spans a canary interval. The tool flushes only an
explicitly identified, live TP1 engine created by `deployment/dsw_service.py`.
It is intended for an isolated test service, never a shared production endpoint.
vLLM's reset route requires its development endpoints to be enabled at startup.
SGLang resets request its native ten-second deferred idle barrier: a completed
HTTP response alone does not establish that the scheduler can reset its pools.
A timeout or rejected reset still fails collection; it never becomes a cold case.

After the serving engine exits, run independent Transformers reference forwards
on the saved IDs. This initial implementation requires unquantized BF16 backbone
and head, one visible GPU, a resident model fitting the bounded allocator, batch
size one and `use_cache=False`. It retains GPU identity, package versions, math
settings, observed head dtype and peak reserved memory. Eager and SDPA are two
separately named reference contracts. Neither is chosen after observing which
one agrees best. The existing single-position CPU failures remain authoritative
for their original execution profiles.

```sh
# In the same source checkout / isolated engine environment used by the campaign:
python -m tests.integration.numerical_suite collect \
  --run-dir /path/to/owned/native-run --output /path/to/new-engine-report.json

# Only after stopping the owned engine and checking its GPU allocation:
CUDA_VISIBLE_DEVICES=7 python -m tests.integration.numerical_suite reference \
  --engine-report /path/to/new-engine-report.json \
  --model-path /path/to/pinned/local/model --attention sdpa \
  --output /path/to/new-reference-report.json

python -m tests.integration.numerical_suite compare \
  --engine-report /path/to/new-engine-report.json \
  --reference-report /path/to/new-reference-report.json \
  --output /path/to/new-comparison.json
```

Reports are created exclusively; failed attempts are retained. The comparison
checks corpus identity, complete case/state denominators, model identity, exact
sequence digests and reference binding to the engine report. It reports selected
logprob error, conditional-probability error, selected-label mass, top-label
agreement, actual near ties and differences between execution states. Non-finite
values, missing labels and incomplete reports are errors rather than dropped
samples. The comparison process returns nonzero on a failed development check.

The unchanged development check is maximum selected-logprob error `<= 0.15`
**and** top-label agreement, including near ties. It is not a newly calibrated
release budget. All reports retain `full_release_gate_passed: false`: this corpus
does not supply held-out task accuracy, calibration, TP/quantized coverage,
long-context coverage or final profile-specific error budgets. Collection has
no throughput/SLO interpretation.

PyTorch documents why device, batching and precision can change numerical
results; this motivates naming the reference execution contract rather than
treating every forward as interchangeable. See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

## DSW Qwen3-0.6B campaign at `877f319`

Both engines collected all 32 cases in all three states: **192 scoring responses**
in total. Four independent reference executions (two attention implementations
in each engine's Python environment) performed **128 forwards**, yielding
**384 comparisons**. The input and label IDs match between the two engines for
every case; their tokenizer implementation fingerprints remain environment-specific.

The unchanged development check is logprob error `<= 0.15` plus first-argmax label
agreement. Every reference contract fails this check on part of the corpus:

| Serving engine | GPU reference | Passing comparisons | Largest logprob error | Largest conditional-probability error | Argmax mismatches |
|---|---|---:|---:|---:|---:|
| vLLM | eager | 22/96 | 0.998153 | 0.182498 | 7/96 |
| vLLM | SDPA | 33/96 | 0.744257 | 0.091741 | 7/96 |
| SGLang | eager | 25/96 | 0.756199 | 0.182076 | 7/96 |
| SGLang | SDPA | 30/96 | 0.707006 | 0.086882 | 8/96 |

Each reference finds six near-tie cases out of 32. A mismatch uses the first
argmax on each ordered label vector, including exact ties. It does not mean a
typed decision returned a wrong business answer: this corpus has no task labels,
and the default decision policy can abstain on an exact tie. Probability errors
are absolute probability-point differences, not relative percentages or accuracy.

The same saved responses also expose changes within each engine, independent of
the Transformers reference. Each row compares 32 vectors with that engine's
serial result after cache reset:

| Engine | Subsequent state | Largest logprob change | Largest conditional-probability change | Argmax changes |
|---|---|---:|---:|---:|
| vLLM | Immediate repeat | 0.249298 | 0.046951 | 1/32 |
| vLLM | Reversed order, client concurrency 4 | 0.623706 | 0.061355 | 4/32 |
| SGLang | Immediate repeat | 0.374370 | 0.062220 | 1/32 |
| SGLang | Reversed order, client concurrency 4 | 0.485589 | 0.068533 | 2/32 |

All these argmax changes have a baseline top-two gap `<= 0.3`; several baseline
vectors have exact ties. Every reset-serial response reports zero cached tokens,
and every immediate repeat reports positive cached tokens. Concurrent requests
have mixed cache counts. Each collector finishes in about three seconds; its
initial canary age plus collection duration is below the recorded 300-second
probe interval. These observations establish variation under the tested state
interventions, not a causal attribution to one specific GPU kernel.

The profile is pinned `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, preserved fast tokenizer, BF16
backbone/head, TP1/API1, eager serving, context limit 2048, at most four running
sequences. vLLM is `0.30.0+cu129` with Transformers `5.17.0`; SGLang is `0.5.19`
with Transformers `5.12.1`. Both use Torch `2.13.0+cu129`. References run resident,
batch one, no KV cache, BF16 head and FP32 log-softmax after stopping the engine.
Engine-dependent results therefore also retain their different Transformers
versions; they are not a controlled comparison of only serving backends.

An orchestration mistake tried to collect SGLang package metadata in the vLLM
Python process. It happened after complete SGLang scoring and live postcheck
profile capture. The first campaign remains failed with `PackageNotFoundError`.
A separate completion script verified that both engines were stopped, checked
the saved live postcheck profiles and empty final registries, collected metadata
in each correct environment, and ran the missing SGLang references on the
original saved inputs. It did not rerun or replace the scoring samples.

The final audit verifies **171 distinct retained owned process identities/groups
are terminal**, GPU7 is back to 11,990 MiB free, seven checkpoint files match the
pinned public Hub digests, and both tokenizer profiles verify. All copied Python
source hashes match `877f319`. Local 420 tests and Python lint/format checks pass.
Only harness/tests/docs changed, so no new runtime wheel installation is claimed.

These findings keep Qwen's numerical status failed and the overall functional
denominator at **24/40**. No quality, performance or release gate is newly passed.
The subsequent [batch-invariance A/B](batch-invariant-mode.md) tests declared
deterministic engine modes against this corpus and corroborates all saved inputs
through native endpoints. It resolves the observed state variation in the tested
enabled profiles, while independent-reference failures remain. Numerical work
still needs precision choices, a frozen reference/budget and independent held-out
inputs. Near-tie abstention needs explicit quality and coverage evaluation;
increasing this tolerance is not a resolution.

- [Recomputed results](../evidence/dsw/numerical-suite-877f319/verified-summary.json)
- [Initial campaign, including failure](../evidence/dsw/numerical-suite-877f319/campaign.json)
- [Reference completion and environment records](../evidence/dsw/numerical-suite-877f319/finish.json)
- [Final source, model and cleanup audit](../evidence/dsw/numerical-suite-877f319/audit.json)
- [Frozen execution scripts](../evidence/harnesses/numerical-suite/)

Recompute all 384 comparisons and execution-state differences independently from
the retained score vectors:

```sh
.venv/bin/python evidence/harnesses/verify_numerical_suite_877f319.py
```
