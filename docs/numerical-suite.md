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
