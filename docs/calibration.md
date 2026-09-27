# Fixed-task calibration

Calibration is tied to a model revision, backbone/readout precision, tokenizer, template, candidate order and
question definitions. Keep the bundle `candidate_policy` set to `fixed` and
`template.mode` set to `joint-label` for this collector. Supported targets are
choice IDs, JSON booleans, and discrete score-level IDs. Rank requires a separate
relevance evaluation and is rejected by this collector.

Changing `model.readout_dtype` changes the scoring contract. Rebuild the bundle
and refit calibration; an old artifact cannot be transferred by editing its
digest. See [precision configuration and legacy migration](readout-precision.md).

Prepare two JSONL datasets with disjoint `sample_id` **and** `group_id`. Use a
conversation/customer/document identifier as the group when examples can leak
information across splits. Each example must label every question in the bundle:

```json
{"sample_id":"case-001","group_id":"conversation-001","input":{"text":"I want a refund."},"labels":{"intent":"billing","refund_requested":true}}
```

Collect scores from the actual configured engine. The commands do not change
active serving aliases. They fail on missing labels or an incomplete engine result.
Input text is not copied into the score artifact.

```sh
jevctl calibration collect gateway.yaml task-bundle.json fit.jsonl fit-scores.json
jevctl calibration collect gateway.yaml task-bundle.json heldout.jsonl heldout-scores.json
jevctl calibration fit task-bundle.json fit-scores.json heldout-scores.json calibration-v2
```

The engine credential comes from `JEV_ENGINE_API_KEY`. For a native vLLM plugin,
use the same value configured as the plugin's `JEV_API_KEY` in its separate process.
For SGLang, use its engine API credential if one is configured.

Use `--method platt` for binary boolean bundles. The positive slope and intercept
are optimized on the fit split only. Temperature fitting supports multiclass
tasks. The report includes held-out accuracy, task-separated macro F1, NLL, Brier,
ECE bin counts, and risk/coverage with Wilson intervals. It reports deterioration
as well as improvement; fitting does not automatically activate a bundle.

Outputs are `bundle.json` with the next version, `calibration.json`, and
`report.json`. Existing destinations are never overwritten. Review the held-out
results, then upload, prepare, and activate the new immutable version through the
ordinary bundle lifecycle. Calibration on arbitrary dynamic candidates is rejected.

The supplied unit tests use constructed distributions to test algorithms. They
are not evidence of calibrated real-world task accuracy. Deployment acceptance
still requires representative business data with grounded labels, enough samples,
and a held-out split that was not used to tune prompts or abstention thresholds.
