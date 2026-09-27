# Public engineering evaluation

`benchmarks/prepare_public_quality.py` pins Hub revisions for
[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) topic classification and
[SetFit SST-2](https://huggingface.co/datasets/SetFit/sst2) sentiment classification.
Dataset Viewer split discovery is checked before freezing the file revisions.
The default is 64 examples per class per split: 256 fit + 256 held-out news
examples and 128 fit + 128 held-out sentiment examples, totaling 768.

Sampling balances classes and orders examples by a fixed hash of normalized text.
Normalization is NFKC, whitespace collapse and case folding. Conflicting-label
groups are excluded; any group present in the held-out source is removed from the
fit source before selection. The manifest records raw/unique counts, exclusions,
source revisions, source-byte hashes and derived JSONL hashes. This prevents exact
normalized duplication across the two splits; it does not detect every semantic
duplicate or establish absence from model pretraining. Retain raw text outside Git;
the repository stores the recipe, metadata and numeric prediction evidence.

```sh
pip install -e '.[quality-data]'
python benchmarks/prepare_public_quality.py \
  --output /absolute/new/dataset-directory --source-commit FULL_SHA
python benchmarks/public_quality.py \
  --run-dir /absolute/task-owned/service \
  --data-root /absolute/dataset-directory \
  --output /absolute/new/evidence-directory \
  --source-commit HARNESS_SHA --runtime-source-commit SERVING_SHA
```

The runner checks every dataset hash and deploys a fixed, uncalibrated task bundle
through the real HTTP API. Each attempted example is written immediately to JSONL,
including errors and abstentions. It refuses calibration when scoring is incomplete.
Temperature (news) or positive-slope Platt (sentiment) is fit using the fit split
only. The held-out report includes accuracy with Wilson intervals, macro-F1, NLL,
Brier, ECE and fixed risk/coverage thresholds; it does not select a threshold using
the held-out data. Actual API decisions and abstentions are separate from argmax
probability metrics. Balanced sample metrics are not deployment-prior estimates.

Scores used for fitting are log conditional label probabilities. Removing the
common per-example logit offset leaves both softmax temperature scaling and binary
log-odds unchanged. A zero/invalid probability is an explicit collection error,
not silently clipped into a finite score. The runner then prepares a new calibrated
bundle, activates it, and checks one live example against the fitted transform at
an unchanged 1e-4 tolerance after matching warmup. Full held-out calibration metrics
are computed from the collected scores; the one-example live check validates the
deployment transformation, not a second full inference cohort.

These are public engineering checks. They do not replace the required two approved
business tasks, owner-defined error thresholds, domain-held-out evaluation, or
the full model/performance matrix. `release_gate_passed` remains false.

## First DSW development evidence

Both engines completed all 768 requests with SmolLM2-1.7B at the frozen checkpoint.
SGLang used a two-worker gateway at `364b6e0` over the native engine at `4e5203f`;
vLLM used its two-worker native plugin at `97271f8`. The runner was `97271f8`.
The following numbers count correct **actual API decisions** over every held-out
example; abstentions stay in the denominator and are not correct decisions.

| Engine | News correct / total | News abstentions | Sentiment correct / total | Sentiment abstentions |
|---|---:|---:|---:|---:|
| SGLang | 191 / 256 | 12 | 117 / 128 | 2 |
| vLLM | 190 / 256 | 12 | 115 / 128 | 3 |

Held-out probability NLL improved after fitting: SGLang news 0.7315 → 0.6998 and
sentiment 0.4132 → 0.2337; vLLM news 0.7346 → 0.7039 and sentiment 0.4186 → 0.2465.
Argmax accuracy was unchanged by calibration in both runs. Each live calibrated
example matched its fitted transform within 1e-4, and all temporary versions were
disabled and retired after draining. These results do not establish equality
between engine kernels or unseen-business quality. The complete reports and every
attempt are in `evidence/dsw/{sglang,vllm}-public-quality-97271f8/`.

The raw-score CLI collector also uses durable preparation/request leases and
dispatch journals, unique random engine IDs, ordinary admission and the runtime's
abort cleanup. An unconfirmed abort retains its lease for explicit recovery.
