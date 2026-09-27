# Installed-wheel release campaign scripts

These `.py.txt` files preserve the exact executed controller/export bytes and are
checked against the SHA256 recorded in their reports. They are historical evidence,
not a generic deployment command. They contain fixed revisions, task-owned paths,
ports and GPU budgets. Intentional reproduction requires fresh output directories
and fresh resource/process-identity checks; never reuse a live process record.

| Script | Role and original result |
|---|---|
| `campaign.py.txt` | Initial vLLM campaign; gateway stages ran, then duplicate exclusive cleanup-file write caused exit 1; original report remains incomplete. |
| `campaign_sglang_followup.py.txt` | Initial SGLang campaign; strict native/gateway tokenizer identity check failed and cleanup passed. |
| `campaign_explicit_profile.py.txt` | Corrected complete campaign on both engines using the checkpoint-preserving Smol tokenizer; previous/canary/active/rollback stages passed. |
| `audit_export.py.txt` | Final source/model rehash, installed-environment inspection, 108-record process/group audit, artifact hashes and allowlisted evidence export. |

The runtime installer is `deployment/release.py` at `4e49524`. Wheels were built
from immutable source snapshots `3377c95` and `861ed3a`. Native serving used
`c48f18d` in existing isolated engine environments; do not describe it as a
wheel-installed engine test. Rebuilt `4e49524` wheels match the candidate bytes.

Tokenizer diagnostic probes and explicit-profile creation were inline SSH/Python
operations. Their structured data, fixed inputs, hashes and results are retained;
they are not presented as byte-exact archived standalone scripts. The two failed
alternate dependency resolutions and first vendored-metadata failure are also
retained. See [the scoped report](../../../docs/release-rollout-validation.md).

Run `python evidence/harnesses/verify_release_packaging_4e49524.py` to verify
local evidence consistency. Integrity/accounting success does not certify model
quality, numerical accuracy, controlled performance or uninterrupted rollout.
