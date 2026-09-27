# Serialized tokenizer preservation campaign

These `.py.txt` files preserve the exact executed bytes. Reports bind their SHA256.
They are task-owned historical controllers with fixed paths, revisions, GPU budget
and fresh-output checks; they are not general installation commands.

| Script | Role |
|---|---|
| `tokenizer_campaign-fbd5b36.py.txt` | Initial CLI/CPU matrix in both environments; includes the three identity-postprocessor rejections, GLM format rejection and eight absent checkpoints. |
| `diagnose_backend-fbd5b36.py.txt` | Exact backend differences for Phi-4, OLMo and Smol. |
| `tokenizer_campaign-cdd0caf.py.txt` | Final CLI/CPU matrix with explicitly checked single-text identity semantics. |
| `olmo_native_diagnosis-cdd0caf.py.txt` | Actual SGLang native tokenizer getter: zero corpus differences and equality to the historical native implementation fingerprint. |
| `native_campaign-cdd0caf.py.txt` | Initial controller; missing Hatchling aborts installation before any server is launched. |
| `native_campaign-cdd0caf-r2.py.txt` | Isolated build-dependency installation and two full native Smol validations; reference failures would remain in their own status. |
| `audit_export-cdd0caf.py.txt` | Source/profile/model rehash, 110-record owned-process audit, environment imports and allowlisted export. |

All corpus inputs are public synthetic fixtures from the earlier Smol diagnosis.
The CLI itself is product code at the named source commit; it is not replaced by
a successful ad hoc conversion script. CPU probes do not imply new GPU model
certifications. See [the scoped report](../../../docs/preserve-fast-validation.md).
