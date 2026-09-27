# GLM4 / R1 Qwen campaign scripts

These `.py.txt` files preserve exact executed bytes and their SHA256 identity.
They are evidence, not a general deployment entry point. Paths, fixed revisions,
run names, available GPU memory and owned-process checks are explicit in each
script. Use fresh outputs before intentional reproduction; never rerun a
campaign while its recorded owned group is live.

| Script | Role |
|---|---|
| `download_verify.py.txt` | Public fixed-revision downloads, selected-file remote/local verification |
| `r1_campaign.py.txt` | Two sequential native R1 Qwen runs, runtime 3377c95 / harness d1ef50e |
| `r1_fidelity.py.txt` | Actual saved R1 input IDs against checkpoint tokenizer data |
| `probe_conversion.py.txt` | Initial GLM prototype; retained added-token construction failure |
| `probe_conversion2.py.txt` | Successful serialization prototype; not used for GPU serving |
| `glm_campaign.py.txt` | Two sequential native GLM runs at c48f18d |
| `profile_equivalence.py.txt` | Production GLM profile vs audited original wrapper in both environments |
| `glm_fidelity.py.txt` | Actual saved GLM input and continuation IDs vs original wrapper |
| `glm_reference.py.txt` | Official local model CPU reference in isolated Transformers 4.44.2 |
| `export_evidence.py.txt` | Source/model rehash, owned-process/group audit, allowlisted export |

Product conversion itself runs the installed `jevctl tokenizer convert-glm4` CLI.
Its manifest identifies `src/jev_runtime/tokenizer_profiles.py` by SHA256. The
isolated reference is the only model forward here that executes checkpoint Python,
after its three local code hashes are checked. The original 5.17.0 loader probe
was an inline read-only check; its structured failure and exact model-code hashes
are retained, but it is not presented as a standalone archived script.

Run `python evidence/harnesses/verify_glm_r1_c48f18d.py` from the repository for
local evidence consistency verification. Passing that verifier confirms integrity
and accounting, including expected numerical failures; it does not turn those
failed model comparisons into passing qualification.
