# Frozen native rollback/cancellation campaign

These `.py.txt` files preserve the exact scripts executed on DSW:

- `campaign.py.txt`: offline wheel installation and the initial two-engine runs;
  both stop at the retained admin-client setup failure.
- `retest_campaign.py.txt`: source-hash-bound retest after fixing client cleanup,
  reusing the installed candidate whose core bytes are unchanged.
- `audit_export.py.txt`: allowlisted evidence export, deduplicated PID/start-tick/
  boot-ID ownership checks, model/profile rehashes and source/import verification.

They are historical records, not generic scripts to rerun on another machine.
Credentials, databases, weights and raw service logs are not exported.
See [the scoped report](../../../docs/native-rollout-cancellation.md) and run
`evidence/harnesses/verify_native_rollout_cancel_b20d3f4.py` from the repository root
to verify the retained artifacts without starting services.
