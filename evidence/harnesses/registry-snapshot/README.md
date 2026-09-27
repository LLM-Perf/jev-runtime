# Frozen registry snapshot/staging campaign

- `campaign.py.txt` records the exact two-engine DSW experiment: snapshot a stopped
  historical test registry into a new private copy, serve with two workers, reject
  live/stale restore attempts, publish generation 3, stop, snapshot/stage current
  state, restart on the staged registry and reject stale generation writes.
- `audit_export.py.txt` re-verifies all six private snapshot payloads on DSW,
  source bytes, model/profile hashes, unchanged original registry state and
  deduplicated task-owned process identities. Its export allowlist contains only
  manifests, receipts and non-secret reports; no databases, keys, weights or logs.

These are historical scripts with fixed isolated DSW paths, not generic deployment
commands. The library and CLI are documented in `docs/registry-snapshots.md`.
Run `evidence/harnesses/verify_registry_snapshot_c4e9cef.py` from the repository
root to verify the exported metadata and serving records without starting services.
