# Frozen image-input and host-entrypoint checks

The `.py.txt` scripts record the exact DSW operations:

- `inspect_base.py.txt`: retrieves official Docker Hub tag/manifest/config metadata
  and verifies content digests without downloading image layers.
- `export_base_bytes.py.txt`: re-fetches those same digest-bound manifest/config
  bytes for independent local verification.
- `campaign.py.txt`: creates the real Linux context and runs the first two engine
  attempts; both fail in a test assertion on a nonexistent response attribute.
- `campaign_r2.py.txt`: reuses the same immutable context, checks the actual
  response status contract and verifies serving plus graceful process cleanup.
- `audit_export.py.txt`: exact source, installed wheel, payload, model/profile and
  deduplicated process-identity checks with an evidence export allowlist.

These are host-process tests of the container entrypoint, not container execution.
The paths refer to isolated historical DSW runs. Raw logs, credentials, registry
files, wheels and model weights are excluded from this evidence export.
