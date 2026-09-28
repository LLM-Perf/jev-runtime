"""Export source-bound raw evidence after owned-group and registry verification."""
import hashlib
import json
import sqlite3
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path('/root/jev-runtime')
HERE = ROOT / 'vllm-shutdown-e3f82d7'
def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
campaign = read(HERE / 'campaign.json')
assert campaign['complete'] and read(HERE / 'contracts.json')['complete']
meta = campaign['sources']
source = ROOT / 'quiescence-sources' / meta['commit']
sys.path[:0] = [str(source), str(source / 'src')]
from deployment.dsw_service import process_identity, get_gpu
from tests.integration.run_native_validation import group_members
files = {name: HERE / name for name in ['campaign.json', 'campaign.log', 'sources.json', 'campaign.py', 'export.py', 'contracts.py', 'contracts.json', 'contracts.log', 'contracts-vllm.log']}
previous_path = ROOT / 'quiescence-7fe8e72/audit.json'
previous = read(previous_path)
records = previous['records']
seen = {(r['identity']['boot_id'], r['identity']['pid'], r['identity']['start_ticks']) for r in records}
secrets = []
for attempt in campaign['attempts']:
    assert attempt['complete'] and attempt['cleanup']['passed']
    run = Path(attempt['run_dir'])
    secrets.extend(v for v in read(run / 'keys.json').values() if isinstance(v, str))
    record = read(run / 'process.json')
    registry = Path(record['registry_path'])
    snapshot = run / 'registry-final.sqlite'
    assert not snapshot.exists()
    with sqlite3.connect(f'file:{registry}?mode=ro', uri=True) as db:
        with sqlite3.connect(snapshot) as target:
            db.backup(target)
            target.execute('PRAGMA journal_mode=DELETE')
        db.row_factory = sqlite3.Row
        state = {'observed_at': time.time(), 'registry_path': str(registry),
                 'integrity_check': db.execute('PRAGMA integrity_check').fetchone()[0],
                 'foreign_key_check': db.execute('PRAGMA foreign_key_check').fetchall(),
                 'workers': [dict(r) for r in db.execute('SELECT w.*,o.identity FROM workers w JOIN owners o ON o.owner=w.owner')],
                 'controls': [dict(r) for r in db.execute('SELECT * FROM backend_controls')],
                 'routes': [dict(r) for r in db.execute('SELECT * FROM routes ORDER BY alias')],
                 'retained_counts': {t: db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in ['leases', 'lease_work', 'lease_tenants', 'admission_tickets', 'raw_work', 'recovery_claims']}}
    assert not any(state['retained_counts'].values()) and state['integrity_check'] == 'ok' and not state['foreign_key_check']
    assert all(r['state'] == 'QUIESCING' for r in state['controls'])
    p = run / 'worker-registry.json'
    assert not p.exists()
    p.write_text(json.dumps(state, indent=2) + '\n')
    for p in run.iterdir():
        if p.suffix in ('.json', '.log', '.sqlite') and p.name != 'keys.json':
            files[attempt['phase'] + '/' + p.name] = p
for p in (ROOT / 'runs').rglob('process.json'):
    identity = read(p)['identity']
    key = (identity['boot_id'], identity['pid'], identity['start_ticks'])
    if key not in seen:
        seen.add(key)
        records.append({'record': str(p), 'identity': identity})
for row in records:
    identity = row['identity']
    row['matching_live_process'] = process_identity(identity['pid']) == identity
    row['live_group'] = group_members(identity['pid'])
assert all(not r['matching_live_process'] and not r['live_group'] for r in records)
gpu = get_gpu(7)
assert gpu['uuid'] == 'GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib']) == 11990
model = ROOT / 'models/SmolLM2-1.7B-Instruct'
assert read(model / 'jev-source.json') == previous['model']
for before in previous['model_files']:
    p = model / before['name']
    a = p.stat(); digest = sha(p); b = p.stat()
    assert a.st_mtime_ns == b.st_mtime_ns and a.st_size == b.st_size == before['size_bytes'] and digest == before['sha256']
assert sha(HERE / 'source.tar.gz') == meta['archive_sha256']
assert all(not any(secret.encode() in p.read_bytes() for secret in secrets) for p in files.values())
native = ROOT / 'envs/vllm/lib/python3.12/site-packages/vllm'
ranges = {'engine/arg_utils.py': [(774, 784), (1765, 1777)], 'entrypoints/cli/serve.py': [(386, 419)], 'entrypoints/launchers/launcher.py': [(127, 164)], 'v1/utils.py': [(605, 670)], 'v1/engine/utils.py': [(35, 70)], 'v1/engine/core.py': [(1476, 1535)]}
excerpts = {}
for name, sections in ranges.items():
    p = native / name
    assert sha(p) == campaign['native_source_files'][name] == campaign['native_source_files_after'][name]
    lines = p.read_text().splitlines()
    excerpts[name] = {'sha256': sha(p), 'path': str(p), 'ranges': [{'first_line': first, 'last_line': last, 'text': '\n'.join(lines[first - 1:last])} for first, last in sections]}
p = HERE / 'native-shutdown-excerpts.json'
assert not p.exists()
p.write_text(json.dumps(excerpts, indent=2) + '\n'); files[p.name] = p
audit = {'passed': True, 'checked_at': time.time(), 'script_sha256': sha(Path(__file__)),
         'owned_records_checked': len(records), 'records': records, 'gpu7': gpu,
         'model': previous['model'], 'model_files': previous['model_files'],
         'model_validation': 'Current hashes match the previously Hub-verified immutable audit; no new Hub query',
         'prior_audit_sha256': sha(previous_path),
         'source_commit': meta['commit'], 'python_files': {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')},
         'credential_values_absent_from_export': True}
p = HERE / 'audit.json'; assert not p.exists()
p.write_text(json.dumps(audit, indent=2) + '\n'); files[p.name] = p
manifest = {'artifacts': {name: {'sha256': sha(p), 'size_bytes': p.stat().st_size} for name, p in files.items()}}
p = HERE / 'export-manifest.json'; assert not p.exists()
p.write_text(json.dumps(manifest, indent=2) + '\n'); files[p.name] = p
archive = HERE / 'evidence.tar.gz'; assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for name, p in sorted(files.items()):
        tar.add(p, arcname=name, recursive=False)
print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'size_bytes': archive.stat().st_size, 'owned_records_checked': len(records)}))
