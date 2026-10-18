import gzip
import hashlib
import json
import sqlite3
import sys
import tarfile
from pathlib import Path

ROOT = Path('/root/jev-runtime')
OUT = ROOT/'strict-traffic-evidence-9da1813'
FINAL = '9da181342326793f6edddbd3f3ada4b0f7dbb320'
FIRST = '4ee78cab97c62d3ae03c5486e628af5bee797a28'
source = ROOT/'strict-traffic-9da1813'/FINAL
sys.path[:0] = [str(source),str(source/'src')]
from deployment.dsw_service import process_identity
from tests.integration.run_native_validation import group_members
from tests.integration.verify_traffic_evidence import verify

assert json.loads((ROOT/'strict-traffic-9da1813/campaign.json').read_text())['passed']
assert not json.loads((ROOT/'strict-traffic-4ee78ca/campaign.json').read_text())['passed']
OUT.mkdir()
secret_values = []
for suffix in ('4ee78ca', '9da1813'):
    for engine in ('vllm','sglang'):
        run = ROOT/'runs'/f'{engine}-strict-traffic-{suffix}'
        secret_values.extend(value.encode() for value in json.loads((run/'keys.json').read_text()).values())

def copy(path, relative):
    raw = path.read_bytes()
    clear = gzip.decompress(raw) if path.suffix == '.gz' else raw
    assert not any(value in clear for value in secret_values), str(path)
    target = OUT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix == '.log':
        target = target.with_suffix('.log.gz')
        target.write_bytes(gzip.compress(raw,mtime=0))
    else:
        target.write_bytes(raw)

sources = {}
counts = {}
for short,commit in [('4ee78ca',FIRST),('9da1813',FINAL)]:
    campaign = ROOT/f'strict-traffic-{short}'
    for path in campaign.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.py','.sha256'):
            copy(path,Path(campaign.name)/path.name)
    source_hashes = {}
    with tarfile.open(campaign/'source.tar') as archive:
        for member in archive.getmembers():
            if member.isfile():
                digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                assert hashlib.sha256((campaign/commit/member.name).read_bytes()).hexdigest()==digest
                source_hashes[member.name]=digest
    sources[commit]=source_hashes
    for engine in ('vllm','sglang'):
        run=ROOT/'runs'/f'{engine}-strict-traffic-{short}'
        record=json.loads((run/'process.json').read_text())
        identity=record['identity']
        assert process_identity(identity['pid'])!=identity and not group_members(identity['pid'])
        validation=json.loads((run/'validation.json').read_text())
        assert validation['cleanup_passed']
        for path in run.iterdir():
            if path.is_file() and path.name!='keys.json' and path.suffix in ('.json','.log','.gz'):
                copy(path,Path('runs')/run.name/path.name)
        db=sqlite3.connect(f'file:{run}/registry.db?mode=ro',uri=True)
        journal={table:db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                 for table in ('leases','lease_work','lease_tenants','admission_tickets','raw_work','recovery_claims')}
        db.close()
        assert all(value==0 for value in journal.values()),(run,journal)
        (OUT/'runs'/run.name/'final-journals.json').write_text(json.dumps(journal,indent=2)+'\n')
        if short=='9da1813':
            counts[engine]=verify(run/'traffic-responses.jsonl.gz',run/'contract.json')
            assert counts[engine]['strict_success_requests']>=10000
            (OUT/'runs'/run.name/'recount.json').write_text(json.dumps(counts[engine],indent=2)+'\n')
(OUT/'source-files.json').write_text(json.dumps(sources,indent=2)+'\n')
files={str(p.relative_to(OUT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
       for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'manifest.json').write_text(json.dumps({'final_source_commit':FINAL,'failed_attempt_source_commit':FIRST,
 'credential_scan':'API/admin exact values absent from plaintext and decompressed logs; keys.json excluded',
 'files':files},indent=2)+'\n')
archive=ROOT/'strict-traffic-evidence-9da1813.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname='strict-traffic-9da1813')
print(json.dumps({'files':len(files),'source_files':{k:len(v) for k,v in sources.items()},
 'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'recount':counts},indent=2))
