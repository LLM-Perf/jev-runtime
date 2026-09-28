"""Export stopped-state migration evidence, retaining immutable inputs and crash artifacts."""
import hashlib,json,sqlite3,sys,tarfile,time
from contextlib import closing
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'registry-migration-0ac5935'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
campaign=read(HERE/'campaign.json');meta=campaign['sources'];source=ROOT/'quiescence-sources'/meta['commit']
assert campaign['complete'] and read(HERE/'contracts.json')['complete'] and read(HERE/'faults.json')['passed']
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members
from jev_runtime.registry_backup import connect,summary,blockers
from jev_runtime.registry_schema import identify
files={name:HERE/name for name in ['campaign.json','campaign.log','sources.json','campaign.py','export.py','contracts.py','contracts.json','contracts.log','contracts-vllm.log','contracts-sglang.log','faults.py','faults.json','faults.log']}
previous_path=ROOT/'vllm-shutdown-e3f82d7/audit.json';previous=read(previous_path);records=previous['records'];seen={(r['identity']['boot_id'],r['identity']['pid'],r['identity']['start_ticks']) for r in records};secrets=[];databases=[]
for engine,entry in campaign['engines'].items():
    for a in entry['attempts']:
        assert a['complete'] and a['cleanup']['passed']
        run=Path(a['run_dir']);secrets.extend(v for v in read(run/'keys.json').values() if isinstance(v,str))
        record=read(run/'process.json');registry=Path(record['registry_path']);snapshot=run/'registry-final.sqlite';assert not snapshot.exists()
        with closing(connect(registry)) as db:
            with closing(sqlite3.connect(snapshot)) as target:db.backup(target);target.execute('PRAGMA journal_mode=DELETE')
            state={'format':identify(db),'summary':summary(db),'blockers':blockers(db)}
        assert not state['blockers']
        p=run/'registry-final.json';assert not p.exists();p.write_text(json.dumps(state,indent=2)+'\n')
        prefix=engine+'-'+a['phase']
        for p in run.iterdir():
            if p.suffix in ['.json','.log','.sqlite'] and p.name!='keys.json':files[prefix+'/'+p.name]=p
        if (run/'live-snapshot').exists():
            for p in (run/'live-snapshot').iterdir():files[prefix+'/live-snapshot/'+p.name]=p
        databases.append({'artifact':prefix+'/registry-final.sqlite','registry_path':str(registry),**state})
    for phase in ['upgrade','rollback']:
        for p in (HERE/(engine+'-'+phase+'-snapshot')).iterdir():files[engine+'-'+phase+'-snapshot/'+p.name]=p
        # These receipts bind the initial staged state, which was verified before
        # startup. The live database has since changed and is exported above as
        # registry-final.sqlite, never mislabeled as the initial staged payload.
        for name in ['migration-intent.json','migration.json']:
            files[engine+'-'+phase+'/'+name]=HERE/(engine+'-'+phase)/name
for name in ['fault-snapshot','fault-interrupted','fault-retry']:
    for p in (HERE/name).iterdir():
        assert p.is_file();files[name+'/'+p.name]=p
for p in (ROOT/'runs').rglob('process.json'):
    i=read(p)['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
    if key not in seen:seen.add(key);records.append({'record':str(p),'identity':i})
for r in records:
    i=r['identity'];r['matching_live_process']=process_identity(i['pid'])==i;r['live_group']=group_members(i['pid'])
assert all(not r['matching_live_process'] and not r['live_group'] for r in records)
gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])==11990
model=ROOT/'models/SmolLM2-1.7B-Instruct';assert read(model/'jev-source.json')==previous['model']
for before in previous['model_files']:
    p=model/before['name'];a=p.stat();h=sha(p);b=p.stat();assert a.st_mtime_ns==b.st_mtime_ns and a.st_size==b.st_size==before['size_bytes'] and h==before['sha256']
assert sha(HERE/'source.tar.gz')==meta['archive_sha256']
assert all(not any(secret.encode() in p.read_bytes() for secret in secrets) for p in files.values())
audit={'passed':True,'checked_at':time.time(),'script_sha256':sha(Path(__file__)),'owned_records_checked':len(records),'records':records,'gpu7':gpu,'databases':databases,'model':previous['model'],'model_files':previous['model_files'],'model_validation':'Current hashes match the earlier immutable Hub audit; no fresh Hub query','prior_audit_sha256':sha(previous_path),'sources':{label:{'commit':commit,'python_files':{str(p.relative_to(ROOT/'quiescence-sources'/commit)):sha(p) for p in (ROOT/'quiescence-sources'/commit).rglob('*.py')}} for label,commit in [('baseline',meta['base_commit']),('candidate',meta['commit'])]},'credential_values_absent_from_export':True}
p=HERE/'audit.json';assert not p.exists();p.write_text(json.dumps(audit,indent=2)+'\n');files[p.name]=p
manifest={'artifacts':{n:{'sha256':sha(p),'size_bytes':p.stat().st_size} for n,p in files.items()}}
p=HERE/'export-manifest.json';assert not p.exists();p.write_text(json.dumps(manifest,indent=2)+'\n');files[p.name]=p
archive=HERE/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for n,p in sorted(files.items()):tar.add(p,arcname=n,recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'owned_records_checked':len(records),'artifacts':len(manifest['artifacts'])}))
