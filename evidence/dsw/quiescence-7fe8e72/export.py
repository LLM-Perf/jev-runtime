import hashlib,json,sqlite3,sys,tarfile,time,subprocess
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'quiescence-7fe8e72';INITIAL=ROOT/'quiescence-919c859'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
campaign=read(HERE/'campaign.json');assert campaign['complete'];meta=campaign['sources'];source=ROOT/'quiescence-sources'/meta['commit'];assert read(HERE/'contracts.json')['complete']
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members
files={name:HERE/name for name in ['campaign.json','campaign.log','sources.json','campaign.py','export.py','contracts.py','contracts.json','contracts.log','contracts-vllm.log','contracts-sglang.log']}
previous=read(ROOT/'response-model-dadb516/audit.json');records=previous['records'];seen={(r['identity']['boot_id'],r['identity']['pid'],r['identity']['start_ticks']) for r in records};secrets=[]
initial=read(INITIAL/'campaign.json');assert not initial['complete']
assert [a['cleanup']['passed'] for a in initial['attempts']]==[True,True,True,False]
remediation=read(Path(initial['attempts'][-1]['run_dir'])/'cleanup-remediation.json');assert remediation['cleanup_confirmed'] and not remediation['remaining_after']
for name in ['campaign.json','campaign.log','sources.json','campaign.py','export.py','contracts.py','contracts.json','contracts.log','contracts-vllm.log','contracts-sglang.log','remediate.py']:files['initial/'+name]=INITIAL/name
for prefix,attempt in [('',a) for a in campaign['attempts']]+[('initial/',a) for a in initial['attempts']]:
 assert attempt['complete']
 if not prefix:assert attempt['cleanup']['passed']
 run=Path(attempt['run_dir']);keys=read(run/'keys.json');secrets.extend(v for v in keys.values() if isinstance(v,str))
 record=read(run/'process.json');registry=Path(record['registry_path'])
 snapshot=run/'registry-final.sqlite';assert not snapshot.exists()
 with sqlite3.connect(f'file:{registry}?mode=ro',uri=True) as db:
  with sqlite3.connect(snapshot) as target:db.backup(target);target.execute('PRAGMA journal_mode=DELETE')
  db.row_factory=sqlite3.Row
  state={'observed_at':time.time(),'registry_path':str(registry),'integrity_check':db.execute('PRAGMA integrity_check').fetchone()[0],'foreign_key_check':db.execute('PRAGMA foreign_key_check').fetchall(),'workers':[dict(r) for r in db.execute('SELECT w.*,o.identity FROM workers w JOIN owners o ON o.owner=w.owner')],'controls':[dict(r) for r in db.execute('SELECT * FROM backend_controls')],'routes':[dict(r) for r in db.execute('SELECT * FROM routes ORDER BY alias')],'retained_counts':{t:db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ['leases','lease_work','lease_tenants','admission_tickets','raw_work','recovery_claims']}}
 assert set(state['retained_counts'].values())=={0} and state['integrity_check']=='ok' and not state['foreign_key_check']
 assert all(x['state']=='QUIESCING' for x in state['controls'])
 snap=run/'worker-registry.json';assert not snap.exists();snap.write_text(json.dumps(state,indent=2)+'\n')
 for p in run.iterdir():
  if p.suffix in ['.json','.log','.sqlite'] and p.name!='keys.json':files[prefix+attempt['engine']+'-'+attempt['phase']+'/'+p.name]=p
for p in (ROOT/'runs').rglob('process.json'):
 i=read(p)['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
 if key not in seen:seen.add(key);records.append({'record':str(p),'identity':i})
for r in records:
 i=r['identity'];r['matching_live_process']=process_identity(i['pid'])==i;r['live_group']=group_members(i['pid'])
assert all(not r['matching_live_process'] and not r['live_group'] for r in records)
gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])==11990
model=ROOT/'models/SmolLM2-1.7B-Instruct';assert read(model/'jev-source.json')==previous['model'];model_files=[]
for before in previous['model_files']:
 p=model/before['name'];a=p.stat();h=sha(p);b=p.stat();assert a.st_mtime_ns==b.st_mtime_ns and a.st_size==b.st_size==before['size_bytes'] and h==before['sha256'];model_files.append(before)
assert sha(HERE/'source.tar.gz')==meta['archive_sha256']
assert all(not any(s.encode() in p.read_bytes() for s in secrets) for p in files.values())
audit={'passed':True,'checked_at':time.time(),'script_sha256':sha(Path(__file__)),'owned_records_checked':len(records),'records':records,'gpu7':gpu,'model':previous['model'],'model_files':model_files,'model_validation':'Current hashes match the previously Hub-verified immutable audit; no new Hub query','prior_audit_sha256':sha(ROOT/'response-model-dadb516/audit.json'),'sources':{label:{'commit':commit,'python_files':{str(p.relative_to(ROOT/'quiescence-sources'/commit)):sha(p) for p in (ROOT/'quiescence-sources'/commit).rglob('*.py')}} for label,commit in [('initial',meta['base_commit']),('corrected',meta['commit'])]},'credential_values_absent_from_export':True}
p=HERE/'audit.json';assert not p.exists();p.write_text(json.dumps(audit,indent=2)+'\n');files['audit.json']=p
manifest={'artifacts':{n:{'sha256':sha(p),'size_bytes':p.stat().st_size} for n,p in files.items()}}
p=HERE/'export-manifest.json';assert not p.exists();p.write_text(json.dumps(manifest,indent=2)+'\n');files['export-manifest.json']=p
archive=HERE/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for n,p in sorted(files.items()):tar.add(p,arcname=n,recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'owned_records_checked':len(records)}))
