import hashlib,json,sqlite3,sys,tarfile,time,subprocess
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'response-model-dadb516';INITIAL=ROOT/'asgi-profile-a475ad4'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
campaign=read(HERE/'campaign.json');assert campaign['complete'];meta=campaign['sources'];before=ROOT/'combined-reservation-sources'/meta['before'];after=ROOT/'response-model-sources'/meta['after']
sys.path[:0]=[str(before),str(before/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members
frameworks={}
for engine in ['vllm','sglang']:
 code="import json,hashlib,importlib.metadata as m;import fastapi.routing,fastapi.encoders,fastapi._compat.v2;from pathlib import Path;print(json.dumps({'versions':{n:m.version(n) for n in ['fastapi','starlette','pydantic','pydantic-core']},'files':{x.__file__:hashlib.sha256(Path(x.__file__).read_bytes()).hexdigest() for x in [fastapi.routing,fastapi.encoders,fastapi._compat.v2]}}))"
 frameworks[engine]=json.loads(subprocess.check_output([str(ROOT/'envs'/engine/'bin/python'),'-c',code],text=True))
p=HERE/'frameworks.json';assert not p.exists();p.write_text(json.dumps({'observed_at':time.time(),'qualification':'Read-only final-audit framework versions and source hashes; no package changes made during this campaign','engines':frameworks},indent=2)+'\n')
files={'frameworks.json':p};secrets=[];old_audit=read(ROOT/'combined-reservation-7391ed3/audit.json');records=old_audit['records'];seen={(r['identity']['boot_id'],r['identity']['pid'],r['identity']['start_ticks']) for r in records}
for directory,prefix,names in [(HERE,'', ['campaign.json','campaign.log','sources.json','campaign.py','run_case_profile.py','export.py','contracts.py','contracts.json','contracts-vllm.log','contracts-sglang.log']), (INITIAL,'initial/', ['campaign.json','campaign.log','campaign.py','run_case_profile.py','recover.py','recovery.json','recovery.log','recovery-before.sqlite','recovery-after.sqlite'])]:
 for n in names:files[prefix+n]=directory/n
 files[prefix+'bootstrap/sitecustomize.py']=directory/'bootstrap/sitecustomize.py'
initial=read(INITIAL/'campaign.json');recovery=read(INITIAL/'recovery.json');assert not initial['complete'] and recovery['complete'] and recovery['cleanup']['passed']
assert len(recovery['candidates'])==2 and len(recovery['recovered'])==2 and not recovery['after']['leases'] and not recovery['after']['lease_work']
runs=[(x['engine']+'-'+x['tag'],Path(x['run_dir'])) for x in campaign['attempts']]+[('initial/'+x['engine'],Path(x['run_dir'])) for x in initial['attempts']]+[('initial/recovery',Path(recovery['run_dir']))]
def strings(value):
 if isinstance(value,str):return [value]
 if isinstance(value,dict):return [s for v in value.values() for s in strings(v)]
 return []
for name,run in runs:
 secrets.extend(s for s in strings(read(run/'keys.json')) if s)
 record=read(run/'process.json');registry=Path(record['registry_path'])
 with sqlite3.connect(f'file:{registry}?mode=ro',uri=True) as db:
  db.row_factory=sqlite3.Row
  state={'observed_at':time.time(),'registry_path':str(registry),'workers':[dict(r) for r in db.execute('SELECT w.*,o.identity FROM workers w JOIN owners o ON o.owner=w.owner')],'prepared':[dict(r) for r in db.execute('SELECT owner,ref,digest FROM worker_bundles')],'retained_counts':{t:db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ['leases','lease_work','lease_tenants','admission_tickets']}}
 assert set(state['retained_counts'].values())=={0}
 snap=run/'worker-registry.json';assert not snap.exists();snap.write_text(json.dumps(state,indent=2)+'\n')
 for p in run.iterdir():
  if p.suffix in ['.json','.log'] and p.name not in ['keys.json','config.json']:files[name+'/'+p.name]=p
 for sub in ['profile','measurement','case','call-profiles','cpu-profiles']:
  for p in (run/sub).glob('*'):
   if p.suffix in ['.json','.jsonl','.prof']:files[name+'/'+sub+'/'+p.name]=p
for p in (ROOT/'runs').rglob('process.json'):
 i=read(p)['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
 if key not in seen:seen.add(key);records.append({'record':str(p),'identity':i})
for r in records:
 i=r['identity'];r['matching_live_process']=process_identity(i['pid'])==i;r['live_group']=group_members(i['pid'])
assert all(not r['matching_live_process'] and not r['live_group'] for r in records)
gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])==11990
model=ROOT/'models/SmolLM2-1.7B-Instruct';assert read(model/'jev-source.json')==old_audit['model'];model_files=[]
for previous in old_audit['model_files']:
 p=model/previous['name'];a=p.stat();h=sha(p);b=p.stat();assert a.st_mtime_ns==b.st_mtime_ns and a.st_size==b.st_size==previous['size_bytes'] and h==previous['sha256']
 model_files.append({**previous,'unchanged_since_Hub_verified_audit':True})
sources={tag:{'commit':meta[tag],'python_files':{str(p.relative_to(src)):sha(p) for p in src.rglob('*.py')}} for tag,src in [('before',before),('after',after)]}
assert sha(HERE/'source.tar.gz')==meta['after_archive_sha256']
assert all(not any(s.encode() in p.read_bytes() for s in secrets) for p in files.values())
audit={'passed':True,'checked_at':time.time(),'script_sha256':sha(Path(__file__)),'owned_records_checked':len(records),'records':records,'gpu7':gpu,'model':old_audit['model'],'model_files':model_files,'prior_Hub_verified_audit_sha256':sha(ROOT/'combined-reservation-7391ed3/audit.json'),'sources':sources,'credential_values_absent_from_export':True}
p=HERE/'audit.json';assert not p.exists();p.write_text(json.dumps(audit,indent=2)+'\n');files['audit.json']=p
manifest={'artifacts':{n:{'sha256':sha(p),'size_bytes':p.stat().st_size} for n,p in files.items()}}
p=HERE/'export-manifest.json';assert not p.exists();p.write_text(json.dumps(manifest,indent=2)+'\n');files['export-manifest.json']=p
archive=HERE/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for n,p in sorted(files.items()):tar.add(p,arcname=n,recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'owned_records_checked':len(records)}))
