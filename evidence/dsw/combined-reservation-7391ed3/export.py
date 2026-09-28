import hashlib,json,sqlite3,sys,tarfile,time
from pathlib import Path
from huggingface_hub import HfApi
ROOT=Path('/root/jev-runtime');HERE=ROOT/'combined-reservation-7391ed3'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
campaign=read(HERE/'campaign.json');meta=campaign['sources'];assert not campaign['complete'] and campaign['linux_contracts_returncode']==1;quota=read(HERE/'quota-campaign.json');assert quota['complete']
source=ROOT/'combined-reservation-sources'/meta['after']['commit'];sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members
files={};secrets=[];records=campaign['preflight']['records'];seen={(r['identity']['boot_id'],r['identity']['pid'],r['identity']['start_ticks']) for r in records}
for name in ['campaign.json','campaign.log','linux-contracts.log','sources.json','run_perf.py','campaign.py','export.py','quota-campaign.json','quota-campaign.log','quota_campaign.py','run_quota.py','linux-contracts-isolated.log','test-tools-install.log','test-tools-install.json']:files[name]=HERE/name
for item in campaign['attempts']+quota['attempts']:
 assert item['returncode']==0
 name=item['engine']+'-'+item['tag'];run=ROOT/'runs'/(item['engine']+'-combined-'+item['label']);files[name+'/wrapper.log']=HERE/(name+'.log')
 record=read(run/'process.json');i=item['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
 if key not in seen:seen.add(key);records.append({'record':name+'/wrapper.log','identity':i})
 def strings(value):
  if isinstance(value,str):return [value]
  if isinstance(value,dict):return [s for v in value.values() for s in strings(v)]
  return []
 secrets.extend(s for s in strings(read(run/'keys.json')) if s)
 with sqlite3.connect(f'file:{run}/registry.db?mode=ro',uri=True) as db:
  db.row_factory=sqlite3.Row
  state={'observed_at':time.time(),'workers':[dict(r) for r in db.execute('SELECT w.*,o.identity FROM workers w JOIN owners o ON o.owner=w.owner')],'prepared':[dict(r) for r in db.execute('SELECT owner,ref,digest FROM worker_bundles')],'retained_counts':{t:db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ['leases','lease_work','lease_tenants','admission_tickets']}}
 assert set(state['retained_counts'].values())=={0} and len(state['workers'])==2
 snap=run/'worker-registry.json';assert not snap.exists();snap.write_text(json.dumps(state,indent=2)+'\n')
 for p in run.iterdir():
  if p.suffix in ['.json','.log'] and p.name not in ['keys.json','config.json']:files[name+'/'+p.name]=p
 for case in ['c1','c16']:
  for p in (run/case).glob('*'):
   if p.suffix in ['.json','.jsonl']:files[name+'/'+case+'/'+p.name]=p
for p in (ROOT/'runs').rglob('process.json'):
 i=read(p)['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
 if key not in seen:seen.add(key);records.append({'record':str(p),'identity':i})
for r in records:
 i=r['identity'];r['matching_live_process']=process_identity(i['pid'])==i;r['live_group']=group_members(i['pid'])
assert all(not r['matching_live_process'] and not r['live_group'] for r in records)
gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])==11990
model=ROOT/'models/SmolLM2-1.7B-Instruct';identity=read(model/'jev-source.json');info=HfApi().model_info(identity['model_id'],revision=identity['revision'],files_metadata=True,token=False);assert info.sha==identity['revision'];remote={s.rfilename:s for s in info.siblings};model_files=[]
for p in sorted(list(model.glob('*.safetensors'))+[model/n for n in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json'] if (model/n).exists()]):
 before=p.stat();h=sha(p);after=p.stat();r=remote[p.name];assert before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size==r.size
 if r.lfs is not None:assert h==r.lfs.sha256
 else:assert hashlib.sha1(b'blob '+str(after.st_size).encode()+b'\0'+p.read_bytes()).hexdigest()==r.blob_id
 model_files.append({'name':p.name,'sha256':h,'size_bytes':after.st_size,'hub_digest_verified':True})
sources={}
for tag,item in meta.items():
 src=ROOT/'combined-reservation-sources'/item['commit'];assert sha(HERE/(tag+'.tar.gz'))==item['sha256']
 sources[tag]={'commit':item['commit'],'archive_sha256':item['sha256'],'python_files':{str(p.relative_to(src)):sha(p) for p in src.rglob('*.py')}}
assert all(not any(s.encode() in p.read_bytes() for s in secrets) for p in files.values())
audit={'passed':True,'checked_at':time.time(),'script_sha256':sha(Path(__file__)),'owned_records_checked':len(records),'records':records,'gpu7':gpu,'model':identity,'model_files':model_files,'sources':sources,'credential_values_absent_from_export':True}
p=HERE/'audit.json';assert not p.exists();p.write_text(json.dumps(audit,indent=2)+'\n');files['audit.json']=p
manifest={'artifacts':{n:{'sha256':sha(p),'size_bytes':p.stat().st_size} for n,p in files.items()}}
p=HERE/'export-manifest.json';assert not p.exists();p.write_text(json.dumps(manifest,indent=2)+'\n');files['export-manifest.json']=p
archive=HERE/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for n,p in sorted(files.items()):tar.add(p,arcname=n,recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'owned_records_checked':len(records)}))
