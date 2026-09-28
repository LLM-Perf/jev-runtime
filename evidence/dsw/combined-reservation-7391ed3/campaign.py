import hashlib,json,os,socket,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'combined-reservation-7391ed3';meta=json.loads((HERE/'sources.json').read_text())
OUT=HERE/'campaign.json';assert not OUT.exists()
report={'started_at':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':meta,'attempts':[]}
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for tag,item in meta.items():
 archive=HERE/(tag+'.tar.gz');assert sha(archive)==item['sha256']
 source=ROOT/'combined-reservation-sources'/item['commit'];source.mkdir(parents=True,exist_ok=False)
 with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
after=meta['after']['commit'];before=meta['before']['commit'];source=ROOT/'combined-reservation-sources'/after
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members

def preflight():
 records=[];seen=set()
 for p in (ROOT/'runs').rglob('process.json'):
  i=json.loads(p.read_text())['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
  if key in seen:continue
  seen.add(key);records.append({'record':str(p),'identity':i,'matching_live_process':process_identity(i['pid'])==i,'live_group':group_members(i['pid'])})
 # Preserve earlier non-engine owned identities (CPU seed and harness processes).
 old=json.loads((ROOT/'recovery-mode-audit-95de3d5.json').read_text())
 for entry in old['records']:
  i=entry['identity'];key=(i['boot_id'],i['pid'],i['start_ticks'])
  if key in seen:continue
  seen.add(key);records.append({'record':entry['record'],'identity':i,'matching_live_process':process_identity(i['pid'])==i,'live_group':group_members(i['pid'])})
 assert all(not r['matching_live_process'] and not r['live_group'] for r in records)
 gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])>=11900
 for port in [18794,18795]:
  with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
 return {'observed_at':time.time(),'owned_records_checked':len(records),'records':records,'gpu7':gpu}
try:
 report['preflight']=preflight();save()
 for engine in ['vllm','sglang']:
  for tag in ['before','after']:
   commit=meta[tag]['commit'];core=ROOT/'combined-reservation-sources'/commit
   env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_'))};env.update(PYTHONPATH=':'.join(map(str,[core/'src',core/'packages/sglang/src',core/'packages/vllm/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   label=tag+'-'+commit[:7];item={'engine':engine,'tag':tag,'source_commit':commit,'label':label,'started_at':time.time()};report['attempts'].append(item);save()
   item['preflight']=preflight();save()
   command=[str(ROOT/'envs'/engine/'bin/python'),str(HERE/'run_perf.py'),after,commit,engine,label]
   item['command']=command
   with (HERE/(engine+'-'+tag+'.log')).open('x') as f:
    child=subprocess.Popen(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);item['identity']=process_identity(child.pid);save()
    item['returncode']=child.wait(timeout=900)
   item['remaining_group']=group_members(child.pid);item['finished_at']=time.time();save()
   assert item['returncode']==0 and not item['remaining_group']
 # Linux process-identity retention tests use the same source, with a controlled engine double.
 env={**os.environ,'PYTHONPATH':':'.join(map(str,[source,source/'src',source/'packages/vllm/src',source/'packages/sglang/src']))}
 command=[str(ROOT/'envs/vllm/bin/python'),'-m','pytest','-q',str(source/'tests/test_combined_reservation.py'),str(source/'tests/test_shared_admission.py'),str(source/'tests/test_lifecycle.py'),str(source/'tests/test_remote_cancel.py')]
 with (HERE/'linux-contracts.log').open('x') as f:report['linux_contracts_returncode']=subprocess.run(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 assert report['linux_contracts_returncode']==0
 report['postflight']=preflight();report['complete']=True
except BaseException as e:
 report['complete']=False;report['failure']={'type':type(e).__name__,'message':str(e)};raise
finally:
 report['finished_at']=time.time();save()
