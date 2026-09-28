import json,os,sqlite3,subprocess,sys,time,hashlib
from pathlib import Path
import httpx
ROOT=Path('/root/jev-runtime');HERE=ROOT/'asgi-profile-a475ad4';COMMIT='7391ed362117ab39a32bfa1dd1554356ee7a6178';source=ROOT/'combined-reservation-sources'/COMMIT
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import get_gpu,process_identity
from tests.integration.run_native_validation import cleanup,group_members,require_owner
old=ROOT/'runs/sglang-asgi-profile-a475ad4';registry=old/'registry.db';run=ROOT/'runs/sglang-asgi-recovery-a475ad4';run.mkdir(exist_ok=False)
report={'source_commit':COMMIT,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time(),'run_dir':str(run),'original_registry':str(registry)}
OUT=HERE/'recovery.json'
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
def view():
 with sqlite3.connect(f'file:{registry}?mode=ro',uri=True) as db:
  db.row_factory=sqlite3.Row
  return {t:[dict(r) for r in db.execute('SELECT * FROM '+t)] for t in ['leases','lease_work','events']}
def backup(name):
 path=HERE/name
 with sqlite3.connect(f'file:{registry}?mode=ro',uri=True) as src,sqlite3.connect(path) as dst:src.backup(dst)
 return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
record=None
try:
 for p in (ROOT/'runs').rglob('process.json'):
  i=json.loads(p.read_text())['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid'])
 assert int(get_gpu(7)['free_mib'])==11990
 report['before']=view();assert len(report['before']['leases'])==2;report['before_snapshot']=backup('recovery-before.sqlite');save()
 env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_CPU_'))};env.update(PYTHONPATH=':'.join(map(str,[source/'src',source/'packages/sglang/src',source/'packages/vllm/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 command=[str(ROOT/'envs/sglang/bin/python'),str(source/'deployment/dsw_service.py'),'launch','--run-dir',str(run),'--engine','sglang','--model-path',str(ROOT/'models/SmolLM2-1.7B-Instruct'),'--gpus','7','--port','18794','--memory-fraction','.65','--registry-path',str(registry),'--recovery-only','--no-bootstrap']
 with (run/'launch.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
 record=json.loads((run/'process.json').read_text());keys=json.loads((run/'keys.json').read_text());start=time.monotonic()
 with httpx.Client(base_url='http://127.0.0.1:18794/plugins/jev-runtime',headers={'Authorization':'Bearer '+keys['admin']},timeout=15,trust_env=False) as c:
  while time.monotonic()-start<420:
   require_owner(record)
   try:
    p=c.get('/admin/profile')
    if p.status_code==200:break
   except httpx.TransportError:pass
   time.sleep(1)
  else:raise TimeoutError('Recovery profile unavailable')
  assert p.json()['recovery_only'];report['profile']=p.json();pending=c.get('/admin/requests/recovery');pending.raise_for_status();rows=pending.json()['requests'];assert len(rows)==2 and all(x['recoverable'] and x['owner_status']=='dead' for x in rows);report['candidates']=rows
  report['recovered']=[]
  for row in rows:
   response=c.post('/admin/requests/'+row['request_id']+'/recover');response.raise_for_status();assert response.json()=={'recovered':True};report['recovered'].append({'request_id':row['request_id'],'response':response.json()});save()
  ready=c.get('/ready',headers={'Authorization':'Bearer '+keys['api']});assert ready.status_code==503
  assert c.get('/admin/requests/recovery').json()['requests']==[]
 report['after']=view();assert not report['after']['leases'] and not report['after']['lease_work'];report['after_snapshot']=backup('recovery-after.sqlite');report['complete']=True
except BaseException as e:
 report['complete']=False;report['failure']={'type':type(e).__name__,'message':str(e)};raise
finally:
 if record is None and (run/'process.json').exists():record=json.loads((run/'process.json').read_text())
 if record:report['cleanup']=cleanup(run,record)
 report['finished_at']=time.time();save()
assert report['cleanup']['passed']
