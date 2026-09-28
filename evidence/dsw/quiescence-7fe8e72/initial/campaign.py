import hashlib,json,os,sqlite3,subprocess,sys,tarfile,time
from pathlib import Path
import httpx
ROOT=Path('/root/jev-runtime');HERE=ROOT/'quiescence-919c859';meta=json.loads((HERE/'sources.json').read_text());source=ROOT/'quiescence-sources'/meta['commit']
assert hashlib.sha256((HERE/'source.tar.gz').read_bytes()).hexdigest()==meta['archive_sha256']
source.mkdir(parents=True,exist_ok=False)
with tarfile.open(HERE/'source.tar.gz') as tar:tar.extractall(source,filter='data')
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import cleanup,group_members,wait_ready,topology
from jev_runtime.schema import DecisionResponse
OUT=HERE/'campaign.json';assert not OUT.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'sources':meta,'script_sha256':sha(Path(__file__)),'started_at':time.time(),'attempts':[],'qualification':'Real GPU two-API-worker quiescence and preserved-route restart; colocated development profile only'}
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
def preflight():
 records=read(ROOT/'response-model-dadb516/audit.json')['records']
 for r in records:
  i=r['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid'])
 for p in (ROOT/'runs').rglob('process.json'):
  i=read(p)['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid']),str(p)
 gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])>=11900
 return {'gpu7':gpu,'prior_records':len(records),'all_owned_groups_terminal':True}
try:
 for engine in ['vllm','sglang']:
  env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_CPU_','JEV_CALL_'))}
  env.update(PYTHONPATH=':'.join(map(str,[source,source/'src',source/'packages/vllm/src',source/'packages/sglang/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  ep=ROOT/'envs'/engine/'bin/python';old=None;previous=None
  for phase in ['initial','restart']:
   run=ROOT/'runs'/f'{engine}-quiescence-{phase}-919c859';run.mkdir(exist_ok=False)
   health=run/'health-settings.json';health.write_text(json.dumps({'interval_seconds':1,'timeout_seconds':10,'max_age_seconds':30}))
   item={'engine':engine,'phase':phase,'run_dir':str(run),'started_at':time.time(),'preflight':preflight()};report['attempts'].append(item);save();record=None
   if phase=='restart':
    registry=Path(old['registry_path']);backend=previous['checks']['drained']['backend'];generation=previous['checks']['drained']['generation']
    command=[str(ep),'-c','from jev_runtime.cli import app;app()','registry','resume-backend',str(registry),backend,str(generation)]
    result=subprocess.check_output(command,env=env,text=True,timeout=30);item['resume']=json.loads(result);assert item['resume']['state']=='OPEN' and item['resume']['generation']==generation+1;save()
   command=[str(ep),str(source/'deployment/dsw_service.py'),'launch','--run-dir',str(run),'--engine',engine,'--model-path',str(ROOT/'models/SmolLM2-1.7B-Instruct'),'--gpus','7','--port','18795' if engine=='vllm' else '18794','--memory-fraction','.07' if engine=='vllm' else '.65','--api-workers','2','--health-config',str(health)]
   if phase=='restart':command.extend(['--registry-path',str(registry),'--no-bootstrap'])
   item['command']=command
   try:
    with (run/'launch.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
    record=read(run/'process.json');keys=read(run/'keys.json');item['ready']=wait_ready(record,keys['api'],420);save()
    (run/'topology.json').write_text(json.dumps(topology(run,record),indent=2)+'\n')
    if phase=='initial':
     command=[str(ep),str(source/'tests/integration/live_quiescence.py'),'--run-dir',str(run),'--output',str(run/'quiescence.json'),'--source-commit',meta['commit']]
     with (run/'check.log').open('x') as f:subprocess.run(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
     previous=read(run/'quiescence.json');assert previous['passed'];old=record
    else:
     base=f"http://127.0.0.1:{record['port']}/plugins/jev-runtime";admin={'Authorization':'Bearer '+keys['admin']};data={'Authorization':'Bearer '+keys['api']}
     seen={};responses=[]
     deadline=time.monotonic()+30
     while len(seen)!=2:
      assert time.monotonic()<deadline
      with httpx.Client(base_url=base,timeout=30,trust_env=False) as c:
       r=c.get('/admin/profile',headers=admin);r.raise_for_status();profile=r.json()
       if profile['worker_id'] in seen:continue
       assert profile['health']['monitor_running'];seen[profile['worker_id']]=profile
       for alias in ['decision-model','quiescence']:
        r=c.post('/v1/decisions',headers=data,json={'model':alias,'input':{'text':'Please refund the duplicate charge.'},'questions':[{'id':'q','type':'boolean','instruction':'Refund requested?'}]});r.raise_for_status();parsed=DecisionResponse.model_validate(r.json());assert parsed.status=='completed' and r.headers['x-jev-worker']==profile['worker_id'];responses.append(parsed.model_dump(mode='json'))
     with httpx.Client(base_url=base,timeout=30,trust_env=False) as c:
      r=c.get('/admin/bundles',headers=admin);r.raise_for_status();listing=r.json();assert listing['routes']==previous['checks']['routes_before'];item['restart_checks']={'profiles':seen,'responses':responses,'routes_unchanged':listing['routes']}
     assert not set(seen)&set(previous['checks']['workers'])
    item['complete']=True
   except BaseException as e:
    item['failure']={'type':type(e).__name__,'message':str(e)};raise
   finally:
    if record is None and (run/'process.json').exists():record=read(run/'process.json')
    if record:item['cleanup']=cleanup(run,record)
    item['finished_at']=time.time();save()
   assert item['cleanup']['passed']
 report['postflight']=preflight();report['complete']=True
except BaseException as e:
 report['complete']=False;report['failure']={'type':type(e).__name__,'message':str(e)};raise
finally:report['finished_at']=time.time();save()
