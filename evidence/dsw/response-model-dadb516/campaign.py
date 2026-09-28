import hashlib,json,os,sqlite3,subprocess,sys,tarfile,time
from pathlib import Path
import httpx
ROOT=Path('/root/jev-runtime');HERE=ROOT/'response-model-dadb516';meta=json.loads((HERE/'sources.json').read_text());before=ROOT/'combined-reservation-sources'/meta['before'];after=ROOT/'response-model-sources'/meta['after']
assert hashlib.sha256((HERE/'source.tar.gz').read_bytes()).hexdigest()==meta['after_archive_sha256'];after.mkdir(parents=True,exist_ok=False)
with tarfile.open(HERE/'source.tar.gz') as tar:tar.extractall(after,filter='data')
sys.path[:0]=[str(before),str(before/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import cleanup,group_members,wait_ready,topology
OUT=HERE/'campaign.json';assert not OUT.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'sources':meta,'script_sha256':sha(Path(__file__)),'hook_sha256':sha(HERE/'bootstrap/sitecustomize.py'),'modified_runner_sha256':sha(HERE/'run_case_profile.py'),'started_at':time.time(),'attempts':[],'qualification':'Matched colocated C1 development measurements, with separately instrumented continuous-window call profiles. No release certification.'}
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
def preflight():
 records=read(ROOT/'combined-reservation-7391ed3/audit.json')['records']
 for r in records:
  i=r['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid'])
 for p in (ROOT/'runs').rglob('process.json'):
  i=read(p)['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid']),str(p)
 gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])>=11900
 return {'gpu7':gpu,'previous_retained_records':len(records),'all_current_engine_records_terminal':True}
try:
 recovery=read(ROOT/'asgi-profile-a475ad4/recovery.json');assert recovery['complete'] and recovery['cleanup']['passed']
 for engine in ['vllm','sglang']:
  for tag,source in [('before',before),('after',after)]:
   commit=meta[tag];run=ROOT/'runs'/f'{engine}-response-{tag}-dadb516';run.mkdir(exist_ok=False)
   health=run/'health-settings.json';health.write_text(json.dumps({'interval_seconds':300,'timeout_seconds':10,'max_age_seconds':900}))
   item={'engine':engine,'tag':tag,'runtime_source_commit':commit,'run_dir':str(run),'started_at':time.time(),'preflight':preflight()};report['attempts'].append(item);save();record=None
   ep=ROOT/'envs'/engine/'bin/python';env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_CPU_','JEV_CALL_'))}
   env.update(PYTHONPATH=':'.join(map(str,[HERE/'bootstrap',source/'src',source/'packages/vllm/src',source/'packages/sglang/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',JEV_CALL_PROFILE_DIR=str(run/'call-profiles'))
   command=[str(ep),str(before/'deployment/dsw_service.py'),'launch','--run-dir',str(run),'--engine',engine,'--model-path',str(ROOT/'models/SmolLM2-1.7B-Instruct'),'--gpus','7','--port','18795' if engine=='vllm' else '18794','--memory-fraction','.07' if engine=='vllm' else '.65','--api-workers','2','--health-config',str(health)]
   item['command']=command
   try:
    with (run/'launch.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
    record=read(run/'process.json');keys=read(run/'keys.json');item['ready']=wait_ready(record,keys['api'],420);save()
    (run/'topology.json').write_text(json.dumps(topology(run,record),indent=2)+'\n')
    benvenv={k:v for k,v in env.items() if k!='JEV_CALL_PROFILE_DIR'}
    for mode in ['profile','measurement']:
     script=HERE/'run_case_profile.py' if mode=='profile' else before/'benchmarks/run_case.py'
     command=[str(ep),str(script),'--run-dir',str(run),'--output',str(run/mode),'--source-commit',meta['before'],'--runtime-source-commit',commit,'--context-tokens','256','--candidates','8','--concurrency','1','--cache','hot','--runtime-timing','--input-variants','1','--duration','10','--repeats','1' if mode=='profile' else '3','--warmup','32']
     with (run/(mode+'.log')).open('x') as f:subprocess.run(command,env=benvenv,cwd=before,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
     item[mode+'_complete']=True;save()
    profiles=[read(p) for p in (run/'call-profiles').glob('*.json')];assert len(profiles)==2 and {p['lane'] for p in profiles}=={'native-label','native-plugin'}
    for p in profiles:assert p['profiled_requests']==128 and p['statuses']=={'200':128} and p['failed_requests']==0 and p['nested_or_concurrent_skips']==0 and p['timing_sanity_passed'] and p['hook_sha256']==report['hook_sha256']
    item['profiles']=[{k:p[k] for k in ['pid','lane','profiled_requests','profiler_total_seconds','window_wall_seconds']} for p in profiles]
    # Run source/import postchecks in the selected engine environment.
    code="import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from tests.integration.run_native_validation import postcheck;r=Path(sys.argv[2]);print(json.dumps(postcheck(r,json.loads((r/'process.json').read_text()),json.loads((r/'keys.json').read_text())['admin'],sys.argv[3])))"
    item['postcheck']=json.loads(subprocess.check_output([str(ep),'-c',code,str(before),str(run),commit],env=benvenv,text=True));save()
    # Quiesce this isolated service's aliases before terminating the host engine,
    # so its background canaries cannot race a process-group SIGTERM.
    with httpx.Client(base_url=f"http://127.0.0.1:{record['port']}/plugins/jev-runtime",headers={'Authorization':'Bearer '+keys['admin']},timeout=15,trust_env=False) as c:
     listing=c.get('/admin/bundles');listing.raise_for_status();disabled=[]
     for route in listing.json()['routes']:
      if route['ref']:
       response=c.post('/admin/bundles/disable',json={'alias':route['alias'],'expected_generation':route['generation']});response.raise_for_status();disabled.append(response.json())
     deadline=time.monotonic()+30
     while True:
      listing=c.get('/admin/bundles');listing.raise_for_status()
      if not listing.json()['leases']:break
      if time.monotonic()>deadline:raise TimeoutError('Quiesce did not drain leases')
      time.sleep(.1)
     item['quiesce']={'disabled_routes':disabled,'leases':listing.json()['leases']}
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
