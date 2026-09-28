import hashlib,importlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'asgi-profile-a475ad4';COMMIT='7391ed362117ab39a32bfa1dd1554356ee7a6178';source=ROOT/'combined-reservation-sources'/COMMIT
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import cleanup,group_members,wait_ready,topology,postcheck
OUT=HERE/'campaign.json';assert not OUT.exists()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'runtime_source_commit':COMMIT,'source_equivalent_to_delivery_commit':'a475ad425b65564d5c93de3d485b2633b24f2763','script_sha256':sha(Path(__file__)),'hook_sha256':sha(HERE/'bootstrap/sitecustomize.py'),'modified_runner_sha256':sha(HERE/'run_case_profile.py'),'started_at':time.time(),'attempts':[],'qualification':'Bounded ASGI thread-CPU profiling. Instrumentation changes timings; no unprofiled throughput or latency claim.'}
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
 for engine in ['vllm','sglang']:
  run=ROOT/'runs'/f'{engine}-asgi-profile-a475ad4';run.mkdir(exist_ok=False)
  item={'engine':engine,'run_dir':str(run),'started_at':time.time(),'preflight':preflight()};report['attempts'].append(item);save();record=None
  ep=ROOT/'envs'/engine/'bin/python';env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_'))}
  env.update(PYTHONPATH=':'.join(map(str,[HERE/'bootstrap',source/'src',source/'packages/vllm/src',source/'packages/sglang/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',JEV_CPU_PROFILE_DIR=str(run/'cpu-profiles'))
  cmd=[str(ep),str(source/'deployment/dsw_service.py'),'launch','--run-dir',str(run),'--engine',engine,'--model-path',str(ROOT/'models/SmolLM2-1.7B-Instruct'),'--gpus','7','--port','18795' if engine=='vllm' else '18794','--memory-fraction','.07' if engine=='vllm' else '.65','--api-workers','2']
  item['command']=cmd
  try:
   with (run/'launch.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
   record=read(run/'process.json');keys=read(run/'keys.json');item['ready']=wait_ready(record,keys['api'],420);save()
   (run/'topology.json').write_text(json.dumps(topology(run,record),indent=2)+'\n')
   bench=[str(ep),str(HERE/'run_case_profile.py'),'--run-dir',str(run),'--output',str(run/'case'),'--source-commit',COMMIT,'--runtime-source-commit',COMMIT,'--context-tokens','256','--candidates','8','--concurrency','1','--cache','hot','--runtime-timing','--input-variants','1','--duration','10','--repeats','1','--warmup','32']
   benvenv={k:v for k,v in env.items() if k!='JEV_CPU_PROFILE_DIR'}
   with (run/'case.log').open('x') as f:subprocess.run(bench,env=benvenv,cwd=source,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
   profiles=[read(p) for p in (run/'cpu-profiles').glob('*.json')];assert len(profiles)==2
   assert {p['lane'] for p in profiles}=={'native-label','native-plugin'}
   for p in profiles:assert p['profiled_requests']==128 and p['statuses']=={'200':128} and p['failed_requests']==0 and p['nested_or_concurrent_skips']==0 and p['hook_sha256']==report['hook_sha256']
   item['profiles']=[{k:p[k] for k in ['pid','lane','profiled_requests','profiler_total_cpu_seconds','wall_seconds']} for p in profiles]
   # postcheck imports source using the selected engine Python and runtime paths.
   command=[str(ep),'-c',"import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from tests.integration.run_native_validation import postcheck;r=Path(sys.argv[2]);print(json.dumps(postcheck(r,json.loads((r/'process.json').read_text()),json.loads((r/'keys.json').read_text())['admin'],sys.argv[3])))",str(source),str(run),COMMIT]
   item['postcheck']=json.loads(subprocess.check_output(command,env=benvenv,text=True));item['complete']=True
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
