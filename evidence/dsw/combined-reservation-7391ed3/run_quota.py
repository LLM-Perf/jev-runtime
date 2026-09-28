import hashlib, importlib, json, subprocess, sys, time
from pathlib import Path

source_id, engine = sys.argv[1:]
root=Path('/root/jev-runtime');source=root/'combined-reservation-sources'/source_id
sys.path.insert(0,str(source))
from tests.integration.run_native_validation import save,wait_ready,topology,postcheck,cleanup
run=root/'runs'/f'{engine}-combined-quota-{source_id[:7]}'
run.mkdir(exist_ok=False)
limits=dict(max_requests=2,max_tokens=524288,max_queue=2,max_tenant_requests=1,
            max_tenant_tokens=262144,max_tenant_queue=1,max_branches=129,max_tenant_branches=128)
save(run/'limits.json',limits)
report={'source_commit':source_id,'engine':engine,'started_at':time.time(),
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stages':{},
        'qualification':'colocated shared quota and cancellation validation'}
record=None
try:
    paths={name:importlib.import_module(name).__file__ for name in ('jev_runtime','jev_'+engine)}
    assert all('/combined-reservation-sources/'+source_id+'/' in p for p in paths.values()),paths
    report['import_paths']=paths
    cmd=[sys.executable,str(source/'deployment/dsw_service.py'),'launch','--run-dir',str(run),
         '--engine',engine,'--model-path',str(root/'models/SmolLM2-1.7B-Instruct'),'--gpus','7',
         '--port','18795' if engine=='vllm' else '18794','--memory-fraction','.07' if engine=='vllm' else '.65',
         '--api-workers','2','--tenant','alpha','--tenant','beta','--admission-config',str(run/'limits.json')]
    with (run/'launch.log').open('x') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=45)
    record=json.loads((run/'process.json').read_text());keys=json.loads((run/'keys.json').read_text())
    report['stages']['ready']=wait_ready(record,keys['api'],300)
    save(run/'topology.json',topology(run,record))
    cmd=[sys.executable,str(source/'tests/integration/live_shared_admission.py'),
         '--run-dir',str(run),'--output',str(run/'quota.json'),'--source-commit',source_id]
    with (run/'quota.log').open('x') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    assert json.loads((run/'quota.json').read_text())['passed'] is True
    report['stages']['postcheck']=postcheck(run,record,keys['admin'],source_id)
    report['passed']=True
except BaseException as exc:
    report['passed']=False;report['failure']={'type':type(exc).__name__,'message':str(exc)}
finally:
    if record is None and (run/'process.json').exists():record=json.loads((run/'process.json').read_text())
    if record:
        report['stages']['cleanup']=cleanup(run,record)
        report['passed']=report.get('passed',False) and report['stages']['cleanup']['passed']
    report['finished_at']=time.time();save(run/'quota-attempt.json',report)
    print(json.dumps(report),flush=True)
if not report['passed']:sys.exit(1)
