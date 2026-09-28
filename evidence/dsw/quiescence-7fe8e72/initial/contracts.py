import json,os,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'quiescence-919c859';meta=json.loads((HERE/'sources.json').read_text());source=ROOT/'quiescence-sources'/meta['commit']
report={'source_commit':meta['commit'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time(),'qualification':'CPU engine doubles and HTTP/registry contract tests in the actual engine environments; no extra GPU execution','engines':{}}
names=['test_quiescence.py','test_quiescent_stop.py','test_recovery.py','test_recovery_mode.py','test_health.py','test_shared_admission.py','test_registry_backup.py']
assert not (HERE/'contracts.json').exists()
for engine in ['vllm','sglang']:
 env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','JEV_CALL_','JEV_CPU_'))};env['PYTHONPATH']=':'.join(map(str,[source,source/'src',source/'packages/vllm/src',source/'packages/sglang/src',ROOT/'combined-reservation-7391ed3/test-tools']))
 command=[str(ROOT/'envs'/engine/'bin/python'),'-m','pytest','-q',*[str(source/'tests'/name) for name in names]]
 with (HERE/('contracts-'+engine+'.log')).open('x') as f:rc=subprocess.run(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 report['engines'][engine]={'command':command,'returncode':rc};(HERE/'contracts.json').write_text(json.dumps(report,indent=2)+'\n');assert rc==0
report['complete']=True;report['finished_at']=time.time();(HERE/'contracts.json').write_text(json.dumps(report,indent=2)+'\n')
