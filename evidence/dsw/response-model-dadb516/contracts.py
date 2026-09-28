import json,os,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'response-model-dadb516';meta=json.loads((HERE/'sources.json').read_text());source=ROOT/'response-model-sources'/meta['after']
assert json.loads((HERE/'campaign.json').read_text())['complete']
report={'source_commit':meta['after'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time(),'qualification':'CPU engine doubles and HTTP contract fixtures in each actual engine environment; not additional GPU execution','engines':{}}
for engine in ['vllm','sglang']:
 env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','JEV_CALL_','JEV_CPU_'))};env['PYTHONPATH']=':'.join(map(str,[source,source/'src',source/'packages/vllm/src',source/'packages/sglang/src',ROOT/'combined-reservation-7391ed3/test-tools']))
 command=[str(ROOT/'envs'/engine/'bin/python'),'-m','pytest','-q',*[str(source/'tests'/name) for name in ['test_response_serialization.py','test_api.py','test_telemetry.py','test_recovery_mode.py']]]
 with (HERE/('contracts-'+engine+'.log')).open('x') as f:rc=subprocess.run(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 report['engines'][engine]={'command':command,'returncode':rc};(HERE/'contracts.json').write_text(json.dumps(report,indent=2)+'\n');assert rc==0
report['complete']=True;report['finished_at']=time.time();(HERE/'contracts.json').write_text(json.dumps(report,indent=2)+'\n')
