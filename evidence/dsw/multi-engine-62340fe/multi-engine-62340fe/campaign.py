import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import time

ROOT=Path('/root/jev-runtime')
HERE=ROOT/'multi-engine-62340fe'
SOURCE=HERE/'source'
COMMIT='62340feddb4d73274c0902d68cead1012da94624'
archives={'source.tar':'c93530fc6791155c1c54e3fa4ee05f5b792238268c0fb0bb41c0fbc387c4f1ed','upstream-contract.tar':'bfd1761a467af3edd448162c809335570d343eba2f1cf035501602f0d1882067'}
for name,digest in archives.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    destination=SOURCE if name=='source.tar' else HERE/'upstream-contract'
    destination.mkdir()
    with tarfile.open(HERE/name) as tar:tar.extractall(destination,filter='data')
sys.path[:0]=[str(SOURCE),str(SOURCE/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members

assert not (HERE/'campaign.json').exists()
report={'source_commit':COMMIT,'started_at':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'archives':archives,'supervisor_identity':process_identity(os.getpid()),'contracts':{},'native':{},'qualification':'CPU contracts plus simultaneous colocated SGLang/vLLM native GPU regression; TokenSpeed GPU not run'}
def save():
    pending=HERE/'campaign.pending'
    pending.write_text(json.dumps(report,indent=2)+'\n')
    pending.replace(HERE/'campaign.json')
save()
records=list((ROOT/'runs').rglob('process.json'))
for path in records:
    identity=json.loads(path.read_text())['identity']
    assert process_identity(identity['pid'])!=identity and not group_members(identity['pid']),str(path)
report['prior_owned_groups_terminal']=len(records)
report['gpus_before']=[get_gpu(i) for i in (6,7)]
for index,uuid,port in [(6,'GPU-b028ddfd-c58e-76d1-863b-21ec06f99dbe',18794),(7,'GPU-b57fb933-0e5a-dd28-7041-a177da03405e',18795)]:
    gpu=get_gpu(index)
    assert gpu['uuid']==uuid and gpu['free_mib']>=10000
    with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
with (HERE/'provider-install.log').open('x') as file:
    subprocess.run([str(ROOT/'envs/vllm/bin/python'),'-m','pip','install','--no-index','--no-deps','--target',str(HERE/'test-provider'),str(HERE/'jev_tokenspeed-0.1.0a1-py3-none-any.whl')],stdout=file,stderr=subprocess.STDOUT,check=True,timeout=60)
def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_CPU_','JEV_CALL_'))}
    env.update(PYTHONPATH=':'.join(map(str,[SOURCE,SOURCE/'src',SOURCE/'packages/vllm/src',SOURCE/'packages/sglang/src',HERE/'test-provider',ROOT/'combined-reservation-7391ed3/test-tools'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env
names=['test_backend_extensions.py','test_backends.py','test_compiler.py','test_scoring.py','test_shared_admission.py','test_quiescence.py','test_recovery.py','test_precision.py','test_execution_mode.py','test_plugin_lifespan.py']
for engine in ('vllm','sglang'):
    cmd=[str(ROOT/'envs'/engine/'bin/python'),'-m','pytest','-q',*[str(SOURCE/'tests'/name) for name in names]]
    with (HERE/('contracts-'+engine+'.log')).open('x') as file:
        rc=subprocess.run(cmd,env=environment(),cwd=SOURCE,stdout=file,stderr=subprocess.STDOUT,timeout=180).returncode
    report['contracts'][engine]={'returncode':rc,'command':cmd};save()
    assert rc==0
cmd=[str(ROOT/'envs/vllm/bin/python'),str(SOURCE/'tests/integration/verify_tokenspeed_readout.py'),'--source-root',str(HERE/'upstream-contract'),'--output',str(HERE/'tokenspeed-source-readout.json')]
with (HERE/'tokenspeed-source-readout.log').open('x') as file:
    rc=subprocess.run(cmd,env=environment(),cwd=SOURCE,stdout=file,stderr=subprocess.STDOUT,timeout=60).returncode
report['source_readout_returncode']=rc;save();assert rc==0

def native(engine,gpu,port,fraction):
    run=ROOT/'runs'/f'{engine}-multi-engine-62340fe'
    cmd=[str(ROOT/'envs'/engine/'bin/python'),str(SOURCE/'tests/integration/run_native_validation.py'),'--run-dir',str(run),'--engine',engine,'--model-path',str(ROOT/'models/SmolLM2-1.7B-Instruct'),'--gpus',str(gpu),'--port',str(port),'--memory-fraction',str(fraction),'--switches','1000','--source-commit',COMMIT,'--runtime-source-commit',COMMIT]
    with (HERE/(engine+'-native.log')).open('x') as file:
        rc=subprocess.run(cmd,env=environment(),cwd=SOURCE,stdout=file,stderr=subprocess.STDOUT,timeout=2400).returncode
    return {'returncode':rc,'command':cmd,'run_dir':str(run),'validation':json.loads((run/'validation.json').read_text())}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures={pool.submit(native,'vllm',7,18795,.07):'vllm',pool.submit(native,'sglang',6,18794,.65):'sglang'}
    for future in concurrent.futures.as_completed(futures):
        engine=futures[future]
        try: report['native'][engine]=future.result()
        except Exception as exc: report['native'][engine]={'runner_failure':repr(exc)}
        save()
report['gpus_after']=[get_gpu(i) for i in (6,7)]
report['finished_at']=time.time()
report['passed']=all(row.get('returncode')==0 and row['validation']['passed'] for row in report['native'].values()) and len(report['native'])==2
save()
print(json.dumps({k:report[k] for k in ('source_commit','passed','finished_at')}),flush=True)
raise SystemExit(0 if report['passed'] else 1)
