import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'multi-engine-55674ec';SOURCE=HERE/'55674ecbe63850dc9c46da14893ffcad59d2506f'
archive=HERE/'source-55674ec.tar';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='674fea1f10440fb1b964e1b6f6e1ffd14ae9035b8f855a26ca16635a8d694b14'
SOURCE.mkdir()
with tarfile.open(archive) as tar:tar.extractall(SOURCE,filter='data')
with (HERE/'install.log').open('x') as file:
 subprocess.run([str(ROOT/'envs/vllm/bin/python'),'-m','pip','install','--no-index','--no-deps','--target',str(HERE/'test-provider'),str(HERE/'jev_tokenspeed-0.1.0a1-py3-none-any.whl')],stdout=file,stderr=subprocess.STDOUT,check=True,timeout=60)
report={'source_commit':SOURCE.name,'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time(),'scope':'CPU contracts in actual SGLang/vLLM Python environments; no TokenSpeed native engine or GPU execution','engines':{}}
names=['test_backend_extensions.py','test_backends.py','test_compiler.py','test_scoring.py','test_shared_admission.py','test_quiescence.py','test_recovery.py','test_precision.py','test_execution_mode.py','test_plugin_lifespan.py']
for engine in ('vllm','sglang'):
 env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_'))}
 env['PYTHONPATH']=':'.join(map(str,[SOURCE,SOURCE/'src',SOURCE/'packages/vllm/src',SOURCE/'packages/sglang/src',HERE/'test-provider',ROOT/'combined-reservation-7391ed3/test-tools']))
 cmd=[str(ROOT/'envs'/engine/'bin/python'),'-m','pytest','-q',*[str(SOURCE/'tests'/name) for name in names]]
 with (HERE/(engine+'.log')).open('x') as file:rc=subprocess.run(cmd,env=env,cwd=SOURCE,stdout=file,stderr=subprocess.STDOUT,timeout=180).returncode
 report['engines'][engine]={'command':cmd,'returncode':rc}
 (HERE/'contracts.json').write_text(json.dumps(report,indent=2)+'\n')
 assert rc==0
report['passed']=True;report['finished_at']=time.time();(HERE/'contracts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
