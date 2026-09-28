import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'combined-reservation-7391ed3';meta=json.loads((HERE/'sources.json').read_text());commit=meta['after']['commit'];source=ROOT/'combined-reservation-sources'/commit
previous=json.loads((HERE/'campaign.json').read_text())
assert len(previous['attempts'])==4 and all(x['returncode']==0 for x in previous['attempts'])
assert not previous['complete'] and previous['linux_contracts_returncode']==1
assert 'No module named pytest' in (HERE/'linux-contracts.log').read_text()
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import get_gpu,process_identity
from tests.integration.run_native_validation import group_members
OUT=HERE/'quota-campaign.json';assert not OUT.exists();report={'source_commit':commit,'started_at':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'attempts':[]}
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
try:
 test_tools=HERE/'test-tools';assert not test_tools.exists()
 command=[str(ROOT/'envs/vllm/bin/python'),'-m','pip','install','--disable-pip-version-check','--no-input','--index-url','https://pypi.org/simple','--target',str(test_tools),'--report',str(HERE/'test-tools-install.json'),'pytest==9.1.1','pytest-asyncio==1.4.0']
 with (HERE/'test-tools-install.log').open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 env={**os.environ,'PYTHONPATH':':'.join(map(str,[source,source/'src',source/'packages/vllm/src',source/'packages/sglang/src',test_tools]))}
 command=[str(ROOT/'envs/vllm/bin/python'),'-m','pytest','-q',str(source/'tests/test_combined_reservation.py'),str(source/'tests/test_shared_admission.py'),str(source/'tests/test_lifecycle.py'),str(source/'tests/test_remote_cancel.py')]
 with (HERE/'linux-contracts-isolated.log').open('x') as f:report['linux_contracts_returncode']=subprocess.run(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 save();assert report['linux_contracts_returncode']==0
 for engine in ['vllm','sglang']:
  for p in (ROOT/'runs').rglob('process.json'):
   i=json.loads(p.read_text())['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid'])
  assert int(get_gpu(7)['free_mib'])>=11900
  item={'engine':engine,'tag':'quota','label':'quota-'+commit[:7],'source_commit':commit,'started_at':time.time()};report['attempts'].append(item);save()
  env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_'))};env.update(PYTHONPATH=':'.join(map(str,[source/'src',source/'packages/sglang/src',source/'packages/vllm/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  command=[str(ROOT/'envs'/engine/'bin/python'),str(HERE/'run_quota.py'),commit,engine];item['command']=command
  with (HERE/(engine+'-quota.log')).open('x') as f:
   child=subprocess.Popen(command,env=env,cwd=source,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);item['identity']=process_identity(child.pid);save();item['returncode']=child.wait(timeout=720)
  item['remaining_group']=group_members(child.pid);item['finished_at']=time.time();save();assert item['returncode']==0 and not item['remaining_group']
 report['complete']=True
except BaseException as e:
 report['complete']=False;report['failure']={'type':type(e).__name__,'message':str(e)};raise
finally:report['finished_at']=time.time();save()
