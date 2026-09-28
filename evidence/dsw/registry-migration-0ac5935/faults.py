"""Kill only a migration subprocess immediately before final database publication."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'registry-migration-0ac5935'
def read(p):return json.loads(p.read_text())
meta=read(HERE/'sources.json');source=ROOT/'quiescence-sources'/meta['commit']
sys.path[:0]=[str(source),str(source/'src')]
from jev_runtime.registry import Registry
from jev_runtime.registry_backup import snapshot
from jev_runtime.registry_migration import inspect,stage_migration,verify_migration
campaign=read(HERE/'campaign.json');assert campaign['complete']
original=Path(campaign['engines']['vllm']['original_registry'])
before=inspect(original);assert not before['migration_blockers']
backup=HERE/'fault-snapshot';snap=snapshot(original,backup)
destination=HERE/'fault-interrupted';OUT=HERE/'faults.json';assert not OUT.exists()
env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_CPU_','JEV_CALL_'))}
env['PYTHONPATH']=':'.join(map(str,[source,source/'src']))
code='''
import os,signal,sys
from pathlib import Path
from jev_runtime.registry_migration import stage_migration
real=os.link
def interrupted(a,b):
    if Path(b).name=='registry.sqlite3':
        os.kill(os.getpid(),signal.SIGKILL)
    return real(a,b)
os.link=interrupted
stage_migration(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),Path(sys.argv[4]))
'''
report={'source_commit':meta['commit'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':time.time(),'qualification':'Actual Linux migration-process SIGKILL; no engine crash or additional GPU execution','snapshot':snap}
try:
    child=subprocess.Popen([str(ROOT/'envs/vllm/bin/python'),'-c',code,str(backup),snap['manifest_sha256'],str(original),str(destination)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=child.communicate(timeout=30)
    report.update(child_pid=child.pid,returncode=child.returncode,stdout=stdout,stderr=stderr)
    assert child.returncode==-signal.SIGKILL
    assert (destination/'migration.json').is_file() and (destination/'registry.pending.sqlite3').is_file()
    assert not (destination/'registry.sqlite3').exists()
    try:Registry(destination/'registry.sqlite3')
    except ValueError as exc:
        assert 'incomplete' in str(exc);report['startup_rejection']=str(exc)
    else:raise AssertionError('Incomplete staging was started')
    assert not (destination/'registry.sqlite3').exists()
    receipt=(destination/'migration.json').read_bytes()
    try:verify_migration(destination,hashlib.sha256(receipt).hexdigest())
    except ValueError as exc:report['verification_rejection']=str(exc)
    else:raise AssertionError('Incomplete staging was verified')
    assert inspect(original)==before
    retry=HERE/'fault-retry';report['retry']=stage_migration(backup,snap['manifest_sha256'],original,retry)
    report['verified_retry']=verify_migration(retry,report['retry']['receipt_sha256'])
    assert inspect(original)==before
    report['source_unchanged']=True;report['passed']=True
except BaseException as exc:
    report['passed']=False;report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    report['finished_at']=time.time();OUT.write_text(json.dumps(report,indent=2)+'\n')
