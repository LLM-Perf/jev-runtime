"""Real two-worker native serving across explicit schema upgrade and code rollback."""
import hashlib,json,os,shutil,socket,sqlite3,subprocess,sys,tarfile,time
from contextlib import closing
from pathlib import Path
import httpx

ROOT=Path('/root/jev-runtime');HERE=ROOT/'registry-migration-0ac5935'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=read(HERE/'sources.json');old_source=ROOT/'quiescence-sources'/meta['base_commit'];source=ROOT/'quiescence-sources'/meta['commit']
assert sha(HERE/'source.tar.gz')==meta['archive_sha256']
shutil.copytree(old_source,source)
with tarfile.open(HERE/'source.tar.gz') as tar:tar.extractall(source,filter='data')
sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import cleanup,group_members,wait_ready,topology
from jev_runtime.registry import Registry
from jev_runtime.registry_backup import snapshot,connect,summary
from jev_runtime.registry_migration import stage_migration,verify_migration,inspect
from jev_runtime.schema import DecisionResponse

OUT=HERE/'campaign.json';assert not OUT.exists()
report={'sources':meta,'script_sha256':sha(Path(__file__)),'started_at':time.time(),'engines':{},'qualification':'Same-host offline schema maintenance with real native GPU serving, not zero-downtime or release certification'}
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
def environment(directory):
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','PIP_','JEV_CPU_','JEV_CALL_'))}
    env.update(PYTHONPATH=':'.join(map(str,[directory,directory/'src',directory/'packages/vllm/src',directory/'packages/sglang/src'])),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env
def preflight(port):
    records=list((ROOT/'runs').rglob('process.json'))
    for p in records:
        i=read(p)['identity'];assert process_identity(i['pid'])!=i and not group_members(i['pid']),str(p)
    gpu=get_gpu(7);assert gpu['uuid']=='GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib'])>=11900
    with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
    return {'gpu7':gpu,'owned_records_checked':len(records),'all_groups_terminal':True}
def resume(ep,directory,registry):
    with closing(connect(registry)) as db:
        row=db.execute('SELECT backend,generation,state FROM backend_controls').fetchone()
    assert row[2]=='QUIESCING'
    command=[str(ep),'-c','from jev_runtime.cli import app;app()','registry','resume-backend',str(registry),row[0],str(row[1])]
    result=json.loads(subprocess.check_output(command,env=environment(directory),text=True,timeout=30))
    assert result['state']=='OPEN' and result['generation']==row[1]+1
    return result
def reject_constructor(ep,directory,registry,expected):
    before=inspect(registry)['summary']
    command=[str(ep),'-c','from jev_runtime.registry import Registry; import sys; Registry(sys.argv[1])',str(registry)]
    r=subprocess.run(command,env=environment(directory),text=True,capture_output=True,timeout=30)
    assert r.returncode!=0 and expected in r.stderr
    assert inspect(registry)['summary']==before
    return {'source_commit':directory.name,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'state_unchanged':True}
def stage(registry,folder,target):
    backup=folder.with_name(folder.name+'-snapshot');snap=snapshot(registry,backup)
    result=stage_migration(backup,snap['manifest_sha256'],registry,folder,target)
    verified=verify_migration(folder,result['receipt_sha256'])
    return {'snapshot':snap,'migration':result,'verified':verified}

options=[{'id':'billing','description':'Billing and refunds'},{'id':'technical','description':'Technical support'}]
questions=[{'id':'boolean','type':'boolean','instruction':'Does the customer ask for a refund?'},
 {'id':'choice','type':'choice','instruction':'Choose the service category.','options':options},
 {'id':'score','type':'score','instruction':'Rate refund urgency.','options':[{'id':'low','description':'Low urgency','value':0.0},{'id':'high','description':'High urgency','value':1.0}]},
 {'id':'rank','type':'rank','instruction':'Rank the service categories by relevance.','options':options}]

def serving(run,record,phase,expected_generation):
    keys=read(run/'keys.json');admin={'Authorization':'Bearer '+keys['admin']};data={'Authorization':'Bearer '+keys['api']}
    base=f"http://127.0.0.1:{record['port']}/plugins/jev-runtime";clients={};profiles={};responses=[]
    result={'generation_on_entry':expected_generation}
    def call(c,path,body=None,headers=admin):
        r=c.get(path,headers=headers) if body is None else c.post(path,headers=headers,json=body)
        r.raise_for_status();return r.json()
    try:
        deadline=time.monotonic()+30
        while len(clients)!=2:
            assert time.monotonic()<deadline
            c=httpx.Client(base_url=base,timeout=30,trust_env=False,limits=httpx.Limits(max_connections=1,max_keepalive_connections=1,keepalive_expiry=None))
            profile=call(c,'/admin/profile');owner=profile['worker_id']
            if owner in clients:c.close();continue
            assert profile['health']['monitor_running']
            clients[owner]=c;profiles[owner]=profile
        first=next(iter(clients.values()));before=call(first,'/admin/bundles')
        assert len(before['routes'])==1
        route=before['routes'][0];assert route['alias']=='decision-model' and route['generation']==expected_generation
        result['routes_before']=before['routes']
        if phase!='rollback':
            call(first,'/admin/bundles/disable',{'alias':'decision-model','expected_generation':expected_generation})
            call(first,'/admin/bundles/activate',{'alias':'decision-model','reference':route['ref'],'expected_generation':expected_generation+1})
            expected_generation+=2
        result['routes_after']=call(first,'/admin/bundles')['routes']
        assert result['routes_after'][0]['generation']==expected_generation
        stale=first.post('/admin/bundles/activate',headers=admin,json={'alias':'decision-model','reference':route['ref'],'expected_generation':expected_generation-2})
        assert stale.status_code==409
        result['stale_generation_rejection']={'status':stale.status_code,'body':stale.json()}
        for owner,c in clients.items():
            for i in range(3):
                body={'model':'decision-model','request_id':f'{phase}-{owner}-{i}','input':{'text':'Please refund the duplicate charge on my account.'},'questions':questions}
                r=c.post('/v1/decisions',headers=data,json=body);r.raise_for_status();parsed=DecisionResponse.model_validate(r.json())
                assert parsed.status=='completed' and parsed.usage.successful_questions==4 and len(parsed.answers)==4
                assert r.headers['x-jev-worker']==owner
                responses.append(parsed.model_dump(mode='json'))
        result.update(profiles=profiles,responses=responses)
        if phase=='candidate':
            control=call(first,'/admin/quiescence')
            drained=call(first,'/admin/quiescence',{'expected_generation':control['generation'],'timeout_seconds':30})
            assert drained['drained']
            result['live_quiesced']=drained
            registry=Path(record['registry_path']);snapdir=run/'live-snapshot';snap=snapshot(registry,snapdir)
            try:stage_migration(snapdir,snap['manifest_sha256'],registry,run/'live-rejected','legacy-quiescence-v0')
            except ValueError as exc:
                assert 'owners_alive' in str(exc);result['live_migration_rejection']={'message':str(exc),'snapshot':snap}
            else:raise AssertionError('Live migration unexpectedly accepted')
            assert not (run/'live-rejected').exists()
            ep=ROOT/'envs'/record['engine']/'bin/python'
            result['old_constructor_rejected']=reject_constructor(ep,old_source,registry,'Registry protocol required')
    finally:
        for c in clients.values():c.close()
    return result

try:
    for engine,port in [('vllm',18795),('sglang',18794)]:
        ep=ROOT/'envs'/engine/'bin/python';entry={'attempts':[]};report['engines'][engine]=entry
        registry=None;original_state=None
        for phase,directory,generation in [('baseline',old_source,1),('candidate',source,3),('rollback',old_source,5)]:
            run=ROOT/'runs'/f'{engine}-migration-{phase}-0ac5935';run.mkdir(exist_ok=False)
            item={'phase':phase,'source_commit':directory.name,'run_dir':str(run),'started_at':time.time(),'preflight':preflight(port)};entry['attempts'].append(item);record=None;save()
            if phase=='candidate':
                original_state=inspect(registry)['summary'];entry['original_registry']=str(registry)
                entry['new_constructor_requires_migration']=reject_constructor(ep,source,registry,'migration required')
                folder=HERE/(engine+'-upgrade');entry['upgrade']=stage(registry,folder,'versioned-v1')
                assert inspect(registry)['summary']==original_state
                registry=folder/'registry.sqlite3'
            elif phase=='rollback':
                entry['versioned_registry']=str(registry)
                folder=HERE/(engine+'-rollback');entry['downgrade']=stage(registry,folder,'legacy-quiescence-v0')
                registry=folder/'registry.sqlite3'
                entry['new_constructor_rejects_downgrade']=reject_constructor(ep,source,registry,'migration required')
            if registry:item['resume']=resume(ep,directory,registry)
            health=run/'health-settings.json';health.write_text(json.dumps({'interval_seconds':1,'timeout_seconds':10,'max_age_seconds':30}))
            command=[str(ep),str(directory/'deployment/dsw_service.py'),'launch','--run-dir',str(run),'--engine',engine,'--model-path',str(ROOT/'models/SmolLM2-1.7B-Instruct'),'--gpus','7','--port',str(port),'--memory-fraction','.07' if engine=='vllm' else '.65','--api-workers','2','--health-config',str(health)]
            if registry:command.extend(['--registry-path',str(registry),'--no-bootstrap'])
            item['command']=command
            try:
                with (run/'launch.log').open('x') as f:subprocess.run(command,env=environment(directory),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
                record=read(run/'process.json');keys=read(run/'keys.json');registry=Path(record['registry_path'])
                item['ready']=wait_ready(record,keys['api'],420)
                (run/'topology.json').write_text(json.dumps(topology(run,record),indent=2)+'\n')
                item['serving']=serving(run,record,phase,generation)
                if phase!='baseline':
                    assert inspect(Path(entry['original_registry']))['summary']==original_state
                    assert not set(item['serving']['profiles'])&set(entry['attempts'][0]['serving']['profiles'])
                item['complete']=True
            except BaseException as exc:
                item['failure']={'type':type(exc).__name__,'message':str(exc)};raise
            finally:
                if record is None and (run/'process.json').exists():record=read(run/'process.json')
                if record:item['cleanup']=cleanup(run,record,timeout=90)
                item['finished_at']=time.time();save()
            assert item['cleanup']['passed']
        entry['original_state_unchanged']=inspect(Path(entry['original_registry']))['summary']==original_state
        assert entry['original_state_unchanged']
    report['postflight']=preflight(18795);report['complete']=True
except BaseException as exc:
    report['complete']=False;report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:report['finished_at']=time.time();save()
