import hashlib,json,os,signal,sqlite3,sys,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');HERE=ROOT/'quiescence-919c859';source=ROOT/'quiescence-sources'/'919c859598bbbd6bb885be280dd0b46360afd532';sys.path[:0]=[str(source),str(source/'src')]
from deployment.dsw_service import process_identity,get_gpu
from tests.integration.run_native_validation import group_members
run=ROOT/'runs/sglang-quiescence-restart-919c859';record=json.loads((run/'process.json').read_text());identity=record['identity'];assert identity['pid']==2756327 and process_identity(identity['pid'])==identity
proc=Path('/proc')/str(identity['pid']);command=(proc/'cmdline').read_bytes().split(b'\0');assert b'sglang.launch_server' in command
env=(proc/'environ').read_bytes().split(b'\0');assert ('JEV_CONFIG='+str(run/'config.json')).encode() in env
with sqlite3.connect(record['registry_path']) as db:
 counts={t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['leases','lease_work','lease_tenants','admission_tickets','raw_work','recovery_claims']};assert not any(counts.values())
 assert db.execute("SELECT state FROM backend_controls").fetchall()==[('QUIESCING',)]
report={'observed_at':time.time(),'identity':identity,'command_matches_owned_sglang':True,'config_matches_owned_run':True,'retained_counts_before':counts,'original_cleanup_sha256':hashlib.sha256((run/'cleanup.json').read_bytes()).hexdigest(),'quiescence_stop_sha256':hashlib.sha256((run/'quiescence-stop.json').read_bytes()).hexdigest(),'remaining_before':group_members(identity['pid']),'signal':'SIGKILL','qualification':'Explicit cleanup of a retained failed native shutdown; not a passing graceful-stop result'}
assert process_identity(identity['pid'])==identity
os.killpg(identity['pid'],signal.SIGKILL)
deadline=time.monotonic()+15
while group_members(identity['pid']) and time.monotonic()<deadline:time.sleep(.1)
report['remaining_after']=group_members(identity['pid']);report['gpu_after']=get_gpu(7);report['cleanup_confirmed']=not report['remaining_after'] and process_identity(identity['pid'])!=identity
assert report['cleanup_confirmed'] and int(report['gpu_after']['free_mib'])==11990
out=run/'cleanup-remediation.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
