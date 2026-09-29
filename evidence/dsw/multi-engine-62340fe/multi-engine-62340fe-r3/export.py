import hashlib,json,shutil,sqlite3,tarfile,time
from pathlib import Path
ROOT=Path('/root/jev-runtime');out=ROOT/'multi-engine-evidence-62340fe';out.mkdir()
secrets=[]
for engine in ('vllm','sglang'):
 for suffix in ('','-r3'):
  p=ROOT/'runs'/f'{engine}-multi-engine-62340fe{suffix}'/'keys.json'
  d=json.loads(p.read_text());secrets.extend([d['api'].encode(),d['admin'].encode()])
def copy(src,dst):
 data=src.read_bytes()
 assert not any(s in data for s in secrets),str(src)
 target=out/dst;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
for attempt in ('multi-engine-62340fe','multi-engine-62340fe-r2','multi-engine-62340fe-r3'):
 for p in (ROOT/attempt).iterdir():
  if p.is_file() and p.suffix in ('.json','.log','.py'):copy(p,Path(attempt)/p.name)
for engine in ('vllm','sglang'):
 for suffix in ('','-r3'):
  run=ROOT/'runs'/f'{engine}-multi-engine-62340fe{suffix}'
  for p in run.iterdir():
   if p.is_file() and p.suffix in ('.json','.log') and p.name!='keys.json':copy(p,Path('runs')/run.name/p.name)
  db=sqlite3.connect(f'file:{run}/registry.db?mode=ro',uri=True)
  tables={row[0] for row in db.execute("select name from sqlite_master where type='table'")}
  journal={t:db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('leases','lease_work','lease_tenants','admission_tickets','raw_work','recovery_claims') if t in tables};db.close()
  (out/'runs'/run.name/'final-journals.json').write_text(json.dumps(journal,indent=2)+'\n')
source=ROOT/'multi-engine-62340fe-r3/62340feddb4d73274c0902d68cead1012da94624'
with tarfile.open(ROOT/'multi-engine-62340fe-r3/source.tar') as archive:
 hashes={}
 for item in archive.getmembers():
  if item.isfile():
   expected=hashlib.sha256(archive.extractfile(item).read()).hexdigest()
   assert hashlib.sha256((source/item.name).read_bytes()).hexdigest()==expected,item.name
   hashes[item.name]=expected
(out/'source-files.json').write_text(json.dumps(hashes,indent=2)+'\n')
manifest={str(p.relative_to(out)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
(out/'manifest.json').write_text(json.dumps({'source_commit':'62340feddb4d73274c0902d68cead1012da94624','credential_scan':'exact test API/admin values absent; keys.json excluded','files':manifest},indent=2)+'\n')
archive=ROOT/'multi-engine-evidence-62340fe.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='multi-engine-62340fe')
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(manifest),'source_files_verified':len(hashes)}))
for engine in ('vllm','sglang'):
 p=out/'runs'/f'{engine}-multi-engine-62340fe-r3'
 for name in ('validation.json','contract.json','cleanup.json'):
  data=json.loads((p/name).read_text())
  print(engine,name,json.dumps(data if name!='contract.json' else {k:v for k,v in data.items() if k not in ('model','checks','reference','responses')} )[:6500])
