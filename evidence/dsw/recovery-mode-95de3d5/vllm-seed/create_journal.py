import json,time,uuid
from pathlib import Path
from jev_runtime.config import Settings,load_compiler,model_identity
from jev_runtime.registry import Registry
from jev_runtime.schema import Bundle
root=Path('/root/jev-runtime');seed=root/'runs/vllm-recovery-seed-95de3d5'
s=Settings(backend='vllm',engine_url='http://127.0.0.1:18795',model_id='Qwen/Qwen3-0.6B',model_revision='c1899de289a04d12100db370d81485cdf75e47ca',tokenizer=str(root/'tokenizers/Qwen3-0.6B-preserved-vllm-cdd0caf'))
r=Registry(seed/'registry.db');b=Bundle(id='crashed-prepare',version=1,model=model_identity(s,load_compiler(s)));r.upload(b)
rid='prepare-'+uuid.uuid4().hex;_,lease=r.begin_prepare(b.reference,'vllm:http://127.0.0.1:18795:Qwen/Qwen3-0.6B:c1899de289a04d12100db370d81485cdf75e47ca',rid);r.record_branches(lease,[rid+'.before-dispatch'])
(seed/'journal.json').write_text(json.dumps({'requests':r.recovery_candidates(),'dispatched_to_engine':False,'owner':r.owner},indent=2)+'\n')
while True:time.sleep(1)
