"""Task-owned continuous-window call profiler. No changes to product packages."""
import cProfile
import hashlib
import json
import os
import pstats
import time
from pathlib import Path

if os.environ.get('JEV_CALL_PROFILE_DIR'):
    from fastapi import FastAPI
    original=FastAPI.__call__
    output=Path(os.environ['JEV_CALL_PROFILE_DIR']);output.mkdir(parents=True,exist_ok=True)
    hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lanes={};active=None;skip=64;limit=128
    def save(lane,entry):
        base=output/(str(os.getpid())+'-'+lane);entry['profile'].dump_stats(str(base)+'.prof')
        stats=pstats.Stats(entry['profile']);functions=[]
        for (file,line,name),(primitive,calls,own,cumulative,callers) in stats.stats.items():
            functions.append({'file':file,'line':line,'name':name,'primitive_calls':primitive,'calls':calls,'own_seconds':own,'cumulative_seconds':cumulative})
        window=time.perf_counter()-entry['start']
        valid=(all(f['own_seconds']>=0 and f['cumulative_seconds']>=0 for f in functions) and
               0<stats.total_tt<=window*1.05)
        data={'pid':os.getpid(),'lane':lane,'timer':'cProfile default monotonic wall clock',
              'skip_requests':skip,'limit_requests':limit,'seen_requests':entry['seen'],
              'profiled_requests':entry['count'],'statuses':entry['statuses'],'failed_requests':entry['failed'],
              'nested_or_concurrent_skips':entry['busy'],'window_wall_seconds':window,
              'profiler_total_seconds':stats.total_tt,'timing_sanity_passed':valid,'hook_sha256':hook_sha256,
              'qualification':'One contiguous window, including event-loop idle/background callbacks. Function times overlap; not CPU-only or GPU time, and not ordinary throughput.',
              'functions':sorted(functions,key=lambda f:f['own_seconds'],reverse=True)}
        base.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
    async def profiled(self,scope,receive,send):
        global active
        lane=dict(scope.get('headers',[])).get(b'x-jev-dev-profile',b'').decode('ascii',errors='ignore')
        path=scope.get('path','')
        eligible=(scope.get('type')=='http' and scope.get('method')=='POST' and
                  ((lane=='native-plugin' and path=='/plugins/jev-runtime/v1/decisions') or
                   (lane=='native-label' and path in ['/generate','/v1/completions'])))
        if not eligible:return await original(self,scope,receive,send)
        entry=lanes.setdefault(lane,{'profile':cProfile.Profile(),'seen':0,'count':0,'failed':0,'busy':0,'inflight':False,'statuses':{}})
        entry['seen']+=1
        if entry['seen']<=skip or entry['count']>=limit:return await original(self,scope,receive,send)
        if entry['inflight'] or (active is not None and active!=lane):
            entry['busy']+=1;return await original(self,scope,receive,send)
        entry['inflight']=True;status=None
        async def observed_send(message):
            nonlocal status
            if message['type']=='http.response.start':status=message['status']
            await send(message)
        if active is None:
            active=lane;entry['start']=time.perf_counter();entry['profile'].enable()
        try:return await original(self,scope,receive,observed_send)
        except BaseException:entry['failed']+=1;raise
        finally:
            entry['inflight']=False;entry['count']+=1
            entry['statuses'][str(status)]=entry['statuses'].get(str(status),0)+1
            if entry['count']==limit:
                entry['profile'].disable();active=None;save(lane,entry)
    FastAPI.__call__=profiled
