"""Task-owned diagnostic hook: bounded ASGI thread-CPU profiles, no product changes."""
import cProfile
import hashlib
import json
import os
import pstats
import threading
import time
from pathlib import Path

if os.environ.get('JEV_CPU_PROFILE_DIR'):
    from fastapi import FastAPI
    original = FastAPI.__call__
    lanes = {}
    active = False
    output = Path(os.environ['JEV_CPU_PROFILE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    hook_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    skip, limit = 64, 128

    def save(lane, entry):
        base = output / (str(os.getpid()) + '-' + lane)
        entry['profile'].dump_stats(str(base) + '.prof')
        stats = pstats.Stats(entry['profile'])
        functions = []
        for (file, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
            functions.append({'file':file,'line':line,'name':name,'primitive_calls':primitive,
                              'calls':calls,'own_cpu_seconds':own,'cumulative_cpu_seconds':cumulative,
                              'callers':[{'file':f,'line':l,'name':n,'values':list(v) if isinstance(v,tuple) else v}
                                         for (f,l,n),v in callers.items()]})
        data={'pid':os.getpid(),'thread':threading.get_ident(),'lane':lane,'timer':'time.thread_time',
              'skip_requests':skip,'limit_requests':limit,'seen_requests':entry['seen'],
              'profiled_requests':entry['count'],'failed_requests':entry['failed'],
              'wall_seconds':entry['wall'],'statuses':entry['statuses'],'nested_or_concurrent_skips':entry['busy'],
              'hook_sha256':hook_sha256,'profiler_total_cpu_seconds':stats.total_tt,
              'qualification':'Instrumented ASGI worker thread CPU only; async/background callbacks may appear. Not unprofiled latency or GPU time.',
              'functions':sorted(functions,key=lambda f:f['own_cpu_seconds'],reverse=True)}
        base.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')

    async def profiled(self, scope, receive, send):
        global active
        lane = dict(scope.get('headers',[])).get(b'x-jev-dev-profile',b'').decode('ascii',errors='ignore')
        path = scope.get('path','')
        eligible = (scope.get('type')=='http' and scope.get('method')=='POST' and
                    ((lane=='native-plugin' and path=='/plugins/jev-runtime/v1/decisions') or
                     (lane=='native-label' and path in ['/generate','/v1/completions'])))
        if not eligible:
            return await original(self,scope,receive,send)
        entry = lanes.setdefault(lane,{'profile':cProfile.Profile(timer=time.thread_time),'seen':0,
                                      'count':0,'failed':0,'wall':0.0,'statuses':{},'busy':0})
        entry['seen'] += 1
        if entry['seen'] <= skip or entry['count'] >= limit:
            return await original(self,scope,receive,send)
        if active:
            entry['busy'] += 1
            return await original(self,scope,receive,send)
        active=True;start=time.perf_counter();status=None
        async def observed_send(message):
            nonlocal status
            if message['type']=='http.response.start':status=message['status']
            await send(message)
        entry['profile'].enable()
        try:
            return await original(self,scope,receive,observed_send)
        except BaseException:
            entry['failed']+=1
            raise
        finally:
            entry['profile'].disable();active=False;entry['count']+=1
            entry['wall']+=time.perf_counter()-start
            entry['statuses'][str(status)]=entry['statuses'].get(str(status),0)+1
            if entry['count']==limit:save(lane,entry)
    FastAPI.__call__ = profiled
