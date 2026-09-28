"""Bounded native vLLM shutdown validation; retain every attempt and raw log."""
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx

ROOT = Path('/root/jev-runtime')
HERE = ROOT / 'vllm-shutdown-e3f82d7'
meta = json.loads((HERE / 'sources.json').read_text())
source = ROOT / 'quiescence-sources' / meta['commit']
assert hashlib.sha256((HERE / 'source.tar.gz').read_bytes()).hexdigest() == meta['archive_sha256']
shutil.copytree(ROOT / 'quiescence-sources' / meta['base_commit'], source)
with tarfile.open(HERE / 'source.tar.gz') as tar:
    tar.extractall(source, filter='data')
sys.path[:0] = [str(source), str(source / 'src')]
from deployment.dsw_service import process_identity, get_gpu
from tests.integration.run_native_validation import cleanup, group_members, wait_ready, topology
from jev_runtime.schema import DecisionResponse

OUT = HERE / 'campaign.json'
assert not OUT.exists()
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
report = {'sources': meta, 'script_sha256': sha(Path(__file__)), 'started_at': time.time(),
          'attempts': [], 'qualification': 'Colocated native vLLM shutdown checks, not performance, quality or release certification'}
def save(): OUT.write_text(json.dumps(report, indent=2) + '\n')
def preflight():
    records = list((ROOT / 'runs').rglob('process.json'))
    for p in records:
        identity = read(p)['identity']
        assert process_identity(identity['pid']) != identity and not group_members(identity['pid']), str(p)
    gpu = get_gpu(7)
    assert gpu['uuid'] == 'GPU-b57fb933-0e5a-dd28-7041-a177da03405e' and int(gpu['free_mib']) >= 11900
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 18795))
    return {'gpu7': gpu, 'owned_records_checked': len(records), 'all_owned_groups_terminal': True, 'port_bindable': True}

env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'JEV_CPU_', 'JEV_CALL_'))}
env.update(PYTHONPATH=':'.join(map(str, [source, source / 'src', source / 'packages/vllm/src', source / 'packages/sglang/src'])), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ep = ROOT / 'envs/vllm/bin/python'
native = ROOT / 'envs/vllm/lib/python3.12/site-packages/vllm'
native_names = ['engine/arg_utils.py', 'entrypoints/cli/serve.py', 'entrypoints/launchers/launcher.py', 'v1/utils.py', 'v1/engine/utils.py', 'v1/engine/core.py']
report['native_source_files'] = {name: sha(native / name) for name in native_names}
report['versions'] = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers', 'tokenizers')}
report['python'] = sys.version
save()
old = previous = None
try:
    for phase, workers, grace in [('single', 1, 30), ('dual', 2, 30), ('restart', 2, 45)]:
        item = {'engine': 'vllm', 'phase': phase, 'api_workers': workers, 'native_grace_seconds': grace, 'started_at': time.time(), 'preflight': preflight()}
        report['attempts'].append(item)
        run = ROOT / 'runs' / ('vllm-shutdown-' + phase + '-e3f82d7')
        run.mkdir(exist_ok=False)
        item['run_dir'] = str(run)
        health = run / 'health-settings.json'
        health.write_text(json.dumps({'interval_seconds': 1, 'timeout_seconds': 10, 'max_age_seconds': 30}))
        record = None
        save()
        if phase == 'restart':
            registry = Path(old['registry_path'])
            backend = previous['checks']['drained']['backend']
            with sqlite3.connect(registry) as db:
                generation = db.execute('SELECT generation FROM backend_controls WHERE backend=?', (backend,)).fetchone()[0]
            command = [str(ep), '-c', 'from jev_runtime.cli import app;app()', 'registry', 'resume-backend', str(registry), backend, str(generation)]
            item['resume'] = json.loads(subprocess.check_output(command, env=env, text=True, timeout=30))
            assert item['resume']['state'] == 'OPEN' and item['resume']['generation'] == generation + 1
        command = [str(ep), str(source / 'deployment/dsw_service.py'), 'launch', '--run-dir', str(run), '--engine', 'vllm', '--model-path', str(ROOT / 'models/SmolLM2-1.7B-Instruct'), '--gpus', '7', '--port', '18795', '--memory-fraction', '.07', '--health-config', str(health)]
        if workers == 2:
            command.extend(['--api-workers', '2'])
        if phase == 'restart':
            command.extend(['--registry-path', str(registry), '--no-bootstrap', '--vllm-shutdown-timeout', str(grace)])
        item['command'] = command
        try:
            with (run / 'launch.log').open('x') as f:
                subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=90)
            record = read(run / 'process.json')
            assert record['native_shutdown_timeout_seconds'] == grace
            assert record['command'][record['command'].index('--shutdown-timeout') + 1] == str(grace)
            keys = read(run / 'keys.json')
            item['ready'] = wait_ready(record, keys['api'], 420)
            (run / 'topology.json').write_text(json.dumps(topology(run, record), indent=2) + '\n')
            save()
            base = f"http://127.0.0.1:{record['port']}/plugins/jev-runtime"
            admin = {'Authorization': 'Bearer ' + keys['admin']}
            data = {'Authorization': 'Bearer ' + keys['api']}
            if phase == 'dual':
                command = [str(ep), str(source / 'tests/integration/live_quiescence.py'), '--run-dir', str(run), '--output', str(run / 'quiescence.json'), '--source-commit', meta['commit']]
                with (run / 'check.log').open('x') as f:
                    subprocess.run(command, env=env, cwd=source, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=240)
                previous = read(run / 'quiescence.json')
                assert previous['passed']
                old = record
            else:
                seen, responses = {}, []
                deadline = time.monotonic() + 30
                while len(seen) != workers:
                    assert time.monotonic() < deadline
                    with httpx.Client(base_url=base, timeout=30, trust_env=False) as c:
                        r = c.get('/admin/profile', headers=admin); r.raise_for_status(); profile = r.json()
                        if profile['worker_id'] in seen:
                            continue
                        assert profile['health']['monitor_running']
                        seen[profile['worker_id']] = profile
                        for alias in (['decision-model'] if phase == 'single' else ['decision-model', 'quiescence']):
                            body = {'model': alias, 'input': {'text': 'Please refund the duplicate charge.'}, 'questions': [{'id': 'q', 'type': 'boolean', 'instruction': 'Refund requested?'}]}
                            r = c.post('/v1/decisions', headers=data, json=body); r.raise_for_status()
                            parsed = DecisionResponse.model_validate(r.json())
                            assert parsed.status == 'completed' and r.headers['x-jev-worker'] == profile['worker_id']
                            responses.append(parsed.model_dump(mode='json'))
                            if phase == 'single':
                                r = c.post('/admin/compile', headers=admin, json=body); r.raise_for_status()
                                raw_body = {**r.json()['sequences'][0], 'request_id': 'single-raw'}
                                r = c.post('/v1/scores', headers=data, json=raw_body); r.raise_for_status()
                                raw = r.json()
                                assert raw['request_id'] == 'single-raw' and raw['raw_logprobs'] and len(raw['logprobs']) == 2
                                assert all(math.isfinite(v) and v <= 0 for v in raw['logprobs'])
                                item['raw_response'] = raw
                with httpx.Client(base_url=base, timeout=30, trust_env=False) as c:
                    r = c.get('/admin/bundles', headers=admin); r.raise_for_status()
                    item['serving_checks'] = {'profiles': seen, 'responses': responses, 'routes': r.json()['routes']}
                if phase == 'restart':
                    assert item['serving_checks']['routes'] == previous['checks']['routes_before']
                    assert not set(seen) & set(previous['checks']['workers'])
            item['complete'] = True
        except BaseException as exc:
            item['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            if record is None and (run / 'process.json').exists():
                record = read(run / 'process.json')
            if record:
                started = time.monotonic()
                item['cleanup'] = cleanup(run, record, timeout=90)
                item['cleanup_elapsed_seconds'] = time.monotonic() - started
                log = (run / 'engine.log').read_text()
                patterns = ['force killing', 'leaked semaphore', 'Traceback (most recent call last)', 'mode=abort', 'mode=drain', 'request processing complete; starting resource teardown']
                item['native_log_counts'] = {pattern: log.count(pattern) for pattern in patterns}
            item['finished_at'] = time.time()
            save()
        assert item['cleanup']['passed']
    report['postflight'] = preflight()
    report['native_source_files_after'] = {name: sha(native / name) for name in native_names}
    assert report['native_source_files'] == report['native_source_files_after']
    report['complete'] = True
except BaseException as exc:
    report['complete'] = False
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    report['finished_at'] = time.time()
    save()
