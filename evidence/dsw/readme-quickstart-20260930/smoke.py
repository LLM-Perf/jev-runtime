import hashlib
import json
import os
import re
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = Path('/root/jev-runtime')
HERE = BASE / 'readme-quickstart-20260930'
SOURCE = BASE / 'strict-traffic-9da1813/9da181342326793f6edddbd3f3ada4b0f7dbb320'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'src')]
from deployment.dsw_service import process_identity, get_gpu
from tests.integration.run_native_validation import group_members
from jev_runtime.schema import DecisionResponse

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'runtime_source': '9da181342326793f6edddbd3f3ada4b0f7dbb320',
    'scope': 'README command smoke in existing engine environments; offline model reuse; no fresh dependency install or Hub download',
    'files': {p.name: sha(p) for p in HERE.iterdir() if p.is_file()},
    'protected_before': {'identity': process_identity(2851725), 'command_sha256': sha(Path('/proc/2851725/cmdline'))},
    'engines': {},
}
assert report['protected_before']['identity']

def save():
    (HERE / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

for engine, gpu, port, fraction in [('vllm', 7, 18795, '0.07'), ('sglang', 6, 18794, '0.65')]:
    row = report['engines'][engine] = {'gpu_before': get_gpu(gpu), 'port': port, 'memory_fraction': fraction}
    assert int(row['gpu_before']['free_mib']) >= 10000
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    root = HERE / engine
    python = str(BASE / 'envs' / engine / 'bin/python')
    cli = str(BASE / 'envs' / engine / 'bin/jevctl')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'JEV_', 'PIP_'))}
    env.update(
        PYTHONPATH=':'.join(str(p) for p in [SOURCE, SOURCE/'src', SOURCE/'packages/vllm/src', SOURCE/'packages/sglang/src']),
        PATH=str(BASE/'envs'/engine/'bin') + ':' + env['PATH'],
        CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    )
    prep = subprocess.run([python, str(HERE/'prepare_quickstart.py'), '--engine', engine,
                           '--output', str(root), '--port', str(port),
                           '--model-path', str(BASE/'models/SmolLM2-1.7B-Instruct')],
                          env=env, text=True, capture_output=True, timeout=90)
    (HERE / (engine+'-prepare.log')).write_text(prep.stdout + prep.stderr)
    row['prepare_returncode'] = prep.returncode
    save()
    assert prep.returncode == 0, prep.stderr
    for line in (root/'env.sh').read_text().splitlines():
        if line.startswith('export '):
            assignment = shlex.split(line)[1]
            key, val = assignment.split('=', 1)
            env[key] = val
    assert (root/'env.sh').stat().st_mode & 0o777 == 0o600
    assert env['JEV_API_KEY'] != env['JEV_ADMIN_KEY']
    repeat = subprocess.run(prep.args, env=env, capture_output=True, text=True, timeout=30)
    row['existing_output_rejected'] = repeat.returncode != 0 and 'already exists' in repeat.stderr
    assert row['existing_output_rejected']
    doc = HERE / ('README.md' if engine == 'vllm' else 'quickstart-sglang.md')
    blocks = re.findall(r'```bash\n(.*?)\n```', doc.read_text(), flags=re.S)
    start = next(b for b in blocks if b.startswith('vllm serve' if engine == 'vllm' else 'python -m sglang.launch_server'))
    start = start.replace('--gpu-memory-utilization 0.8', '--gpu-memory-utilization '+fraction)
    start = start.replace('--mem-fraction-static 0.8', '--mem-fraction-static '+fraction)
    row['start_command_from_document'] = start
    log = open(HERE / (engine+'-engine.log'), 'x')
    proc = subprocess.Popen(['bash', '-c', 'exec '+start], env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    row['identity'] = process_identity(proc.pid)
    save()
    print(engine + ': started', flush=True)
    client = httpx.Client(base_url=env['JEV_URL'], headers={'Authorization': 'Bearer '+env['JEV_API_KEY']}, timeout=30)
    admin = httpx.Client(base_url=env['JEV_URL'], headers={'Authorization': 'Bearer '+env['JEV_ADMIN_KEY']}, timeout=60)
    ready = False
    try:
        begun = time.monotonic()
        while time.monotonic()-begun < 240:
            assert proc.poll() is None, 'Engine exited; see log'
            try:
                response = client.get('/ready', timeout=3)
                if response.status_code == 200 and response.json().get('ready') is True:
                    ready = True
                    row['ready'] = response.json()
                    break
            except httpx.TransportError:
                pass
            time.sleep(1)
        assert ready, 'Readiness timeout'
        row['startup_seconds'] = time.monotonic()-begun
        def call(*args):
            result = subprocess.run([cli, *args, '--url', env['JEV_URL']], env=env, capture_output=True, text=True, timeout=180)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        row['bundle_before'] = call('bundle', 'list')
        row['decisions'] = []
        def decide(ref, generation):
            result = call('decide', str(HERE/'request.json'))
            parsed = DecisionResponse.model_validate(result)
            assert parsed.status == 'completed'
            assert parsed.bundle == ref and parsed.generation == generation
            assert set(parsed.answers) == {'intent', 'refund_requested'}
            assert parsed.engine['name'] == engine
            row['decisions'].append(result)
            save()
        decide('default@1', 1)
        row['build'] = call('bundle', 'build-remote', str(root/'demo-v2.json'), '--name', 'demo', '--version', '2')
        row['upload'] = call('bundle', 'upload', str(root/'demo-v2.json'))
        row['prepare'] = call('bundle', 'prepare', 'demo@2')
        row['activate'] = call('bundle', 'activate', 'demo@2', 'decision-model', '1')
        decide('demo@2', 2)
        row['rollback'] = call('bundle', 'activate', 'default@1', 'decision-model', '2')
        decide('default@1', 3)
        response = client.post('/v1/decisions', json=json.loads((HERE/'request.json').read_text()))
        response.raise_for_status()
        DecisionResponse.model_validate(response.json())
        row['direct_http_decision'] = response.json()
        row['passed'] = True
        print(engine + ': requests, activation and rollback passed', flush=True)
    finally:
        if proc.poll() is None:
            status = admin.get('/admin/quiescence')
            status.raise_for_status()
            quiesce = admin.post('/admin/quiescence', json={'expected_generation': status.json()['generation'], 'timeout_seconds': 30})
            quiesce.raise_for_status()
            row['quiescence'] = quiesce.json()
            assert row['quiescence']['drained'] is True
            assert process_identity(proc.pid) == row['identity']
            os.kill(proc.pid, signal.SIGTERM)
        row['exit_code'] = proc.wait(timeout=90)
        deadline = time.monotonic()+60
        while group_members(proc.pid) and time.monotonic() < deadline:
            time.sleep(1)
        row['remaining_group_members'] = group_members(proc.pid)
        log.close()
        client.close()
        admin.close()
        row['gpu_after'] = get_gpu(gpu)
        if (root/'registry.db').exists():
            con = sqlite3.connect(root/'registry.db')
            row['registry_integrity'] = con.execute('pragma integrity_check').fetchone()[0]
            con.close()
        save()
        assert not row['remaining_group_members']
        print(engine + ': cleanup verified', flush=True)

report['protected_after'] = {'identity': process_identity(2851725), 'command_sha256': sha(Path('/proc/2851725/cmdline'))}
assert report['protected_before'] == report['protected_after']
report['passed'] = all(r.get('passed') for r in report['engines'].values())
save()
print('README smoke passed', flush=True)
