import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import time

ROOT = Path('/root/jev-runtime')
HERE = ROOT / 'strict-traffic-4ee78ca'
COMMIT = '4ee78cab97c62d3ae03c5486e628af5bee797a28'
SOURCE = HERE / COMMIT
archive = HERE / 'source.tar'
expected = (HERE / 'source.sha256').read_text().strip()
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
SOURCE.mkdir()
with tarfile.open(archive) as tar:
    tar.extractall(SOURCE, filter='data')
sys.path[:0] = [str(SOURCE), str(SOURCE / 'src')]
from deployment.dsw_service import process_identity, get_gpu
from tests.integration.run_native_validation import group_members

report = {'source_commit': COMMIT, 'source_archive_sha256': expected,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'supervisor_identity': process_identity(os.getpid()), 'started_at': time.time(),
          'minimum_requests_per_engine': 10000, 'minimum_switches_per_engine': 1000,
          'concurrency_per_engine': 4, 'contracts': {}, 'native': {},
          'qualification': 'simultaneous colocated native functional traffic; not performance or soak'}
assert not (HERE / 'campaign.json').exists()
def save():
    pending = HERE / 'campaign.pending'
    pending.write_text(json.dumps(report, indent=2) + '\n')
    pending.replace(HERE / 'campaign.json')
save()
records = list((ROOT / 'runs').rglob('process.json'))
for path in records:
    identity = json.loads(path.read_text())['identity']
    assert process_identity(identity['pid']) != identity and not group_members(identity['pid']), str(path)
report['prior_owned_groups_terminal'] = len(records)
protected = process_identity(2851725)
assert protected is not None
protected_hash = hashlib.sha256(Path('/proc/2851725/cmdline').read_bytes()).hexdigest()
report['protected_service_before'] = {'identity': protected, 'cmdline_sha256': protected_hash}
report['gpus_before'] = [get_gpu(i) for i in (6, 7)]
for index, gpu_uuid, port in [(6, 'GPU-b028ddfd-c58e-76d1-863b-21ec06f99dbe', 18794),
                              (7, 'GPU-b57fb933-0e5a-dd28-7041-a177da03405e', 18795)]:
    gpu = get_gpu(index)
    assert gpu['uuid'] == gpu_uuid and int(gpu['free_mib']) >= 10000
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))

def environment():
    env = {k:v for k,v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'JEV_'))}
    env.update(PYTHONPATH=':'.join(map(str, [SOURCE, SOURCE/'src', SOURCE/'packages/vllm/src',
                 SOURCE/'packages/sglang/src', ROOT/'combined-reservation-7391ed3/test-tools'])),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env
for engine in ('vllm', 'sglang'):
    cmd = [str(ROOT/'envs'/engine/'bin/python'), '-m', 'pytest', '-q',
           *[str(SOURCE/'tests'/name) for name in ('test_hot_switch_traffic.py',
                'test_integration_contract.py', 'test_native_validation_runner.py')]]
    with (HERE / f'contracts-{engine}.log').open('x') as file:
        rc = subprocess.run(cmd, env=environment(), cwd=SOURCE, stdout=file,
                            stderr=subprocess.STDOUT, timeout=180).returncode
    report['contracts'][engine] = {'returncode': rc, 'command': cmd}
    save()
    assert rc == 0

def native(engine, gpu, port, fraction):
    run = ROOT / 'runs' / f'{engine}-strict-traffic-4ee78ca'
    cmd = [str(ROOT/'envs'/engine/'bin/python'), str(SOURCE/'tests/integration/run_native_validation.py'),
           '--run-dir', str(run), '--engine', engine, '--model-path', str(ROOT/'models/SmolLM2-1.7B-Instruct'),
           '--gpus', str(gpu), '--port', str(port), '--memory-fraction', str(fraction),
           '--minimum-requests', '10000', '--switches', '1000', '--traffic-concurrency', '4',
           '--traffic-timeout', '1200', '--check-timeout', '1500',
           '--source-commit', COMMIT, '--runtime-source-commit', COMMIT]
    with (HERE / f'{engine}-native.log').open('x') as file:
        rc = subprocess.run(cmd, env=environment(), cwd=SOURCE, stdout=file,
                            stderr=subprocess.STDOUT, timeout=3000).returncode
    return {'returncode': rc, 'command': cmd, 'run_dir': str(run),
            'validation': json.loads((run/'validation.json').read_text())}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = {pool.submit(native, 'vllm', 7, 18795, .07): 'vllm',
               pool.submit(native, 'sglang', 6, 18794, .65): 'sglang'}
    for future in concurrent.futures.as_completed(futures):
        engine = futures[future]
        try:
            report['native'][engine] = future.result()
        except Exception as exc:
            report['native'][engine] = {'runner_failure': repr(exc)}
        save()
report['gpus_after'] = [get_gpu(i) for i in (6, 7)]
report['protected_service_after'] = {'identity': process_identity(2851725),
    'cmdline_sha256': hashlib.sha256(Path('/proc/2851725/cmdline').read_bytes()).hexdigest()}
report['protected_service_unchanged'] = report['protected_service_before'] == report['protected_service_after']
report['finished_at'] = time.time()
report['passed'] = len(report['native']) == 2 and report['protected_service_unchanged'] and all(
    row.get('returncode') == 0 and row['validation']['passed'] for row in report['native'].values())
save()
print(json.dumps({'passed': report['passed'], 'source_commit': COMMIT}), flush=True)
raise SystemExit(0 if report['passed'] else 1)
