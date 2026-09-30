import subprocess
import sys
from pathlib import Path


def test_demo_fixture_runs_the_full_serving_loop():
    script = Path(__file__).parents[1] / "examples" / "demo_fixture.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert '"bundle": "support-intent@1"' in result.stdout
    assert '"bundle": "support-intent@2"' in result.stdout
    assert '"generation": 2' in result.stdout
    assert '"status": "completed"' in result.stdout
    assert "not model-quality" in result.stdout
