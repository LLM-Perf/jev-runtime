import json
from types import SimpleNamespace

import pytest

from deployment import dsw_service


@pytest.mark.parametrize("drained,changed", [(False, False), (True, True), (True, False)])
def test_launcher_signals_only_after_drain_and_fresh_identity(
    tmp_path, monkeypatch, drained, changed
):
    import httpx

    identity = {"pid": 999, "start_ticks": "111", "boot_id": "fixture"}
    (tmp_path / "process.json").write_text(
        json.dumps({"identity": identity, "port": 18794, "mode": "native-plugin"})
    )
    (tmp_path / "keys.json").write_text(json.dumps({"admin": "private-test-key"}))
    identities = iter([identity, {**identity, "start_ticks": "222"} if changed else identity])
    monkeypatch.setattr(dsw_service, "process_identity", lambda pid: next(identities))
    signalled = []
    monkeypatch.setattr(dsw_service.os, "killpg", lambda *args: signalled.append(args))

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["base_url"] == "http://127.0.0.1:18794/plugins/jev-runtime"
            assert kwargs["headers"] == {"Authorization": "Bearer private-test-key"}
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, path):
            assert path == "/admin/quiescence"
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"generation": 3})

        def post(self, path, json):
            assert path == "/admin/quiescence"
            assert json == {"expected_generation": 3, "timeout_seconds": 60}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"drained": drained})

    monkeypatch.setattr(httpx, "Client", Client)
    if not drained or changed:
        with pytest.raises(SystemExit):
            dsw_service.stop(SimpleNamespace(run_dir=tmp_path))
        assert not signalled
    else:
        dsw_service.stop(SimpleNamespace(run_dir=tmp_path))
        assert signalled == [(999, dsw_service.signal.SIGTERM)]
    assert json.loads((tmp_path / "quiescence-stop.json").read_text())["drained"] == drained
