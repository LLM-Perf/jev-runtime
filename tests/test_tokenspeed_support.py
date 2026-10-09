import asyncio
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
from jev_tokenspeed import preflight
from jev_tokenspeed.backend import TokenSpeedNative, validate_profile
from jev_tokenspeed.plugin import submit_to_engine, validate_binding, verify_source
from jev_tokenspeed.receipts import CompletionReceipts

from jev_runtime.config import Settings
from jev_runtime.errors import JevError
from tests.test_backend_extensions import native_output, profile, request


@pytest.fixture
def generate_input(monkeypatch):
    module = ModuleType("tokenspeed.runtime.engine.io_struct")
    module.GenerateReqInput = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)


async def test_owner_loop_concurrency_is_not_limited_by_http_thread_pool():
    # A blocking llm.run per request deadlocks this rendezvous with two threads.
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    started.wait(2)
    count = 0
    gate = asyncio.Event()

    async def receive():
        nonlocal count
        count += 1
        if count == 64:
            gate.set()
        await gate.wait()
        assert threading.current_thread() is thread
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Exercise under a deliberately tiny executor without changing pytest's loop.
        async def run_all():
            asyncio.get_running_loop().set_default_executor(pool)
            return await asyncio.wait_for(
                asyncio.gather(*(submit_to_engine(loop, receive()) for _ in range(64))), 5
            )

        try:
            results = await asyncio.to_thread(lambda: asyncio.run(run_all()))
            assert len(results) == 64 and len(set(results)) == 1
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(2)
    assert not thread.is_alive()


async def test_closed_owner_loop_closes_unsubmitted_coroutine():
    loop = asyncio.new_event_loop()
    loop.close()

    async def receive():
        pass

    coro = receive()
    with pytest.raises(RuntimeError):
        await submit_to_engine(loop, coro)
    assert coro.cr_frame is None


async def test_durable_receipts_survive_restart_and_cache_eviction(tmp_path, generate_input):
    calls = []

    async def generate(obj):
        calls.append(obj.rid)
        yield native_output(obj.rid)

    async def submit(coro):
        return await coro

    store = CompletionReceipts(tmp_path / "receipts.db", "same-model")
    manager = SimpleNamespace(generate_request=generate)
    original = TokenSpeedNative(manager, "fixture", submit, receipts=store)
    assert (await original.score(request())).logprobs == (-2.0,)
    original._completed.clear()
    await original.cancel("r")
    restarted = TokenSpeedNative(
        manager,
        "fixture",
        submit,
        receipts=CompletionReceipts(tmp_path / "receipts.db", "same-model"),
    )
    await restarted.cancel("r")
    with pytest.raises(JevError, match="already dispatched"):
        await restarted.score(request())
    assert calls == ["r"] and not restarted._attempts
    store.reserve("unfinished")
    for rid in ("unknown", "unfinished"):
        with pytest.raises(JevError) as exc:
            await restarted.cancel(rid)
        assert exc.value.code == "cancellation_unconfirmed"
    other = TokenSpeedNative(
        manager,
        "fixture",
        submit,
        receipts=CompletionReceipts(tmp_path / "receipts.db", "different-model"),
    )
    with pytest.raises(JevError):
        await other.cancel("r")


async def test_receipt_write_failure_retains_request_until_durable_drain(
    tmp_path, generate_input, monkeypatch
):
    async def generate(obj):
        yield native_output(obj.rid)

    async def submit(coro):
        return await coro

    store = CompletionReceipts(tmp_path / "receipts.db", "model")
    complete = store.complete

    def fail(rid):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(store, "complete", fail)
    backend = TokenSpeedNative(
        SimpleNamespace(generate_request=generate), "fixture", submit, receipts=store
    )
    with pytest.raises(sqlite3.OperationalError):
        await backend.score(request())
    assert "r" in backend._attempts and not store.completed("r")
    with pytest.raises(sqlite3.OperationalError):
        await backend.cancel("r")
    monkeypatch.setattr(store, "complete", complete)
    await backend.cancel("r")
    assert store.completed("r") and not backend._attempts


async def test_wrong_terminal_identity_never_confirms_receipt(tmp_path, generate_input):
    async def generate(obj):
        yield native_output("wrong-id")

    async def submit(coro):
        return await coro

    store = CompletionReceipts(tmp_path / "receipts.db", "model")
    backend = TokenSpeedNative(
        SimpleNamespace(generate_request=generate), "fixture", submit, receipts=store
    )
    with pytest.raises(JevError):
        await backend.score(request())
    with pytest.raises(JevError) as exc:
        await backend.cancel("r")
    assert exc.value.code == "cancellation_unconfirmed" and not store.completed("r")


def test_receipt_reservation_is_atomic_and_does_not_accept_foreign_database(tmp_path):
    store = CompletionReceipts(tmp_path / "receipts.db", "model")

    def reserve(_):
        try:
            store.reserve("same")
            return True
        except JevError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(32))) == 1
    assert not store.completed("same")
    with pytest.raises(ValueError):
        store.complete("unknown")
    path = tmp_path / "registry.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE bundle(id TEXT)")
    with pytest.raises(ValueError, match="new database"):
        CompletionReceipts(path, "model")
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=1")
    with pytest.raises(ValueError, match="compatible receipt store"):
        CompletionReceipts(path, "model")


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (12, 0), None])
def test_hardware_preflight_rejects_unsupported_devices(monkeypatch, capability):
    torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: capability is not None,
            get_device_capability=lambda _: capability,
            device_count=lambda: 1,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(JevError) as exc:
        preflight.check_hardware()
    assert exc.value.code == "tokenspeed_hardware"


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (10, 7)])
def test_hardware_preflight_records_supported_device(monkeypatch, capability):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="fixture",
            version=SimpleNamespace(hip=None, cuda="fixture"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_capability=lambda _: capability,
                get_device_name=lambda _: "fixture",
            ),
        ),
    )
    assert preflight.check_hardware()["capability"] == list(capability)


def test_hardware_preflight_checks_the_actual_selected_gpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="fixture",
            version=SimpleNamespace(hip=None, cuda="fixture"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
                get_device_capability=lambda index: [(8, 9), (9, 0)][index],
                get_device_name=lambda index: ["unsupported", "supported"][index],
            ),
        ),
    )
    assert preflight.check_hardware(1)["device"] == "supported"
    for index in (0, 2, -1, "1", True):
        with pytest.raises(JevError):
            preflight.check_hardware(index)


def test_model_and_separate_tokenizer_paths_must_both_match():
    settings = Settings(
        backend="tokenspeed",
        model_id="/weights",
        model_revision="a" * 40,
        tokenizer="/preserved/tokenizer",
    )
    args = SimpleNamespace(model="/weights", revision="a" * 40, tokenizer="/preserved/tokenizer")
    validate_binding(settings, args)
    for key in ("model", "revision", "tokenizer"):
        invalid = SimpleNamespace(**{**vars(args), key: "wrong"})
        with pytest.raises(ValueError):
            validate_binding(settings, invalid)


def test_source_check_requires_every_guarded_file(tmp_path, monkeypatch):
    import hashlib

    from jev_tokenspeed import plugin

    package = tmp_path / "plugin"
    package.mkdir()
    root = tmp_path / "tokenspeed"
    root.mkdir()
    (root / "sample.py").write_text("correct")
    (package / "source-profile.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "files": {
                    "sample.py": hashlib.sha256(b"correct").hexdigest(),
                    "output.py": hashlib.sha256(b"correct output").hexdigest(),
                },
            }
        )
    )
    monkeypatch.setattr(plugin, "__file__", str(package / "plugin.py"))
    with pytest.raises(JevError, match="output.py"):
        verify_source(root)
    (root / "output.py").write_text("correct output")
    assert verify_source(root) == "a" * 40
    (root / "sample.py").write_text("changed")
    with pytest.raises(JevError, match="sample.py"):
        verify_source(root)


def test_untested_logprob_order_is_rejected():
    args = profile()
    args.logprob_order = "megatron"
    with pytest.raises(JevError) as exc:
        validate_profile(args)
    assert exc.value.code == "tokenspeed_profile"


def test_preparation_generates_matching_paths_and_preserves_existing_run(tmp_path, monkeypatch):
    import stat

    from examples import prepare_tokenspeed
    from jev_runtime.config import load_settings

    model = tmp_path / "weights with spaces"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").touch()  # File presence only, not a real checkpoint.
    output = tmp_path / "run with spaces"
    copied = []
    monkeypatch.setattr(
        prepare_tokenspeed,
        "preserve_fast_tokenizer",
        lambda source, target: copied.append((source, target)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_tokenspeed",
            "--model-path",
            str(model),
            "--output",
            str(output),
            "--port",
            "8896",
        ],
    )
    prepare_tokenspeed.main()
    settings = load_settings(output / "config.json")
    options = json.loads((output / "engine.json").read_text())
    validate_binding(settings, SimpleNamespace(**options))
    assert settings.model_id == str(model) and copied == [(model, output / "tokenizer")]
    assert options["enforce_eager"] and options["enable_output_logprobs"]
    assert settings.engine_url == "http://127.0.0.1:8896"
    assert stat.S_IMODE((output / "env.sh").stat().st_mode) == 0o600
    before = (output / "env.sh").read_bytes()
    assert b"JEV_ENGINE_CONFIG=" in before and b"SGLANG_PLUGINS" not in before
    source = json.loads((output / "model-source.json").read_text())
    assert source["model_id"] == str(model) and source["revision"] == settings.model_revision
    with pytest.raises(SystemExit) as exc:
        prepare_tokenspeed.main()
    assert exc.value.code == 2 and (output / "env.sh").read_bytes() == before


def test_source_only_cli_never_imports_the_engine(tmp_path, monkeypatch, capsys):
    from jev_tokenspeed import launcher

    monkeypatch.setattr(launcher, "verify_source", lambda root: "a" * 40)
    monkeypatch.setattr(sys, "argv", ["jev-tokenspeed", "--check-source", str(tmp_path)])
    before = set(sys.modules)
    launcher.main()
    assert not any(name.startswith("tokenspeed") for name in set(sys.modules) - before)
    assert json.loads(capsys.readouterr().out)["gpu_validated"] is False


def test_full_preflight_requires_configs(monkeypatch):
    from jev_tokenspeed import launcher

    monkeypatch.setattr(sys, "argv", ["jev-tokenspeed", "--check"])
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2
