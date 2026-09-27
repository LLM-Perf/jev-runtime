from __future__ import annotations

import json
import shutil
import zipfile

import httpx
import pytest

from deployment import container_gateway, image_context, release

BASE = "docker.io/library/python:3.12-slim@sha256:" + "a" * 64


@pytest.fixture
def locked(tmp_path):
    root = tmp_path / "locked"
    (root / "wheels").mkdir(parents=True)
    wheel = root / "wheels/jev_runtime_core-0.1.0a1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "jev_runtime_core-0.1.0a1.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: jev-runtime-core\nVersion: 0.1.0a1\n",
        )
    manifest = {
        "schema_version": 1,
        "kind": "jev-gateway-lock-v1",
        "source_commit": "a" * 40,
        "target": {
            "platform": "linux",
            "implementation": "CPython",
            "python": "3.12",
            "machine": "x86_64",
            "soabi": "cpython-312-x86_64-linux-gnu",
        },
        "distributions": {
            "jev-runtime-core": {**release.wheel_info(wheel), "wheel": "wheels/" + wheel.name}
        },
        "files": release.inventory(root),
    }
    release.save(root / "manifest.json", manifest)
    return root, release.sha(root / "manifest.json")


def test_context_snapshot_binds_linux_lock_and_never_copies_checkout(locked, tmp_path):
    source, digest = locked
    (tmp_path / "credentials.env").write_text("secret")
    context = tmp_path / "context"
    result = image_context.prepare(source, digest, BASE, context)
    manifest = image_context.verify(context, result["manifest_sha256"])
    assert manifest["image"]["base_image"] == BASE
    assert manifest["image"]["platform"] == "linux/amd64" and result["image_built"] is False
    assert "credentials.env" not in manifest["files"]
    assert "__JEV_" not in (context / "Dockerfile").read_text()
    assert (context / "Dockerfile").read_text().count(BASE) == 2
    with pytest.raises(FileExistsError):
        image_context.prepare(source, digest, BASE, context)
    # The context is an independent snapshot of the accepted immutable input.
    next((source / "wheels").glob("*.whl")).write_bytes(b"changed after snapshot")
    image_context.verify(context, result["manifest_sha256"])


@pytest.mark.parametrize(
    "tamper", ["dockerfile", "helper", "wheel", "extra", "descriptor", "symlink"]
)
def test_context_rejects_changed_instructions_or_payload(locked, tmp_path, tamper):
    source, digest = locked
    context = tmp_path / "context"
    result = image_context.prepare(source, digest, BASE, context)
    if tamper == "symlink":
        (context / "extra").symlink_to(source / "manifest.json")
    else:
        paths = {
            "dockerfile": context / "Dockerfile",
            "helper": context / "release.py",
            "wheel": next((context / "locked/wheels").glob("*.whl")),
            "extra": context / "unexpected",
            "descriptor": context / "image.json",
        }
        paths[tamper].write_bytes(b"changed")
    with pytest.raises(ValueError):
        image_context.verify(context, result["manifest_sha256"])


@pytest.mark.parametrize(
    "base",
    [
        "python:latest",
        "python@sha256:" + "a" * 63,
        "python@sha256:" + "a" * 64 + "\nRUN something",
        "https://registry/python@sha256:" + "a" * 64,
        "user:password@registry/python@sha256:" + "a" * 64,
    ],
)
def test_base_must_be_one_digest_bound_reference(locked, tmp_path, base):
    source, digest = locked
    context = tmp_path / "context"
    with pytest.raises(ValueError, match="base image"):
        image_context.prepare(source, digest, base, context)
    assert not context.exists()


def test_wrong_platform_or_modified_lock_rejected_before_creation(locked, tmp_path):
    source, digest = locked
    with pytest.raises(ValueError):
        image_context.prepare(source, "0" * 64, BASE, tmp_path / "bad")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["target"]["platform"] = "darwin"
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Linux"):
        image_context.prepare(source, release.sha(source / "manifest.json"), BASE, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.fixture
def configuration(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    values = {
        "backend": "vllm",
        "model_id": "fixture",
        "model_revision": "b" * 40,
        "release_id": "a" * 40,
        "registry_path": str(tmp_path / "registry.db"),
        "tokenizer": str(tokenizer),
        "host": "0.0.0.0",
        "workers": 2,
    }
    config, descriptor = tmp_path / "gateway.json", tmp_path / "image.json"
    config.write_text(json.dumps(values))
    descriptor.write_text(json.dumps({"source_commit": "a" * 40}))
    return config, descriptor


@pytest.mark.parametrize("change", ["release", "registry", "tokenizer", "missing_mount", "host"])
def test_container_configuration_rejects_ambiguous_identity_or_paths(configuration, change):
    config, descriptor = configuration
    data = json.loads(config.read_text())
    if change == "release":
        data["release_id"] = "b" * 40
    elif change == "registry":
        data["registry_path"] = "relative.db"
    elif change == "tokenizer":
        data["tokenizer"] = "organization/remote-model"
    elif change == "missing_mount":
        shutil.rmtree(data["tokenizer"])
    else:
        data["host"] = "external.example"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        container_gateway.configuration(config, descriptor)


def test_container_exec_passes_config_workers_and_grace_without_shell(configuration, monkeypatch):
    calls = []
    monkeypatch.setattr(container_gateway.os, "execv", lambda *args: calls.append(args))
    monkeypatch.setenv("JEV_GRACEFUL_TIMEOUT_SECONDS", "37")
    container_gateway.serve(*configuration)
    executable, argv = calls.pop()
    assert executable == argv[0] and argv[1:4] == ["-I", "-m", "uvicorn"]
    assert argv[argv.index("--workers") + 1] == "2"
    assert argv[argv.index("--timeout-graceful-shutdown") + 1] == "37"
    assert container_gateway.os.environ["JEV_CONFIG"] == str(configuration[0])


@pytest.mark.parametrize(
    "status,body,valid",
    [
        (
            200,
            {"ready": True, "engine": "vllm", "prepared_bundles": ["a@1"], "worker_id": "one"},
            True,
        ),
        (200, {"ready": True}, False),
        (
            200,
            {"ready": True, "engine": "sglang", "prepared_bundles": ["a@1"], "worker_id": "one"},
            False,
        ),
        (200, {"ready": True, "engine": "vllm", "prepared_bundles": [], "worker_id": "one"}, False),
        (503, {"error": {"code": "engine_unavailable"}}, False),
    ],
)
def test_container_probe_requires_semantic_readiness(
    configuration, monkeypatch, status, body, valid
):
    monkeypatch.setenv("JEV_API_KEY", "test-only-secret")
    real_client = httpx.Client

    def handle(request):
        assert str(request.url) == "http://127.0.0.1:8795/ready"
        assert request.headers["authorization"] == "Bearer test-only-secret"
        return httpx.Response(status, json=body)

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(container_gateway.httpx, "Client", client)
    if valid:
        container_gateway.healthcheck(*configuration)
    else:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            container_gateway.healthcheck(*configuration)
