from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from deployment.release import PACKAGES, build, install, inventory, resolve, sha, target, verify


def git(repo, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Jev Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("release")
    repo = root / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("Test project\n")
    for name, (directory, prefix) in PACKAGES.items():
        module = repo / prefix
        module.mkdir(parents=True)
        (module / "__init__.py").write_text('SOURCE = "committed"\n')
        package_root = Path(prefix).relative_to(directory) if directory != "." else Path(prefix)
        (repo / directory / "pyproject.toml").write_text(
            '[build-system]\nrequires=["hatchling>=1.26"]\nbuild-backend="hatchling.build"\n'
            f'[project]\nname="{name}"\nversion="0.1.0a1"\nrequires-python=">=3.11"\n'
            f'[tool.hatch.build.targets.wheel]\npackages=["{package_root}"]\n'
        )
    git(repo, "init")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Fixture")
    commit = git(repo, "rev-parse", "HEAD")
    # The working checkout differs; the artifact must still contain the Git revision.
    (repo / "src/jev_runtime/__init__.py").write_text('SOURCE = "uncommitted"\n')
    output = root / "bundle"
    result = build(repo, commit, output)
    return repo, output, result


def test_build_uses_immutable_git_source_and_all_packages(built):
    repo, output, result = built
    manifest = verify(output, result["manifest_sha256"], "jev-wheels-v1")
    assert manifest["source_commit"] == result["source_commit"]
    assert set(manifest["distributions"]) == set(PACKAGES)
    for distribution in manifest["distributions"].values():
        assert distribution["verified_python_files"] == 1
        with zipfile.ZipFile(output / distribution["wheel"]) as wheel:
            source = [n for n in wheel.namelist() if n.endswith("__init__.py")]
            assert wheel.read(source[0]) == b'SOURCE = "committed"\n'
    assert '"uncommitted"' in (repo / "src/jev_runtime/__init__.py").read_text()
    with pytest.raises(FileExistsError):
        build(repo, result["source_commit"], output)
    verify(output, result["manifest_sha256"])


@pytest.mark.parametrize("change", ["wheel", "extra", "nested_manifest", "manifest", "symlink"])
def test_tampered_release_rejected_before_resolving(built, tmp_path, change):
    _, original, result = built
    bundle = tmp_path / "bundle"
    shutil.copytree(original, bundle)
    if change == "wheel":
        next((bundle / "wheels").glob("*.whl")).write_bytes(b"changed")
    elif change == "manifest":
        (bundle / "manifest.json").write_text("{}")
    elif change == "symlink":
        (bundle / "outside").symlink_to(original / "manifest.json")
    elif change == "nested_manifest":
        (bundle / "wheels/manifest.json").write_text("unlisted")
    else:
        (bundle / "extra.whl").write_bytes(b"unlisted")
    output = tmp_path / "resolved"
    with pytest.raises(ValueError):
        resolve(bundle, result["manifest_sha256"], output)
    assert not output.exists()


def test_manifest_paths_and_distribution_metadata_must_match_payload(built, tmp_path):
    _, original, _ = built
    bundle = tmp_path / "bundle"
    shutil.copytree(original, bundle)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["distributions"]["jev-runtime-core"]["wheel"] = "../../outside.whl"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside"):
        verify(bundle, sha(path))
    manifest["distributions"]["jev-runtime-core"]["wheel"] = next(
        n for n in manifest["files"] if "core" in n and n.endswith(".whl")
    )
    manifest["distributions"]["jev-runtime-core"]["version"] = "1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata"):
        verify(bundle, sha(path))


def fake_lock(bundle, destination):
    shutil.copytree(bundle, destination)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(kind="jev-gateway-lock-v1", target=target())
    manifest["files"] = inventory(destination)
    path.write_text(json.dumps(manifest))
    return path


def test_install_never_replaces_existing_environment(built, tmp_path):
    _, bundle, _ = built
    locked = tmp_path / "locked"
    manifest = fake_lock(bundle, locked)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "user-file"
    marker.write_text("preserve me")
    with pytest.raises(FileExistsError):
        install(locked, sha(manifest), existing)
    assert marker.read_text() == "preserve me"
    assert list(existing.iterdir()) == [marker]


def test_wrong_target_or_manifest_rejected_before_environment_creation(built, tmp_path):
    _, bundle, _ = built
    locked = tmp_path / "locked"
    path = fake_lock(bundle, locked)
    manifest = json.loads(path.read_text())
    manifest["target"]["machine"] = "different-machine"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "env"
    with pytest.raises(ValueError, match="ABI"):
        install(locked, sha(path), output)
    assert not output.exists()
    with pytest.raises(ValueError, match="hash mismatch"):
        install(locked, "0" * 64, output)
    assert not output.exists()


def test_vendored_metadata_does_not_shadow_root_distribution(tmp_path):
    from deployment.release import gateway_extras, wheel_info

    wheel = tmp_path / "old_core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "jev_runtime_core-0.1.0a1.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: jev-runtime-core\nVersion: 0.1.0a1\n"
            "Provides-Extra: tokenizers\n",
        )
        archive.writestr(
            "jev_runtime/_vendor/other-1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: other\nVersion: 1.0\n",
        )
    assert wheel_info(wheel)["name"] == "jev-runtime-core"
    # Older core releases do not provide conversion; do not ask pip to silently ignore it.
    assert gateway_extras(wheel) == ["tokenizers"]
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("unexpected-1.dist-info/METADATA", "Name: unexpected\nVersion: 1\n")
    with pytest.raises(ValueError, match="one metadata"):
        wheel_info(wheel)
