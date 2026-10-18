import pytest

from deployment.sglang_text_dependencies import (
    CUDA_SUBSTITUTIONS,
    REMOVED_REQUIREMENTS,
    patch_pyproject,
)


def fixture_text() -> str:
    requirements = "\n".join(f'  "{name}",' for name in sorted(REMOVED_REQUIREMENTS))
    return (
        "[project]\n"
        'name = "sglang"\n'
        "dependencies = [\n"
        f"{requirements}\n"
        '  "torch==2.13.0",\n'
        '  "cuda-python>=13.0",\n'
        '  "flashinfer_python[cu13]",\n'
        '  "nvidia-cutlass-dsl[cu13]",\n'
        "]\n\n"
        "[project.optional-dependencies]\n"
        'test = ["datasets"]\n'
    )


def test_patch_removes_only_base_non_text_dependencies(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(fixture_text())

    assert set(patch_pyproject(path)) == REMOVED_REQUIREMENTS
    result = path.read_text()

    for name in REMOVED_REQUIREMENTS:
        assert f'  "{name}",' not in result
    for old, new in CUDA_SUBSTITUTIONS:
        assert old not in result
        assert new in result
    assert 'test = ["datasets"]' in result
    assert '"torch==2.13.0"' in result


def test_patch_fails_closed_when_upstream_dependencies_change(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(fixture_text().replace('  "timm",\n', ""))

    with pytest.raises(RuntimeError, match="no longer matches"):
        patch_pyproject(path)
