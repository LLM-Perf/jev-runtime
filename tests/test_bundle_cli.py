from typer.testing import CliRunner

from jev_runtime import cli
from jev_runtime.schema import Bundle


def test_build_remote_preserves_serving_profile_and_refuses_overwrite(
    tmp_path, bundle, monkeypatch
):
    calls = []

    def profile(url, path):
        calls.append((url, path))
        return {"model": bundle.model.model_dump(mode="json")}

    monkeypatch.setattr(cli, "admin_request", profile)
    destination = tmp_path / "bundle.json"
    args = [
        "bundle", "build-remote", str(destination), "--url", "http://localhost/plugin",
        "--name", "bound", "--version", "2",
    ]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    actual = Bundle.model_validate_json(destination.read_text())
    assert actual.model == bundle.model and actual.reference == "bound@2"
    assert calls == [("http://localhost/plugin", "/admin/profile")]
    assert CliRunner().invoke(cli.app, args).exit_code != 0
    assert Bundle.model_validate_json(destination.read_text()) == actual
