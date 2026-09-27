from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import typer

from jev_runtime.config import build_runtime, load_settings, model_identity
from jev_runtime.schema import Bundle, DecisionRequest

app = typer.Typer(help="Typed decisions, immutable bundles and reproducible engine validation")
bundle_app = typer.Typer(help="Bundle build, prepare, activate, disable and retire")
app.add_typer(bundle_app, name="bundle")
calibration_app = typer.Typer(help="Collect fixed-task scores and fit held-out calibration")
app.add_typer(calibration_app, name="calibration")
recovery_app = typer.Typer(help="Inspect and recover durable leases, including before API startup")
app.add_typer(recovery_app, name="recovery")
adapter_app = typer.Typer(help="Register, load, drain and unload immutable local LoRA artifacts")
app.add_typer(adapter_app, name="adapter")
tokenizer_app = typer.Typer(help="Prepare immutable tokenizer profiles from local checkpoint data")
app.add_typer(tokenizer_app, name="tokenizer")


@tokenizer_app.command("convert-glm4")
def tokenizer_convert_glm4(model_dir: Path, destination: Path):
    """Convert GLM4 tiktoken data to a verified fast tokenizer in a new directory."""
    from jev_runtime.tokenizer_profiles import convert_glm4_tokenizer

    output(convert_glm4_tokenizer(model_dir, destination))


def output(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def admin_call(url: str, path: str, body: dict | None = None):
    output(admin_request(url, path, body))


def admin_request(url: str, path: str, body: dict | None = None):
    key = os.environ.get("JEV_ADMIN_KEY")
    if not key:
        raise typer.BadParameter("Set JEV_ADMIN_KEY; secrets are not accepted in CLI arguments")
    with httpx.Client(
        base_url=url, timeout=180, headers={"Authorization": f"Bearer {key}"}
    ) as client:
        result = client.get(path) if body is None else client.post(path, json=body)
        result.raise_for_status()
        return result.json()


@app.command()
def serve(config: Path):
    """Run the independent gateway from an explicit configuration file."""
    import uvicorn

    from jev_runtime.api import create_app

    settings = load_settings(config)
    if settings.workers == 1:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    else:
        os.environ["JEV_CONFIG"] = str(config.resolve())
        uvicorn.run(
            "jev_runtime.api:create_app_from_env",
            factory=True,
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
        )


def serve_entrypoint():
    app()


@app.command()
def inspect(config: Path):
    """Inspect configured engine and tokenizer identity; does not certify task quality."""

    async def run():
        runtime = await build_runtime(load_settings(config))
        try:
            await runtime.start()
            output(
                {
                    "capabilities": runtime.capabilities.model_dump(mode="json"),
                    "tokenizer_digest": runtime.compiler.tokenizer_digest,
                    "tokenizer_implementation_digest": (
                        runtime.compiler.tokenizer_implementation_digest
                    ),
                    "template_digest": runtime.compiler.template_digest,
                }
            )
        finally:
            await runtime.close()

    asyncio.run(run())


@recovery_app.command("list")
def recovery_list(config: Path):
    """Inspect the local dispatch journal without starting an engine or API worker."""
    from jev_runtime.registry import Registry

    output({"requests": Registry(load_settings(config).registry_path).recovery_candidates()})


@recovery_app.command("recover")
def recovery_recover(config: Path, request_id: str):
    """Abort an eligible recorded request against its original configured backend."""

    async def run():
        runtime = await build_runtime(load_settings(config))
        try:
            # Do not call start/bootstrap: a crashed bootstrap can itself leave
            # PREPARING state. Recovery checks identity and dispatch ownership.
            await runtime.backend.probe()
            output({"recovered": await runtime.recover_cancelled(request_id)})
        finally:
            await runtime.close()

    asyncio.run(run())


@bundle_app.command("build")
def bundle_build(config: Path, destination: Path, name: str = "default", version: int = 1):
    """Build an uncalibrated dynamic bundle; preparation runs a real model canary."""
    from jev_runtime.config import load_compiler

    settings = load_settings(config)
    compiler = load_compiler(settings)
    bundle = Bundle(id=name, version=version, model=model_identity(settings, compiler))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(bundle.model_dump_json(indent=2) + "\n")
    output({"path": str(destination), "reference": bundle.reference, "digest": bundle.digest})


@bundle_app.command("build-remote")
def bundle_build_remote(
    destination: Path,
    url: str = "http://127.0.0.1:8795",
    name: str = "default",
    version: int = 1,
    adapter: str | None = None,
):
    """Build against the running worker's tokenizer profile without activating it."""
    profile = admin_request(url, "/admin/profile")
    if not profile.get("model"):
        raise typer.BadParameter("The serving runtime does not expose a configured model identity")
    model = dict(profile["model"])
    if adapter:
        matches = [
            row for row in admin_request(url, "/admin/adapters") if row["reference"] == adapter
        ]
        if len(matches) != 1:
            raise typer.BadParameter("Register the immutable adapter before building its bundle")
        artifact = matches[0]["binding"]["artifact"]
        if (artifact["base_model_id"], artifact["base_model_revision"]) != (
            model["id"],
            model["revision"],
        ):
            raise typer.BadParameter("Adapter and serving base model differ")
        model.update(adapter_id=artifact["id"], adapter_revision=artifact["revision"])
    bundle = Bundle(id=name, version=version, model=model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as file:
        file.write(bundle.model_dump_json(indent=2) + "\n")
    output({"path": str(destination), "reference": bundle.reference, "digest": bundle.digest})


@adapter_app.command("list")
def adapter_list(url: str = "http://127.0.0.1:8795"):
    admin_call(url, "/admin/adapters")


@adapter_app.command("register")
def adapter_register(name: str, source: str, url: str = "http://127.0.0.1:8795"):
    """Copy an allowed directory on the server into its immutable artifact store."""
    admin_call(url, "/admin/adapters/register", {"id": name, "source": source})


@adapter_app.command("load")
def adapter_load(reference: str, url: str = "http://127.0.0.1:8795"):
    admin_call(url, "/admin/adapters/load", {"reference": reference})


@adapter_app.command("unload")
def adapter_unload(reference: str, url: str = "http://127.0.0.1:8795", recover: bool = False):
    """Disable all referencing aliases and drain leases before calling this command."""
    admin_call(url, "/admin/adapters/unload", {"reference": reference, "recover": recover})


@bundle_app.command("upload")
def bundle_upload(path: Path, url: str = "http://127.0.0.1:8795"):
    bundle = Bundle.model_validate_json(path.read_text())
    admin_call(url, "/admin/bundles", bundle.model_dump(mode="json"))


@bundle_app.command("prepare")
def bundle_prepare(reference: str, url: str = "http://127.0.0.1:8795"):
    admin_call(url, "/admin/bundles/prepare", {"reference": reference})


@bundle_app.command("activate")
def bundle_activate(
    reference: str, alias: str, expected_generation: int, url: str = "http://127.0.0.1:8795"
):
    admin_call(
        url,
        "/admin/bundles/activate",
        {"reference": reference, "alias": alias, "expected_generation": expected_generation},
    )


@bundle_app.command("disable")
def bundle_disable(alias: str, expected_generation: int, url: str = "http://127.0.0.1:8795"):
    admin_call(
        url, "/admin/bundles/disable", {"alias": alias, "expected_generation": expected_generation}
    )


@bundle_app.command("retire")
def bundle_retire(reference: str, url: str = "http://127.0.0.1:8795"):
    admin_call(url, "/admin/bundles/retire", {"reference": reference})


@bundle_app.command("list")
def bundle_list(url: str = "http://127.0.0.1:8795"):
    admin_call(url, "/admin/bundles")


@app.command()
def decide(path: Path, url: str = "http://127.0.0.1:8795"):
    request = DecisionRequest.model_validate_json(path.read_text())
    key = os.environ.get("JEV_API_KEY")
    result = httpx.post(
        url.rstrip("/") + "/v1/decisions",
        json=request.model_dump(mode="json"),
        timeout=request.execution.timeout_ms / 1000 + 10,
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    result.raise_for_status()
    output(result.json())


@calibration_app.command("collect")
def calibration_collect(config: Path, bundle_path: Path, dataset: Path, destination: Path):
    from jev_runtime.evaluation import collect_scores, read_samples

    if destination.exists():
        raise typer.BadParameter("Destination exists; choose a new artifact path")
    bundle = Bundle.model_validate_json(bundle_path.read_text())
    report = asyncio.run(collect_scores(load_settings(config), bundle, read_samples(dataset)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    output({"scores": str(destination), "rows": len(report["rows"])})


@calibration_app.command("fit")
def calibration_fit(
    bundle_path: Path,
    fit_scores: Path,
    heldout_scores: Path,
    destination: Path,
    method: str = "temperature",
):
    from jev_runtime.calibration import bind_calibration, fit_platt, fit_temperature
    from jev_runtime.evaluation import read_scores

    if method not in {"temperature", "platt"}:
        raise typer.BadParameter("Method must be temperature or platt")
    if destination.exists():
        raise typer.BadParameter("Destination exists; choose a new artifact directory")
    bundle = Bundle.model_validate_json(bundle_path.read_text())
    fit = read_scores(json.loads(fit_scores.read_text()), bundle)
    heldout = read_scores(json.loads(heldout_scores.read_text()), bundle)
    fitter = fit_platt if method == "platt" else fit_temperature
    artifact, report = fitter(fit, heldout, bundle.scoring_contract_digest)
    calibrated = bind_calibration(bundle, artifact)
    calibrated = Bundle.model_validate({**calibrated.model_dump(), "version": bundle.version + 1})
    destination.mkdir(parents=True)
    (destination / "bundle.json").write_text(calibrated.model_dump_json(indent=2) + "\n")
    (destination / "calibration.json").write_text(artifact.model_dump_json(indent=2) + "\n")
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    output(
        {
            "artifacts": str(destination),
            "bundle": calibrated.reference,
            "heldout_uncalibrated_nll": report["heldout_uncalibrated"]["nll"],
            "heldout_calibrated_nll": report["heldout_calibrated"]["nll"],
        }
    )
