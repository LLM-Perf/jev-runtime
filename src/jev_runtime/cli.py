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


def output(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def admin_call(url: str, path: str, body: dict | None = None):
    key = os.environ.get("JEV_ADMIN_KEY")
    if not key:
        raise typer.BadParameter("Set JEV_ADMIN_KEY; secrets are not accepted in CLI arguments")
    with httpx.Client(
        base_url=url, timeout=180, headers={"Authorization": f"Bearer {key}"}
    ) as client:
        result = client.get(path) if body is None else client.post(path, json=body)
        result.raise_for_status()
        output(result.json())


@app.command()
def serve(config: Path):
    """Run the independent gateway from an explicit configuration file."""
    import uvicorn

    from jev_runtime.api import create_app

    settings = load_settings(config)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


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
                    "template_digest": runtime.compiler.template_digest,
                }
            )
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
