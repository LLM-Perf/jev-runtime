import json
from dataclasses import asdict

from typer.testing import CliRunner

from jev_runtime.calibration import LabeledScores
from jev_runtime.cli import app
from jev_runtime.schema import Bundle, Question


def test_calibration_cli_produces_bound_version_and_report(tmp_path, bundle):
    bundle = Bundle.model_validate(
        {
            **bundle.model_dump(),
            "candidate_policy": "fixed",
            "questions": [
                Question(id="q", type="boolean", instruction="Is this true?").model_dump()
            ],
        }
    )
    source = tmp_path / "source.json"
    source.write_text(bundle.model_dump_json())
    for split in ("fit", "heldout"):
        rows = [
            asdict(LabeledScores(f"{split}-{i}", f"{split}-{i}", (0, 0), int(i % 4 == 0), "q"))
            for i in range(40)
        ]
        (tmp_path / f"{split}.json").write_text(
            json.dumps({"scoring_contract_digest": bundle.scoring_contract_digest, "rows": rows})
        )
    destination = tmp_path / "artifacts"
    args = [
        "calibration",
        "fit",
        str(source),
        str(tmp_path / "fit.json"),
        str(tmp_path / "heldout.json"),
        str(destination),
        "--method",
        "platt",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    calibrated = Bundle.model_validate_json((destination / "bundle.json").read_text())
    assert calibrated.version == bundle.version + 1
    assert calibrated.calibration.contract_digest == bundle.scoring_contract_digest
    assert json.loads((destination / "report.json").read_text())["fit_samples"] == 40
    assert CliRunner().invoke(app, args).exit_code != 0
