from __future__ import annotations

from conftest import write_test_project

from ndr.cli import parser, run


def _run(config, *arguments):
    return run(parser().parse_args(["--config", str(config), *arguments]))


def test_prepare_train_evaluate_and_fuse(tmp_path) -> None:
    config = write_test_project(tmp_path)
    manifest = _run(config, "prepare")
    assert manifest["train_rows"] == 26
    assert manifest["validation_rows"] == 2
    training = _run(config, "train", "--model", "all")
    assert {row["model"] for row in training} == {"ocsvm", "ae"}
    ocsvm = _run(config, "evaluate", "--model", "ocsvm", "--period", "development")
    assert ocsvm["rows"] == 52
    fusion_grid = tmp_path / "fusion.yaml"
    fusion_grid.write_text(
        "fusion:\n  alpha_start: 0.0\n  alpha_stop: 1.0\n  alpha_step: 0.5\n",
        encoding="utf-8",
    )
    selection = _run(config, "search", "--model", "fusion", "--grid", str(fusion_grid))
    assert 0 <= selection["alpha"] <= 1
    fusion = _run(config, "evaluate", "--model", "fusion", "--period", "development")
    assert fusion["rows"] == 52
