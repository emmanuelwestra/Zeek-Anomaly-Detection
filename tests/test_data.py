from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import synthetic_flows, write_test_project

from ndr.config import load_config
from ndr.data import prepare, source_files


def _shard(config_path: Path) -> Path:
    config = load_config(config_path)
    return source_files(config["data"]["urls"], config["data"]["raw_dir"])[0].path


def test_identical_canonical_duplicates_are_collapsed(tmp_path) -> None:
    config_path = write_test_project(tmp_path)
    shard = _shard(config_path)
    frame = pd.read_parquet(shard)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(shard, index=False)

    manifest = prepare(load_config(config_path))

    assert manifest["train_rows"] + manifest["validation_rows"] == 28
    prepared = pd.read_parquet(tmp_path / "processed/train.parquet")
    assert not prepared.duplicated(["uid", "ts"]).any()
    inventory = pd.read_csv(tmp_path / "artifacts/reports/source_inventory.csv")
    assert inventory.columns.tolist() == ["source", "path_within_raw_dir", "rows"]
    assert inventory.loc[0, "path_within_raw_dir"].startswith("UWF-ZeekData22/")
    assert not Path(inventory.loc[0, "path_within_raw_dir"]).is_absolute()


def test_conflicting_duplicate_identity_is_rejected(tmp_path) -> None:
    config_path = write_test_project(tmp_path)
    shard = _shard(config_path)
    frame = pd.read_parquet(shard)
    conflict = frame.iloc[[0]].copy()
    conflict["duration"] += 1
    pd.concat([frame, conflict], ignore_index=True).to_parquet(shard, index=False)

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare(load_config(config_path))


def test_attack_labeled_baseline_is_rejected(tmp_path) -> None:
    config_path = write_test_project(tmp_path)
    shard = _shard(config_path)
    frame = synthetic_flows()
    frame.loc[0, "label_tactic"] = "reconnaissance"
    frame.to_parquet(shard, index=False)

    with pytest.raises(ValueError, match="baseline contains 1 attack"):
        prepare(load_config(config_path))


def test_manifest_rejects_duplicate_and_missing_entries(tmp_path) -> None:
    config_path = write_test_project(tmp_path)
    config = load_config(config_path)
    shard = _shard(config_path)
    urls = config["data"]["urls"]
    urls.write_text(f"{shard}\n{shard}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate source path"):
        source_files(urls, config["data"]["raw_dir"])

    urls.write_text(str(shard.with_name("missing.parquet")) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing source file"):
        source_files(urls, config["data"]["raw_dir"])
