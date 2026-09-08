from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_SECTIONS = {
    "data",
    "splits",
    "features",
    "models",
    "thresholds",
    "search",
    "fusion",
    "benchmark",
    "artifacts",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config and resolve every configured path from the repository root."""

    config_path = Path(path) if path else PROJECT_ROOT / "configs/default.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    unexpected = set(config) - ALLOWED_SECTIONS
    if unexpected:
        raise ValueError(f"Unsupported config sections: {sorted(unexpected)}")
    config = deepcopy(config)
    for section, keys in {
        "data": ("urls", "raw_dir", "processed_dir"),
        "artifacts": (
            "models_dir",
            "preprocessors_dir",
            "thresholds_dir",
            "score_cache_dir",
            "reports_dir",
        ),
    }.items():
        for key in keys:
            value = Path(config[section][key])
            config[section][key] = value if value.is_absolute() else PROJECT_ROOT / value
    return config


def load_grid(path: str | Path, section: str) -> dict[str, Any]:
    grid_path = Path(path)
    if not grid_path.is_absolute():
        grid_path = PROJECT_ROOT / grid_path
    with grid_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    grid = value.get(section)
    if not isinstance(grid, dict):
        raise ValueError(f"{grid_path} must contain a {section!r} mapping")
    return grid


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
