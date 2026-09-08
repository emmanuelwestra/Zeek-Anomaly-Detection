from __future__ import annotations

import pandas as pd

from ndr.config import PROJECT_ROOT, load_grid
from ndr.workflow import scaling_fits


def test_compact_grids_contain_promoted_winners_and_unique_combinations() -> None:
    oc = load_grid(PROJECT_ROOT / "configs/ocsvm-grid.yaml", "search")
    ae = load_grid(PROJECT_ROOT / "configs/ae-grid.yaml", "search")
    assert 0.02 in oc["gamma"] and 0.001 in oc["nu"]
    assert 0.5 in ae["training_numeric_weight"] and 0.1 in ae["scoring_numeric_weight"]
    combinations = {(g, n, q) for g in oc["gamma"] for n in oc["nu"] for q in oc["quantiles"]}
    assert len(combinations) == len(oc["gamma"]) * len(oc["nu"]) * len(oc["quantiles"])
    names = {f"g{g:g}_nu{n:g}_q{q:g}" for g, n, q in combinations}
    assert len(names) == len(combinations)
    ae_names = {
        f"train{train:g}_score{score:g}_seed{seed}"
        for train in ae["training_numeric_weight"]
        for score in ae["scoring_numeric_weight"]
        for seed in ae["seeds"]
    }
    assert len(ae_names) == 27


def test_scaling_fits_recover_power_and_linear_shapes() -> None:
    rows = pd.DataFrame({
        "phase": ["train"] * 6,
        "model": ["ocsvm"] * 3 + ["ae"] * 3,
        "rows": [100, 200, 400, 100, 200, 400],
        "seconds": [1, 4, 16, 1, 2, 4],
    })
    fits = scaling_fits(rows)
    assert abs(fits["ocsvm_power"]["b"] - 2) < 1e-9
    assert abs(fits["ae_linear"]["slope"] - 0.01) < 1e-9


def test_url_manifest_contains_exactly_fifty_primary_shards() -> None:
    lines = (PROJECT_ROOT / "data/uwf_urls.txt").read_text().splitlines()
    assert len(lines) == 50
    assert len(set(lines)) == 50
    assert all(line.endswith(".parquet") and "/parquet/" in line for line in lines)
