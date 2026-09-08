from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def synthetic_flows(rows: int = 80, *, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamps = pd.Timestamp(start, tz="UTC").timestamp() + np.arange(rows) * 6 * 3600
    attacks = np.arange(rows) >= 28
    return pd.DataFrame({
        "uid": [f"flow-{index}" for index in range(rows)],
        "ts": timestamps,
        "duration": rng.uniform(0.01, 4, rows),
        "orig_bytes": rng.integers(1, 5000, rows),
        "resp_bytes": rng.integers(0, 5000, rows),
        "orig_pkts": rng.integers(1, 50, rows),
        "resp_pkts": rng.integers(0, 50, rows),
        "orig_ip_bytes": rng.integers(1, 6000, rows),
        "resp_ip_bytes": rng.integers(0, 6000, rows),
        "src_port": rng.choice([53, 443, 50000], rows),
        "dest_port": rng.choice([53, 80, 443, 5353], rows),
        "proto": rng.choice(["tcp", "udp"], rows),
        "service": rng.choice(["dns", "http"], rows),
        "conn_state": rng.choice(["SF", "S0"], rows),
        "history": rng.choice(["D", "Dd"], rows),
        "local_orig": rng.choice(["true", "false"], rows),
        "local_resp": rng.choice(["true", "false"], rows),
        "src_ip_zeek": [f"10.0.0.{index % 20 + 1}" for index in range(rows)],
        "dest_ip_zeek": [f"192.168.0.{index % 30 + 1}" for index in range(rows)],
        "community_id": [f"community-{index}" for index in range(rows)],
        "label_tactic": np.where(attacks, "reconnaissance", "none"),
    })


def write_test_project(tmp_path: Path) -> Path:
    raw = tmp_path / "raw/UWF-ZeekData22/parquet/2024-01-01 - 2024-01-22"
    raw.mkdir(parents=True)
    shard = raw / "part.parquet"
    synthetic_flows().to_parquet(shard, index=False)
    urls = tmp_path / "urls.txt"
    urls.write_text(str(shard) + "\n", encoding="utf-8")
    config = {
        "data": {
            "urls": str(urls), "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"), "memory_limit": "1GB", "threads": 1,
        },
        "splits": {
            "baseline_days": 7, "validation_fraction": 0.05,
            "holdout_start": "2025-01-01T00:00:00Z", "window": "1h",
        },
        "features": {
            "top_k_dest_ports": 4, "ocsvm_expected_width": None, "ae_expected_width": None,
            "unknown_category_token": "<unknown>",
        },
        "models": {
            "ocsvm": {"kernel": "rbf", "gamma": 0.02, "nu": 0.05, "scoring_threads": 1},
            "ae": {
                "hidden_dims": [8], "latent_dim": 2, "activation": "relu",
                "normalization": "layernorm", "dropout": 0.0,
                "categorical_loss": "cross_entropy", "training_numeric_weight": 0.5,
                "scoring_numeric_weight": 0.1, "epochs": 2, "batch_size": 8,
                "learning_rate": 0.01, "weight_decay": 0.0, "patience": 1,
                "internal_validation_fraction": 0.1, "random_state": 7, "device": "cpu",
            },
        },
        "thresholds": {"ocsvm_quantile": 0.9, "ae_quantile": 0.9},
        "search": {"target_recall": 0.8, "ocsvm": {}, "ae": {}},
        "fusion": {"alpha_start": 0, "alpha_stop": 1, "alpha_step": 0.5},
        "benchmark": {
            "training_rows": [10, 20], "inference_rows": [10, 20],
            "inference_repetitions": 1, "warmup_rows": 2,
        },
        "artifacts": {
            "models_dir": str(tmp_path / "artifacts/models"),
            "preprocessors_dir": str(tmp_path / "artifacts/preprocessors"),
            "thresholds_dir": str(tmp_path / "artifacts/thresholds"),
            "score_cache_dir": str(tmp_path / "artifacts/score_cache"),
            "reports_dir": str(tmp_path / "artifacts/reports"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
