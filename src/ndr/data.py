from __future__ import annotations

import math
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import pyarrow.parquet as pq

from ndr.config import write_yaml

SOURCE_PRIORITY = {
    "UWF-ZeekData22": 0,
    "UWF-ZeekDataFall22": 1,
    "UWF-ZeekData24": 2,
    "UWF-ZeekDataFall24-2": 3,
    "UWF-ZeekDataSum25-1": 4,
    "UWF-ZeekDataSum25-2": 5,
}
NUMERIC = {
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "src_port",
    "dest_port",
}
CANONICAL = (
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "src_port",
    "dest_port",
    "protocol",
    "service",
    "conn_state",
    "history",
    "local_orig",
    "local_resp",
    "src_ip",
    "dest_ip",
    "community_id",
)
ALIASES = {
    "protocol": ("protocol", "proto"),
    "src_port": ("src_port", "src_port_zeek"),
    "dest_port": ("dest_port", "dest_port_zeek"),
    "src_ip": ("src_ip", "src_ip_zeek"),
    "dest_ip": ("dest_ip", "dest_ip_zeek"),
    "mitre_attack_tactics": ("mitre_attack_tactics", "label_tactic"),
}
WEEK_SECONDS = 7 * 86_400.0
SUNDAY_ANCHOR = 3 * 86_400.0


@dataclass(frozen=True)
class SourceFile:
    source: str
    path: Path
    week_start: pd.Timestamp
    week_end: pd.Timestamp
    priority: int


@dataclass(frozen=True)
class Corpus:
    root: Path

    @property
    def train_path(self) -> Path:
        return self.root / "train.parquet"

    @property
    def validation_path(self) -> Path:
        return self.root / "validation.parquet"

    def train(self) -> pd.DataFrame:
        return pd.read_parquet(self.train_path)

    def validation(self) -> pd.DataFrame:
        return pd.read_parquet(self.validation_path)

    def parts(self) -> list[Path]:
        return sorted((self.root / "deployment").glob("week=*/part.parquet"))

    def iter_period(self, period: str, holdout_start: str) -> Iterator[pd.DataFrame]:
        if period not in {"development", "holdout"}:
            raise ValueError("period must be 'development' or 'holdout'")
        boundary = pd.Timestamp(holdout_start).timestamp()
        for path in self.parts():
            frame = pd.read_parquet(path)
            mask = frame["ts"].lt(boundary) if period == "development" else frame["ts"].ge(boundary)
            selected = frame.loc[mask]
            if not selected.empty:
                yield selected


def source_files(url_list: Path, raw_dir: Path) -> list[SourceFile]:
    """Resolve the exact downloaded files declared by ``uwf_urls.txt``."""

    lines = [line.strip() for line in url_list.read_text(encoding="utf-8").splitlines()]
    entries: list[SourceFile] = []
    seen: set[Path] = set()
    for index, line in enumerate(line for line in lines if line and not line.startswith("#")):
        parsed = urlparse(line)
        if parsed.scheme in {"http", "https"}:
            parts = Path(unquote(parsed.path)).parts
            try:
                data_index = parts.index("data")
                source, kind, week, filename = parts[data_index + 1 : data_index + 5]
            except (ValueError, IndexError) as error:
                raise ValueError(f"Invalid UWF URL on line {index + 1}: {line}") from error
            if kind != "parquet" or not filename.endswith(".parquet"):
                raise ValueError(f"URL is not a primary Parquet shard: {line}")
            path = raw_dir / source / kind / week / filename
        else:
            path = Path(line)
            source = path.parents[2].name if len(path.parents) >= 3 else "synthetic"
            week = path.parent.name
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Duplicate source path: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}) - (\d{4}-\d{2}-\d{2})", week)
        if not match:
            raise ValueError(f"Cannot infer capture dates from directory {week!r}")
        start = pd.Timestamp(match.group(1), tz="UTC")
        end = pd.Timestamp(match.group(2), tz="UTC")
        if end <= start:
            raise ValueError(f"Invalid capture interval for {path}")
        seen.add(path)
        entries.append(SourceFile(source, path, start, end, SOURCE_PRIORITY.get(source, index)))
    if not entries:
        raise ValueError(f"No source URLs in {url_list}")
    return entries


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Build leakage-controlled train, validation, and weekly deployment splits."""

    import duckdb

    data = config["data"]
    splits = config["splits"]
    files = source_files(data["urls"], data["raw_dir"])
    output = Path(data["processed_dir"])
    work = output / ".work"
    reports = Path(config["artifacts"]["reports_dir"])
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    deployment = output / "deployment"
    if deployment.exists():
        shutil.rmtree(deployment)
    deployment.mkdir()

    connection = duckdb.connect(str(work / "prepare.duckdb"))
    connection.execute(f"SET memory_limit={sql(str(data.get('memory_limit', '4GB')))}")
    connection.execute(f"SET threads={int(data.get('threads', 2))}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("CREATE OR REPLACE VIEW raw_flows AS " + " UNION ALL ".join(_source_select(f) for f in files))

    core = ", ".join(CANONICAL)
    conflict_query = (
        f"SELECT uid, ts, count(DISTINCT hash({core})) AS variants FROM raw_flows "
        "GROUP BY uid, ts HAVING variants > 1 ORDER BY ts, uid"
    )
    conflict_path = reports / "data_conflicts.csv"
    connection.execute(f"COPY ({conflict_query}) TO {sql(str(conflict_path))} (HEADER, DELIMITER ',')")
    if connection.execute(conflict_query + " LIMIT 1").fetchone():
        connection.close()
        raise ValueError(f"Conflicting duplicate flows found; see {conflict_path}")

    week_expr = f"floor((ts - {SUNDAY_ANCHOR}) / {WEEK_SECONDS}) * {WEEK_SECONDS} + {SUNDAY_ANCHOR}"
    weeks = connection.execute(f"SELECT DISTINCT {week_expr} FROM raw_flows ORDER BY 1").fetchall()
    for index, (week_start,) in enumerate(weeks):
        start = float(week_start)
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW week_raw AS SELECT * FROM raw_flows "
            f"WHERE ts >= {start!r} AND ts < {start + WEEK_SECONDS!r}"
        )
        prefix = "CREATE OR REPLACE TABLE canonical_flows AS " if index == 0 else "INSERT INTO canonical_flows "
        connection.execute(prefix + _canonical_query())

    duplicate_query = "SELECT uid, ts, count(*) AS rows FROM canonical_flows GROUP BY uid, ts HAVING rows > 1"
    duplicate_path = reports / "canonical_duplicates.csv"
    connection.execute(f"COPY ({duplicate_query}) TO {sql(str(duplicate_path))} (HEADER, DELIMITER ',')")
    if connection.execute(duplicate_query + " LIMIT 1").fetchone():
        connection.close()
        raise ValueError(f"Canonical identities are not unique; see {duplicate_path}")

    minimum = float(connection.execute("SELECT min(ts) FROM canonical_flows").fetchone()[0])
    baseline_end = minimum + float(splits.get("baseline_days", 7)) * 86_400
    attacks = int(connection.execute(
        "SELECT count(*) FROM canonical_flows WHERE ts < ? AND is_attack", [baseline_end]
    ).fetchone()[0])
    if attacks:
        connection.close()
        raise ValueError(f"The baseline contains {attacks} attack-labeled flows")
    baseline_rows = int(connection.execute(
        "SELECT count(*) FROM canonical_flows WHERE ts < ?", [baseline_end]
    ).fetchone()[0])
    validation_rows = max(1, math.ceil(baseline_rows * float(splits["validation_fraction"])))
    train_rows = baseline_rows - validation_rows
    if train_rows < 1:
        raise ValueError("The baseline is too small to split")
    connection.execute(
        "CREATE OR REPLACE VIEW ranked_baseline AS SELECT *, "
        "row_number() OVER (ORDER BY ts, uid) AS split_row FROM canonical_flows "
        f"WHERE ts < {baseline_end!r}"
    )
    connection.execute(
        f"COPY (SELECT * EXCLUDE split_row FROM ranked_baseline WHERE split_row <= {train_rows}) "
        f"TO {sql(str(output / 'train.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT * EXCLUDE split_row FROM ranked_baseline WHERE split_row > {train_rows}) "
        f"TO {sql(str(output / 'validation.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    deployment_weeks = connection.execute(
        f"SELECT DISTINCT {week_expr} FROM canonical_flows WHERE ts >= ? ORDER BY 1", [baseline_end]
    ).fetchall()
    deployment_rows = 0
    for (week_start,) in deployment_weeks:
        start = float(week_start)
        count = int(connection.execute(
            "SELECT count(*) FROM canonical_flows WHERE ts >= ? AND ts >= ? AND ts < ?",
            [baseline_end, start, start + WEEK_SECONDS],
        ).fetchone()[0])
        token = pd.to_datetime(start, unit="s", utc=True).strftime("%Y%m%d")
        target = deployment / f"week={token}"
        target.mkdir()
        connection.execute(
            f"COPY (SELECT * FROM canonical_flows WHERE ts >= {baseline_end} AND ts >= {start} "
            f"AND ts < {start + WEEK_SECONDS} ORDER BY ts, uid) TO {sql(str(target / 'part.parquet'))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        deployment_rows += count

    attacks, benign = connection.execute(
        "SELECT count(*) FILTER (WHERE is_attack), count(*) FILTER (WHERE NOT is_attack) "
        "FROM canonical_flows WHERE ts >= ?", [baseline_end]
    ).fetchone()
    manifest = {
        "source_files": len(files),
        "minimum_timestamp": minimum,
        "minimum_datetime": str(pd.to_datetime(minimum, unit="s", utc=True)),
        "baseline_end_timestamp": baseline_end,
        "baseline_end_datetime": str(pd.to_datetime(baseline_end, unit="s", utc=True)),
        "validation_fraction": float(splits["validation_fraction"]),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "deployment_rows": deployment_rows,
        "deployment_attacks": int(attacks),
        "deployment_benign": int(benign),
        "deployment_parts": len(deployment_weeks),
        "holdout_start": splits["holdout_start"],
    }
    write_yaml(manifest, output / "split_manifest.yaml")
    pd.DataFrame(
        [
            {
                "source": item.source,
                "path_within_raw_dir": str(
                    Path(item.source) / "parquet" / item.path.parent.name / item.path.name
                ),
                "rows": pq.ParquetFile(item.path).metadata.num_rows,
            }
            for item in files
        ]
    ).to_csv(reports / "source_inventory.csv", index=False)
    connection.close()
    (work / "prepare.duckdb").unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    return manifest


def _source_select(item: SourceFile) -> str:
    names = set(pq.ParquetFile(item.path).schema_arrow.names)

    def column(name: str, kind: str = "VARCHAR") -> str:
        source = next((candidate for candidate in ALIASES.get(name, (name,)) if candidate in names), None)
        return f'CAST("{source}" AS {kind}) AS "{name}"' if source else f'NULL::{kind} AS "{name}"'

    values = ["CAST(uid AS VARCHAR) AS uid", "CAST(ts AS DOUBLE) AS ts"]
    values.extend(column(name, "DOUBLE" if name in NUMERIC else "VARCHAR") for name in CANONICAL)
    values.extend([
        column("mitre_attack_tactics"),
        f"{sql(item.source)} AS source_dataset_raw",
        f"{item.priority} AS source_priority",
        f"TIMESTAMP {sql(item.week_start.tz_localize(None).isoformat())} AS week_start_raw",
    ])
    return (
        f"SELECT {', '.join(values)} FROM read_parquet({sql(str(item.path))}) "
        "WHERE uid IS NOT NULL AND try_cast(ts AS DOUBLE) IS NOT NULL"
    )


def _canonical_query() -> str:
    core = ", ".join(CANONICAL)
    tactic = "nullif(trim(mitre_attack_tactics), '')"
    return f"""
        SELECT uid, ts, {core},
          coalesce(string_agg(DISTINCT {tactic}, '|' ORDER BY {tactic})
            FILTER (WHERE lower({tactic}) <> 'none'), 'none') AS mitre_attack_tactics,
          bool_or(lower(coalesce(trim(mitre_attack_tactics), 'none')) <> 'none') AS is_attack,
          arg_min(source_dataset_raw, source_priority) AS source_dataset,
          arg_min(week_start_raw, source_priority) AS capture_week_start,
          count(*) AS source_rows
        FROM week_raw GROUP BY uid, ts, {core}
    """


def sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
