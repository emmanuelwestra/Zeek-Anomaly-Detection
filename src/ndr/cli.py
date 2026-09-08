from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ndr.config import load_config
from ndr.data import prepare
from ndr.workflow import (
    benchmark,
    evaluate,
    operating_point,
    search_ae,
    search_fusion,
    search_ocsvm,
    train,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ndr",
        description="Train and evaluate static OC-SVM and mixed-AE network anomaly detectors.",
    )
    root.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build the leakage-controlled chronological corpus.")

    training = commands.add_parser("train", help="Fit models and benign-only thresholds.")
    training.add_argument("--model", choices=("ocsvm", "ae", "all"), default="all")

    searching = commands.add_parser("search", help="Run a focused development-only grid search.")
    searching.add_argument("--model", choices=("ocsvm", "ae", "fusion"), required=True)
    searching.add_argument("--grid", type=Path, help="Override the model's compact grid YAML.")

    evaluation = commands.add_parser("evaluate", help="Evaluate a frozen operating point.")
    evaluation.add_argument("--model", choices=("ocsvm", "ae", "fusion"), required=True)
    evaluation.add_argument("--period", choices=("development", "holdout"), required=True)

    benchmark_command = commands.add_parser("benchmark", help="Measure model-only throughput.")
    benchmark_command.add_argument("--training-rows", nargs="+", type=int)

    point = commands.add_parser(
        "operating-point",
        help="Retrospectively optimize a threshold under a precision or recall constraint.",
    )
    point.add_argument("--model", choices=("ocsvm", "ae", "fusion"), required=True)
    point.add_argument("--period", choices=("development", "holdout"), default="holdout")
    constraint = point.add_mutually_exclusive_group(required=True)
    constraint.add_argument("--min-precision", type=_probability)
    constraint.add_argument("--min-recall", type=_probability)
    return root


def run(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    if args.command == "prepare":
        return prepare(config)
    if args.command == "train":
        return train(config, args.model)
    if args.command == "evaluate":
        return evaluate(config, args.model, args.period)
    if args.command == "benchmark":
        return benchmark(config, training_rows=args.training_rows)
    if args.command == "operating-point":
        return operating_point(
            config, args.model, args.period, args.min_precision, args.min_recall
        )
    if args.command == "search":
        default = Path(f"configs/{args.model}-grid.yaml")
        grid = args.grid or default
        functions = {"ocsvm": search_ocsvm, "ae": search_ae, "fusion": search_fusion}
        return functions[args.model](config, grid)
    raise RuntimeError(f"Unhandled command: {args.command}")


def main() -> None:
    result = run(parser().parse_args())
    print(yaml.safe_dump(result, sort_keys=False).rstrip())


def _probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return number


if __name__ == "__main__":
    main()
