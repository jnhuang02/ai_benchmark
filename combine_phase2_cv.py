#!/usr/bin/env python3
"""Combine repeated-CV partitions and compute pre-registered summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file
from evaluation.phase2_metrics import (
    bootstrap_repeated_cv,
    group_appendix,
    partition_metrics,
    pooled_mape_appendix,
    selected_gnn_counts,
)


ROOT = Path(__file__).resolve().parent


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_frames = []
    input_files = []
    for input_dir in args.input:
        for path in sorted(input_dir.glob("predictions_partition_*.csv")):
            prediction_frames.append(pd.read_csv(path, dtype={"public_id": str}))
            input_files.append(path)
    if not prediction_frames:
        raise FileNotFoundError("No partition prediction files found")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_partitions = {
        f"cv5_partition_{index}" for index in range(5)
    }
    if set(predictions["protocol"]) != expected_partitions:
        raise AssertionError(
            f"Expected five partitions, found {sorted(predictions['protocol'].unique())}"
        )
    duplicates = predictions.duplicated(
        ["method", "protocol", "fold_id", "seed", "row_uid"]
    )
    if duplicates.any():
        raise AssertionError("Combined predictions contain duplicates")
    prediction_path = output / "predictions_by_partition_seed.csv"
    predictions.to_csv(prediction_path, index=False)

    summary, partition = partition_metrics(predictions)
    bootstrap, draws, pairs = bootstrap_repeated_cv(
        predictions, replicates=args.bootstrap_replicates
    )
    summary = summary.merge(bootstrap, on="method", how="left")
    summary.to_csv(output / "main_table.csv", index=False)
    partition.to_csv(output / "partition_metrics.csv", index=False)
    draws.to_csv(output / "bootstrap_by_method.csv", index=False)
    pairs.to_csv(output / "paired_differences.csv", index=False)
    group_appendix(predictions).to_csv(
        output / "per_group_results.csv", index=False
    )
    pooled_mape_appendix(predictions).to_csv(
        output / "pooled_mape_appendix.csv", index=False
    )
    selected_gnn_counts(predictions).to_csv(
        output / "selected_gnn_counts.csv", index=False
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "fold_assignments_sha256": sha256_file(
            ROOT / "folds" / "phase2_fold_assignments.csv"
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "input_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in input_files
        ],
        "output_sha256": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
        "phase2_loso_run": False,
        "phase2_sensitivity_run": False,
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
