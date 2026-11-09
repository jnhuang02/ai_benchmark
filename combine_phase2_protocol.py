#!/usr/bin/env python3
"""Combine a sharded LOSO or sensitivity run and compute paired results."""

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
    pooled_mape_appendix,
    protocol_summary,
    selected_gnn_counts,
)


ROOT = Path(__file__).resolve().parent


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["loso", "sensitivity"], required=True)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    input_paths = []
    input_manifests = []
    for directory in args.input:
        path = directory / "predictions.csv"
        frames.append(pd.read_csv(path, dtype={"public_id": str}))
        input_paths.append(path)
        input_manifests.append(
            json.loads((directory / "run_manifest.json").read_text())
        )
    predictions = pd.concat(frames, ignore_index=True)
    duplicate = predictions.duplicated(
        ["method", "protocol", "fold_id", "seed", "row_uid"]
    )
    if duplicate.any():
        raise AssertionError("Combined protocol has duplicate predictions")
    assignments = pd.read_csv(ROOT / "folds" / "phase2_fold_assignments.csv")
    expected_folds = set(
        assignments.loc[
            assignments["protocol"].eq(args.protocol), "fold_id"
        ]
    )
    manifest_folds = set().union(
        *(set(manifest["fold_ids"]) for manifest in input_manifests)
    )
    if manifest_folds != expected_folds:
        missing = sorted(expected_folds - manifest_folds)
        raise AssertionError(f"Combined protocol is missing folds: {missing}")
    path = output / "predictions_by_fold_seed.csv"
    predictions.to_csv(path, index=False)
    summary = protocol_summary(predictions)
    bootstrap, draws, pairs = bootstrap_repeated_cv(
        predictions, replicates=args.bootstrap_replicates
    )
    summary = summary.merge(bootstrap, on="method", how="left")
    summary.to_csv(output / "main_table.csv", index=False)
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
        "protocol": args.protocol,
        "bootstrap_replicates": args.bootstrap_replicates,
        "data_sha256": sha256_file(DEFAULT_DATA),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "fold_assignments_sha256": sha256_file(
            ROOT / "folds" / "phase2_fold_assignments.csv"
        ),
        "input_files": [
            {"path": str(value), "sha256": sha256_file(value)}
            for value in input_paths
        ],
        "output_sha256": {
            str(file.relative_to(output)): sha256_file(file)
            for file in sorted(output.iterdir())
            if file.is_file()
        },
        "loafo_run": False,
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
