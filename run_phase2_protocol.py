#!/usr/bin/env python3
"""Run Phase 2 LOSO or sensitivity in deterministic fold shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file
from evaluation.evaluate import GNNConfig
from evaluation.phase2 import (
    fit_deterministic_fold,
    fit_neural_models,
    load_phase2_frame,
)


ROOT = Path(__file__).resolve().parent


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["loso", "sensitivity"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard coordinates")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = load_phase2_frame()
    assignments = pd.read_csv(ROOT / "folds" / "phase2_fold_assignments.csv")
    all_folds = sorted(
        assignments.loc[
            assignments["protocol"].eq(args.protocol), "fold_id"
        ].unique()
    )
    folds = [
        fold
        for index, fold in enumerate(all_folds)
        if index % args.shard_count == args.shard_index
    ]
    config = GNNConfig()
    fold_frames = []
    started = time.perf_counter()
    fold_runtimes = []
    for number, fold_id in enumerate(folds, 1):
        fold_started = time.perf_counter()
        deterministic = fit_deterministic_fold(
            frame,
            assignments,
            protocol=args.protocol,
            fold_id=fold_id,
        )
        frames = [deterministic]
        for seed in args.seeds:
            frames.append(
                fit_neural_models(
                    frame,
                    assignments,
                    protocol=args.protocol,
                    fold_id=fold_id,
                    seed=seed,
                    config=config,
                )
            )
        nonempty_frames = [value for value in frames if not value.empty]
        fold_predictions = (
            pd.concat(nonempty_frames, ignore_index=True)
            if nonempty_frames
            else pd.DataFrame()
        )
        if not fold_predictions.empty:
            fold_frames.append(fold_predictions)
        fold_runtimes.append(
            {
                "fold_id": fold_id,
                "rows": len(fold_predictions),
                "runtime_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"protocol={args.protocol} shard={args.shard_index} "
            f"fold={number}/{len(folds)} fold_id={fold_id} "
            f"rows={len(fold_predictions)} "
            f"elapsed_s={time.perf_counter() - fold_started:.3f}",
            flush=True,
        )
    predictions = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames
        else pd.DataFrame()
    )
    duplicate = (
        predictions.duplicated(
            ["method", "protocol", "fold_id", "seed", "row_uid"]
        )
        if not predictions.empty
        else pd.Series(dtype=bool)
    )
    if duplicate.any():
        raise AssertionError("Protocol shard produced duplicate predictions")
    prediction_path = output / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(fold_runtimes).to_csv(
        output / "fold_runtime.csv", index=False
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "protocol": args.protocol,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "fold_count": len(folds),
        "empty_folds": sorted(set(folds) - set(predictions.get("fold_id", []))),
        "fold_ids": folds,
        "seeds": args.seeds,
        "methods": sorted(predictions["method"].unique().tolist()),
        "rows": len(predictions),
        "runtime_seconds": time.perf_counter() - started,
        "data_sha256": sha256_file(DEFAULT_DATA),
        "fold_assignments_sha256": sha256_file(
            ROOT / "folds" / "phase2_fold_assignments.csv"
        ),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "prediction_sha256": sha256_file(prediction_path),
        "loafo_run": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
