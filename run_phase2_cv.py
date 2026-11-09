#!/usr/bin/env python3
"""Run the pre-registered repeated five-fold Phase 2 model ladder."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file
from evaluation.evaluate import GNNConfig
from evaluation.phase2 import (
    load_phase2_frame,
    fit_deterministic_fold,
    fit_neural_models,
)


ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(7, 17))


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in [
        "numpy",
        "pandas",
        "sklearn",
        "statsmodels",
        "torch",
        "torch_geometric",
        "scipy",
    ]:
        imported = __import__(module)
        versions[module] = str(getattr(imported, "__version__", "unknown"))
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = load_phase2_frame()
    assignments = pd.read_csv(ROOT / "folds" / "phase2_fold_assignments.csv")
    config = GNNConfig()
    started_all = time.perf_counter()
    runtimes = []
    for partition_index in args.partitions:
        protocol = f"cv5_partition_{partition_index}"
        fold_ids = sorted(
            assignments.loc[
                assignments["protocol"].eq(protocol), "fold_id"
            ].unique()
        )
        partition_frames = []
        started_partition = time.perf_counter()
        for fold_number, fold_id in enumerate(fold_ids, start=1):
            started_fold = time.perf_counter()
            deterministic = fit_deterministic_fold(
                frame,
                assignments,
                protocol=protocol,
                fold_id=fold_id,
            )
            fold_frames = [deterministic]
            for seed in SEEDS:
                fold_frames.append(
                    fit_neural_models(
                        frame,
                        assignments,
                        protocol=protocol,
                        fold_id=fold_id,
                        seed=seed,
                        config=config,
                    )
                )
            fold_predictions = pd.concat(
                [value for value in fold_frames if not value.empty],
                ignore_index=True,
            )
            checkpoint = (
                output / "checkpoints" / f"partition_{partition_index}_{fold_id}.csv"
            )
            checkpoint.parent.mkdir(exist_ok=True)
            fold_predictions.to_csv(checkpoint, index=False)
            partition_frames.append(fold_predictions)
            elapsed = time.perf_counter() - started_fold
            print(
                f"partition={partition_index} fold={fold_number}/5 "
                f"fold_id={fold_id} rows={len(fold_predictions)} "
                f"elapsed_s={elapsed:.3f}",
                flush=True,
            )
        partition_predictions = pd.concat(
            partition_frames, ignore_index=True
        )
        duplicate = partition_predictions.duplicated(
            ["method", "protocol", "fold_id", "seed", "row_uid"]
        )
        if duplicate.any():
            raise AssertionError("Phase 2 partition has duplicate predictions")
        partition_path = (
            output / f"predictions_partition_{partition_index}.csv"
        )
        partition_predictions.to_csv(partition_path, index=False)
        partition_elapsed = time.perf_counter() - started_partition
        runtimes.append(
            {
                "partition": partition_index,
                "protocol": protocol,
                "folds": len(fold_ids),
                "rows": len(partition_predictions),
                "methods": partition_predictions["method"].nunique(),
                "seeds": partition_predictions["seed"].nunique(),
                "runtime_seconds": partition_elapsed,
                "prediction_sha256": sha256_file(partition_path),
            }
        )
        pd.DataFrame(runtimes).to_csv(
            output / "partition_runtime.csv", index=False
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
        "partitions": args.partitions,
        "neural_seeds": list(SEEDS),
        "gnn_config": config.__dict__,
        "software_versions": software_versions(),
        "total_runtime_seconds": time.perf_counter() - started_all,
        "partition_runtime": runtimes,
        "phase2_loso_run": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
