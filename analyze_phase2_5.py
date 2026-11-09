#!/usr/bin/env python3
"""Generate Phase 2.5 tail and missing-LOSO-detail artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file
from evaluation.evaluate import _frame_for_systems, fold_role_sets
from evaluation.features import StrictTargetScaler
from evaluation.phase2 import (
    fit_hierarchical_mixed_detailed,
    load_phase2_frame,
)


ROOT = Path(__file__).resolve().parent
CV = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
LOSO = ROOT / "results" / "phase2-loso-final-a689ba5"
METHODS = ("random_forest", "mlp", "graphsage")
QUANTILES = (0.5, 0.75, 0.9, 0.95)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "method",
        "protocol",
        "fold_id",
        "row_uid",
        "system_id",
        "result",
        "direction",
    ]
    aggregated = predictions.groupby(identifiers, as_index=False).agg(
        signed_log_prediction=("signed_log_prediction", "mean"),
        seed_count=("seed", "nunique"),
    )
    aggregated["absolute_log_ratio"] = np.abs(
        aggregated["signed_log_prediction"]
        - aggregated["direction"] * np.log(aggregated["result"])
    )
    return aggregated


def tail_metrics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=float)
    result = {
        "median": float(np.quantile(values, 0.5)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "within_25pct": float(np.mean(values <= np.log(1.25))),
        "within_50pct": float(np.mean(values <= np.log(1.50))),
        "support_rows": len(values),
    }
    return result


def cv_tail_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregated = aggregate_seed_predictions(
        predictions.loc[predictions["method"].isin(METHODS)].copy()
    )
    partition_rows = []
    for (method, protocol), local in aggregated.groupby(
        ["method", "protocol"], sort=True
    ):
        partition_rows.append(
            {
                "method": method,
                "protocol": protocol,
                **tail_metrics(local["absolute_log_ratio"].to_numpy()),
            }
        )
    partition_table = pd.DataFrame(partition_rows)
    rows = []
    for method, local in partition_table.groupby("method", sort=True):
        row = {
            "protocol": "cv5_partitioned",
            "method": method,
            "aggregation": (
                "seed mean per row; tail metric per partition; "
                "mean across five partitions"
            ),
            "partitions": len(local),
        }
        for metric in [
            "median",
            "p75",
            "p90",
            "p95",
            "max",
            "within_25pct",
            "within_50pct",
            "support_rows",
        ]:
            row[metric] = float(local[metric].mean())
            row[f"{metric}_partition_sd"] = float(local[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def loso_tail_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregated = aggregate_seed_predictions(
        predictions.loc[predictions["method"].isin(METHODS)].copy()
    )
    rows = []
    for method, local in aggregated.groupby("method", sort=True):
        rows.append(
            {
                "protocol": "loso",
                "method": method,
                "aggregation": (
                    "seed mean per row; tail metric on pooled out-of-fold rows"
                ),
                "partitions": 1,
                **tail_metrics(local["absolute_log_ratio"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def empty_loso_folds() -> pd.DataFrame:
    assignments = pd.read_csv(ROOT / "folds" / "phase2_fold_assignments.csv")
    loso = assignments.loc[assignments["protocol"].eq("loso")]
    rows = []
    for manifest_path in sorted(
        (ROOT / "results").glob("phase2-loso-shard*/run_manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text())
        for fold_id in manifest["empty_folds"]:
            system_ids = sorted(
                loso.loc[
                    loso["fold_id"].eq(fold_id) & loso["role"].eq("test"),
                    "system_id",
                ].tolist()
            )
            if len(system_ids) != 1:
                raise AssertionError(f"{fold_id} does not identify one system")
            rows.append(
                {
                    "protocol": "loso",
                    "fold_id": fold_id,
                    "system_id": system_ids[0],
                    "supported_rows": 0,
                    "reason": "no held-out row had a supported target group",
                    "shard_manifest": str(manifest_path.relative_to(ROOT)),
                    "shard_manifest_sha256": sha256_file(manifest_path),
                }
            )
    output = pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
    if len(output) != 6:
        raise AssertionError(f"Expected six empty LOSO folds, found {len(output)}")
    return output


def replay_powell_fold() -> dict[str, object]:
    assignments = pd.read_csv(ROOT / "folds" / "phase2_fold_assignments.csv")
    fold_id = "holdout_089"
    frame = load_phase2_frame()
    train_systems, validation_systems, test_systems = fold_role_sets(
        assignments, "loso", fold_id
    )
    if validation_systems:
        # Validation does not enter the mixed-model variance estimate.
        pass
    train = _frame_for_systems(frame, set(train_systems))
    test_all = _frame_for_systems(frame, set(test_systems))
    scaler = StrictTargetScaler.fit(train)
    supported = set(scaler.supported_groups)
    train_labeled = train.loc[train["target_group"].isin(supported)].copy()
    test = test_all.loc[test_all["target_group"].isin(supported)].copy()
    prediction, diagnostics = fit_hierarchical_mixed_detailed(
        train_labeled, test, scaler
    )
    return {
        "protocol": "loso",
        "fold_id": fold_id,
        "system_id": sorted(test_systems)[0],
        "supported_rows": len(test),
        "converged": diagnostics["converged"],
        "optimizer": diagnostics["optimizer"],
        "formula": diagnostics["formula"],
        "prediction_finite": bool(np.isfinite(prediction).all()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv_predictions = pd.read_csv(CV / "predictions_by_partition_seed.csv")
    loso_predictions = pd.read_csv(LOSO / "predictions_by_fold_seed.csv")
    tails = pd.concat(
        [
            cv_tail_summary(cv_predictions),
            loso_tail_summary(loso_predictions),
        ],
        ignore_index=True,
    )
    tails.to_csv(output / "tail_summary.csv", index=False)
    empty = empty_loso_folds()
    empty.to_csv(output / "loso_empty_folds.csv", index=False)
    powell = pd.DataFrame([replay_powell_fold()])
    if powell["optimizer"].iloc[0] != "powell" or not bool(
        powell["prediction_finite"].iloc[0]
    ):
        raise AssertionError("Expected the logged Powell fallback to reproduce")
    powell.to_csv(output / "loso_powell_fallback.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "cv5_predictions_sha256": sha256_file(
            CV / "predictions_by_partition_seed.csv"
        ),
        "loso_predictions_sha256": sha256_file(
            LOSO / "predictions_by_fold_seed.csv"
        ),
        "tail_methods": list(METHODS),
        "tail_percentiles": list(QUANTILES),
        "empty_loso_folds": empty["fold_id"].tolist(),
        "powell_fallback_fold": powell["fold_id"].iloc[0],
        "powell_fallback_system": powell["system_id"].iloc[0],
        "output_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
