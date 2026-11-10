#!/usr/bin/env python3
"""Run the approved Phase 2.5 leave-one-family-out stress test."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import feature_set_manifest
from evaluation.data import DEFAULT_DATA, sha256_file
from evaluation.evaluate import GNNConfig
from evaluation.phase2 import (
    fit_deterministic_fold,
    fit_neural_models,
    load_phase2_frame,
)


ROOT = Path(__file__).resolve().parent
FOLD_ASSIGNMENTS = ROOT / "folds" / "loafo_fold_assignments.csv"
FAMILY_MAP = ROOT / "folds" / "loafo_family_mapping_proposed.csv"
SYSTEM_ASSIGNMENT = ROOT / "folds" / "loafo_system_assignment_proposed.csv"
SEEDS = (7, 11, 16)
REPORTED_METHODS = (
    "group_median",
    "ridge",
    "random_forest",
    "mlp",
    "graphsage",
    "gcn",
    "gat",
    "best_gnn",
)
PAIR_METHODS = ("graphsage", "gcn", "gat", "best_gnn")


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
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def load_loafo_frame() -> pd.DataFrame:
    frame = load_phase2_frame()
    mapping = pd.read_csv(
        SYSTEM_ASSIGNMENT, usecols=["system_id", "proposed_family_id"]
    )
    frame = frame.drop(columns=["accelerator_family"]).merge(
        mapping,
        on="system_id",
        how="left",
        validate="many_to_one",
    )
    if frame["proposed_family_id"].isna().any():
        raise AssertionError("LOAFO frame contains an unmapped system")
    return frame.rename(columns={"proposed_family_id": "accelerator_family"})


def aggregate_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "method",
        "protocol",
        "fold_id",
        "row_uid",
        "system_id",
        "target_group",
        "result",
        "direction",
    ]
    aggregated = (
        predictions.groupby(identifiers, as_index=False)
        .agg(
            prediction_log=("prediction_log", "mean"),
            signed_log_prediction=("signed_log_prediction", "mean"),
            y=("y", "mean"),
            seeds=("seed", "nunique"),
        )
        .copy()
    )
    aggregated["prediction"] = np.exp(
        aggregated["direction"] * aggregated["signed_log_prediction"]
    )
    return aggregated


def primary_error(frame: pd.DataFrame) -> float:
    return float(
        np.median(
            np.abs(
                np.log(frame["prediction"].to_numpy(dtype=float))
                - np.log(frame["result"].to_numpy(dtype=float))
            )
        )
    )


def _draw_systems(
    rng: np.random.Generator,
    system_values: np.ndarray,
) -> np.ndarray:
    return rng.choice(system_values, size=len(system_values), replace=True)


def bootstrap_family(
    aggregated: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = np.array(sorted(aggregated["system_id"].unique()), dtype=object)
    method_positions = {}
    method_errors = {}
    for method, local in aggregated.groupby("method", sort=True):
        local = local.reset_index(drop=True)
        positions = {
            system_id: local.index[local["system_id"].eq(system_id)].to_numpy()
            for system_id in systems
        }
        method_positions[method] = positions
        method_errors[method] = np.abs(
            np.log(local["prediction"].to_numpy(dtype=float))
            - np.log(local["result"].to_numpy(dtype=float))
        )
    expected_systems = set(systems)
    for method, positions in method_positions.items():
        if set(positions) != expected_systems:
            raise AssertionError(f"System sets differ for {method}")

    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = _draw_systems(rng, systems)
        selected = {}
        for method in method_positions:
            indices = np.concatenate(
                [method_positions[method][system_id] for system_id in sampled]
            )
            selected[method] = float(np.median(method_errors[method][indices]))
        rows.append(
            {
                "bootstrap_replicate": replicate,
                **selected,
            }
        )
    draws = pd.DataFrame(rows)
    summaries = []
    for method in method_positions:
        values = draws[method].to_numpy(dtype=float)
        summaries.append(
            {
                "method": method,
                "primary_error_95ci_low": float(np.quantile(values, 0.025)),
                "primary_error_95ci_high": float(np.quantile(values, 0.975)),
            }
        )
    pairs = []
    for left in PAIR_METHODS:
        difference = draws[left] - draws["mlp"]
        pairs.append(
            {
                "left": left,
                "right": "mlp",
                "bootstrap_mean_difference": float(difference.mean()),
                "difference_95ci_low": float(difference.quantile(0.025)),
                "difference_95ci_high": float(difference.quantile(0.975)),
                "probability_left_better": float((difference < 0).mean()),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(pairs)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assignments = pd.read_csv(FOLD_ASSIGNMENTS)
    families = pd.read_csv(FAMILY_MAP)
    frame = load_loafo_frame()
    config = GNNConfig()
    started_all = time.perf_counter()
    all_predictions = []
    runtime_rows = []
    runtimes = []

    expected_folds = sorted(assignments["fold_id"].unique())
    for fold_number, fold_id in enumerate(expected_folds, start=1):
        started = time.perf_counter()
        deterministic = fit_deterministic_fold(
            frame,
            assignments,
            protocol="loafo",
            fold_id=fold_id,
            methods=("group_median", "ridge", "random_forest"),
        )
        frames = [deterministic]
        seed_frames = []
        for seed in SEEDS:
            neural = fit_neural_models(
                frame,
                assignments,
                protocol="loafo",
                fold_id=fold_id,
                seed=seed,
                config=config,
                include_secondary_graph_variants=False,
            )
            seed_frames.append(neural)
        neural = pd.concat(seed_frames, ignore_index=True)
        neural = neural.loc[
            neural["method"].isin({"mlp", "graphsage", "gcn", "gat", "best_gnn"})
        ]
        frames.append(neural)
        fold_predictions = pd.concat(frames, ignore_index=True)
        if fold_predictions.empty:
            raise AssertionError(f"{fold_id} produced no supported predictions")
        row_counts = fold_predictions.groupby("method")["row_uid"].nunique()
        if row_counts.nunique() != 1:
            raise AssertionError(f"{fold_id} methods have unequal supported rows")
        if set(fold_predictions["method"]) != set(REPORTED_METHODS):
            raise AssertionError(f"{fold_id} has an unexpected method set")
        all_predictions.append(fold_predictions)
        elapsed = time.perf_counter() - started
        runtime_rows.append(
            {
                "fold_id": fold_id,
                "fold_number": fold_number,
                "systems": fold_predictions["system_id"].nunique(),
                "supported_rows": int(row_counts.iloc[0]),
                "prediction_rows": len(fold_predictions),
                "runtime_seconds": elapsed,
            }
        )
        print(
            f"LOAFO fold={fold_number}/{len(expected_folds)} fold_id={fold_id} "
            f"supported_rows={int(row_counts.iloc[0])} elapsed_s={elapsed:.3f}",
            flush=True,
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / "predictions_by_fold_seed.csv", index=False)
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(output / "fold_runtime.csv", index=False)
    aggregated = aggregate_seed_predictions(predictions)

    metric_rows = []
    pair_rows = []
    draw_frames = []
    for fold_number, (fold_id, local) in enumerate(
        aggregated.groupby("fold_id", sort=True)
    ):
        family = families.set_index("family_id").loc[
            assignments.loc[
                assignments["fold_id"].eq(fold_id), "family_id"
            ].iloc[0]
        ]
        point_errors = {
            method: primary_error(method_frame)
            for method, method_frame in local.groupby("method", sort=True)
        }
        intervals, pairs = bootstrap_family(
            local,
            replicates=args.bootstrap_replicates,
            seed=20261107 + fold_number,
        )
        intervals = intervals.set_index("method")
        for method in REPORTED_METHODS:
            metric_rows.append(
                {
                    "fold_id": fold_id,
                    "family_id": family.name,
                    "device_class": family["device_class"],
                    "holdout_disposition": family["holdout_disposition"],
                    "systems": int(family["systems"]),
                    "rows": int(family["rows"]),
                    "supported_rows": int(
                        local.loc[
                            local["method"].eq(method), "row_uid"
                        ].nunique()
                    ),
                    "method": method,
                    "primary_error": point_errors[method],
                    "primary_error_95ci_low": intervals.loc[
                        method, "primary_error_95ci_low"
                    ],
                    "primary_error_95ci_high": intervals.loc[
                        method, "primary_error_95ci_high"
                    ],
                }
            )
        for row in pairs.to_dict(orient="records"):
            paired = pairs.loc[
                pairs["left"].eq(row["left"]) & pairs["right"].eq("mlp")
            ].iloc[0]
            pair_rows.append(
                {
                    "fold_id": fold_id,
                    "family_id": family.name,
                    "device_class": family["device_class"],
                    "holdout_disposition": family["holdout_disposition"],
                    "systems": int(family["systems"]),
                    "supported_rows": int(
                        local.loc[
                            local["method"].eq(row["left"]), "row_uid"
                        ].nunique()
                    ),
                    "left": row["left"],
                    "right": "mlp",
                    "point_difference": point_errors[row["left"]]
                    - point_errors["mlp"],
                    "bootstrap_mean_difference": paired[
                        "bootstrap_mean_difference"
                    ],
                    "difference_95ci_low": paired["difference_95ci_low"],
                    "difference_95ci_high": paired["difference_95ci_high"],
                    "probability_left_better": paired[
                        "probability_left_better"
                    ],
                    "bootstrap_replicates": args.bootstrap_replicates,
                }
            )
        draws = []
        # Recompute the draw table only for the compact saved audit artifact.
        system_values = np.array(
            sorted(local["system_id"].unique()), dtype=object
        )
        method_errors = {
            method: np.abs(
                np.log(method_frame["prediction"].to_numpy(dtype=float))
                - np.log(method_frame["result"].to_numpy(dtype=float))
            )
            for method, method_frame in local.groupby("method")
        }
        method_positions = {
            method: {
                system_id: method_frame.reset_index(drop=True).index[
                    method_frame.reset_index(drop=True)["system_id"].eq(system_id)
                ]
                for system_id in system_values
            }
            for method, method_frame in local.groupby("method")
        }
        rng = np.random.default_rng(20261107 + fold_number)
        for replicate in range(args.bootstrap_replicates):
            sampled = _draw_systems(rng, system_values)
            draws.append(
                {
                    "fold_id": fold_id,
                    "family_id": family.name,
                    "bootstrap_replicate": replicate,
                    **{
                        method: float(
                            np.median(
                                method_errors[method][
                                    np.concatenate(
                                        [
                                            method_positions[method][system_id]
                                            for system_id in sampled
                                        ]
                                    )
                                ]
                            )
                        )
                        for method in REPORTED_METHODS
                    },
                }
            )
        draw_frames.append(pd.DataFrame(draws))

    pd.DataFrame(metric_rows).to_csv(
        output / "main_table_by_family.csv", index=False
    )
    pd.DataFrame(pair_rows).to_csv(
        output / "paired_differences_by_family.csv", index=False
    )
    pd.concat(draw_frames, ignore_index=True).to_csv(
        output / "bootstrap_by_family.csv", index=False
    )
    selected = predictions.loc[
        predictions["method"].eq("best_gnn")
    ].groupby(["fold_id", "seed"], as_index=False).agg(
        selected_gnn=("selected_gnn", "first")
    )
    selected_counts = (
        selected.groupby(["fold_id", "selected_gnn"], as_index=False)
        .size()
        .rename(columns={"size": "seed_count"})
    )
    selected_counts.to_csv(output / "selected_gnn_by_family.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        **feature_set_manifest(),
        "protocol": "loafo",
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "family_mapping_sha256": sha256_file(FAMILY_MAP),
        "system_assignment_sha256": sha256_file(SYSTEM_ASSIGNMENT),
        "fold_assignments_sha256": sha256_file(FOLD_ASSIGNMENTS),
        "neural_seeds": list(SEEDS),
        "reported_methods": list(REPORTED_METHODS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "mixed_always_train": True,
        "gnn_config": config.__dict__,
        "software_versions": software_versions(),
        "folds": len(expected_folds),
        "systems": predictions["system_id"].nunique(),
        "supported_rows_by_fold": {
            row.fold_id: int(row.supported_rows)
            for row in runtime.itertuples(index=False)
        },
        "total_runtime_seconds": time.perf_counter() - started_all,
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
