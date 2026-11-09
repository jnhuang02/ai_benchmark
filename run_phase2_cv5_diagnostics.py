#!/usr/bin/env python3
"""Generate the requested non-modifying diagnostics for the accepted CV5 ladder."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import DEFAULT_DATA, load_family_map, sha256_file
from evaluation.evaluate import GNNConfig, fold_role_sets
from evaluation.features import (
    NUMERIC_COLUMNS,
    PHASE2_NUMERIC_COLUMNS,
    StrictTargetScaler,
    normalized_category,
)
from evaluation.metrics import group_metrics
from evaluation.phase2 import (
    _frame_for_systems,
    fit_hierarchical_mixed_detailed,
    fit_phase2_graph_model,
    load_phase2_frame,
)
from evaluation.phase2_metrics import aggregate_predictions


ROOT = Path(__file__).resolve().parent
CV_BUNDLE = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
PRE_RUN_BUNDLE = ROOT / "results" / "phase2-prerun-1122699"
PHASE15_BUNDLE = ROOT / "results" / "phase1_5-final-8fbe960"
SEEDS = tuple(range(7, 17))


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def primary_from_columns(frame: pd.DataFrame) -> float:
    return float(
        np.median(
            np.abs(
                np.log(frame["prediction"].to_numpy(dtype=float))
                - np.log(frame["result"].to_numpy(dtype=float))
            )
        )
    )


def gat_diagnostics(
    frame: pd.DataFrame, assignments: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    saved = pd.read_csv(CV_BUNDLE / "predictions_by_partition_seed.csv")
    saved_gat = saved.loc[saved["method"].eq("gat")]
    comparisons = []
    for partition_index in range(5):
        protocol = f"cv5_partition_{partition_index}"
        fold_ids = sorted(
            assignments.loc[
                assignments["protocol"].eq(protocol), "fold_id"
            ].unique()
        )
        for fold_id in fold_ids:
            for seed in SEEDS:
                prediction, diagnostics = fit_phase2_graph_model(
                    frame,
                    assignments,
                    protocol=protocol,
                    fold_id=fold_id,
                    model_name="gat",
                    seed=seed,
                    config=GNNConfig(),
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "fold_id": fold_id,
                        "seed": seed,
                        "rows": len(prediction),
                        **diagnostics,
                    }
                )
                old = saved_gat.loc[
                    saved_gat["protocol"].eq(protocol)
                    & saved_gat["fold_id"].eq(fold_id)
                    & saved_gat["seed"].eq(seed)
                ]
                merged = old[["row_uid", "prediction"]].merge(
                    prediction[["row_uid", "prediction"]],
                    on="row_uid",
                    suffixes=("_saved", "_rerun"),
                    validate="one_to_one",
                )
                comparisons.append(
                    {
                        "protocol": protocol,
                        "fold_id": fold_id,
                        "seed": seed,
                        "max_abs_prediction_difference": float(
                            np.max(
                                np.abs(
                                    merged["prediction_rerun"]
                                    - merged["prediction_saved"]
                                )
                            )
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "gat_training_by_fold_seed.csv", index=False)
    summary = (
        detail.groupby(["protocol", "fold_id"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            rows_min=("rows", "min"),
            rows_max=("rows", "max"),
            prediction_variance_mean=("prediction_variance", "mean"),
            prediction_variance_min=("prediction_variance", "min"),
            prediction_variance_max=("prediction_variance", "max"),
            normalized_prediction_variance_mean=(
                "normalized_prediction_variance_mean",
                "mean",
            ),
            prediction_unique_mean=("prediction_unique", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_min=("best_epoch", "min"),
            best_epoch_max=("best_epoch", "max"),
            epochs_run_mean=("epochs_run", "mean"),
            epochs_run_min=("epochs_run", "min"),
            epochs_run_max=("epochs_run", "max"),
            final_train_loss_mean=("final_train_loss", "mean"),
            final_validation_loss_mean=("final_validation_loss", "mean"),
        )
        .sort_values(["protocol", "fold_id"])
    )
    summary.to_csv(output / "gat_training_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        output / "gat_saved_prediction_comparison.csv", index=False
    )

    phase1_prediction, phase1_diagnostics = fit_phase2_graph_model(
        frame,
        assignments,
        protocol="cv5_partition_0",
        fold_id="p0_fold_0",
        model_name="gat",
        seed=7,
        config=GNNConfig(),
        numeric_columns=tuple(NUMERIC_COLUMNS),
    )
    phase2_prediction = saved_gat.loc[
        saved_gat["protocol"].eq("cv5_partition_0")
        & saved_gat["fold_id"].eq("p0_fold_0")
        & saved_gat["seed"].eq(7)
    ]
    comparison = pd.DataFrame(
        [
            {
                "run": "Phase 2 GAT, p0_fold_0 seed 7",
                **detail.loc[
                    detail["protocol"].eq("cv5_partition_0")
                    & detail["fold_id"].eq("p0_fold_0")
                    & detail["seed"].eq(7)
                ].iloc[0][
                    [
                        "prediction_variance",
                        "normalized_prediction_variance_mean",
                        "prediction_unique",
                        "best_epoch",
                        "epochs_run",
                        "final_train_loss",
                        "final_validation_loss",
                    ]
                ].to_dict(),
                "primary_error": primary_from_columns(phase2_prediction),
            },
            {
                "run": "Phase 1 feature path, four-head GAT, same fold/seed",
                **phase1_diagnostics,
                "primary_error": primary_from_columns(phase1_prediction),
            },
        ]
    )
    comparison.to_csv(output / "gat_phase1_comparison.csv", index=False)
    return summary, comparison


def graphsage_regression(
    frame: pd.DataFrame, assignments: pd.DataFrame, output: Path
) -> pd.DataFrame:
    feature_variants = {
        "phase2_full_with_cpu_indicator": tuple(PHASE2_NUMERIC_COLUMNS),
        "phase1_numeric_only": tuple(NUMERIC_COLUMNS),
    }
    all_seed_rows = []
    for feature_set, numeric_columns in feature_variants.items():
        for seed in SEEDS:
            fold_predictions = [
                fit_phase2_graph_model(
                    frame,
                    assignments,
                    protocol="cv5_partition_0",
                    fold_id=fold_id,
                    model_name="graphsage",
                    seed=seed,
                    config=GNNConfig(),
                    numeric_columns=numeric_columns,
                )[0]
                for fold_id in sorted(
                    assignments.loc[
                        assignments["protocol"].eq("cv5_partition_0"),
                        "fold_id",
                    ].unique()
                )
            ]
            merged_folds = pd.concat(
                fold_predictions,
                ignore_index=True,
            )
            all_seed_rows.append(
                {
                    "feature_set": feature_set,
                    "seed": seed,
                    "rows": len(merged_folds),
                    "primary_error": primary_from_columns(merged_folds),
                }
            )
    seed_table = pd.DataFrame(all_seed_rows)
    seed_table.to_csv(output / "graphsage_regression_by_seed.csv", index=False)
    saved = pd.read_csv(CV_BUNDLE / "predictions_by_partition_seed.csv")
    saved = saved.loc[
        saved["method"].eq("graphsage")
        & saved["protocol"].eq("cv5_partition_0")
    ]
    saved_seed_rows = []
    for seed, local in saved.groupby("seed"):
        saved_seed_rows.append(
            {
                "seed": seed,
                "rows": len(local),
                "primary_error": primary_from_columns(local),
            }
        )
    saved_seed_table = pd.DataFrame(saved_seed_rows)
    ensemble = aggregate_predictions(saved)
    comparison = pd.DataFrame(
        [
            {
                "quantity": "Phase 2 GraphSAGE full-feature seed mean",
                "value": saved_seed_table["primary_error"].mean(),
            },
            {
                "quantity": "Phase 2 GraphSAGE full-feature seed SD",
                "value": saved_seed_table["primary_error"].std(ddof=1),
            },
            {
                "quantity": "Phase 2 GraphSAGE full-feature seed-ensemble primary",
                "value": primary_from_columns(ensemble),
            },
            {
                "quantity": "Phase 2 code, Phase 1 numeric-feature seed mean",
                "value": seed_table.loc[
                    seed_table["feature_set"].eq("phase1_numeric_only"),
                    "primary_error",
                ].mean(),
            },
            {
                "quantity": "Phase 2 code, Phase 1 numeric-feature seed SD",
                "value": seed_table.loc[
                    seed_table["feature_set"].eq("phase1_numeric_only"),
                    "primary_error",
                ].std(ddof=1),
            },
            {
                "quantity": "Phase 1.5 fixed-isolation seed mean",
                "value": 0.46231398720767014,
            },
            {
                "quantity": "Phase 2 minus Phase 1.5 seed-mean difference",
                "value": (
                    saved_seed_table["primary_error"].mean()
                    - 0.46231398720767014
                ),
            },
        ]
    )
    comparison.to_csv(output / "graphsage_regression_comparison.csv", index=False)
    return comparison


def spearman_diagnostics(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    aggregated = aggregate_predictions(predictions)
    rows = []
    for (method, protocol), local in aggregated.groupby(
        ["method", "protocol"], sort=True
    ):
        scores = []
        for _, group in local.groupby("target_group", sort=True):
            if (
                len(group) >= 3
                and group["result"].nunique() >= 2
                and group["prediction"].nunique() >= 2
            ):
                score = group["result"].rank().corr(group["prediction"].rank())
                if np.isfinite(score):
                    scores.append(float(score))
        value = float(np.median(scores)) if scores else float("nan")
        if method == "group_median":
            value = float("nan")
        rows.append(
            {
                "method": method,
                "protocol": protocol,
                "qualifying_groups": len(scores),
                "median_group_spearman": value,
                "group_median_forced_na": method == "group_median",
                "all_group_predictions_constant": bool(
                    local.groupby("target_group")["prediction"]
                    .nunique()
                    .eq(1)
                    .all()
                ),
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "spearman_group_counts.csv", index=False)
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            partitions=("protocol", "nunique"),
            qualifying_groups_min=("qualifying_groups", "min"),
            qualifying_groups_max=("qualifying_groups", "max"),
            median_group_spearman_partition_mean=(
                "median_group_spearman",
                "mean",
            ),
            group_median_forced_na=("group_median_forced_na", "first"),
        )
        .sort_values("method")
    )
    summary.to_csv(output / "spearman_group_summary.csv", index=False)
    corrected = pd.read_csv(CV_BUNDLE / "main_table.csv")
    mask = corrected["method"].eq("group_median")
    for column in [
        "median_group_spearman_mean",
        "median_group_spearman_95ci_low",
        "median_group_spearman_95ci_high",
    ]:
        corrected.loc[mask, column] = np.nan
    corrected.to_csv(output / "main_table_corrected.csv", index=False)
    return detail


def residual_diagnostics(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    aggregated = aggregate_predictions(predictions)
    aggregated["z_residual"] = (
        aggregated["prediction_log"] - aggregated["y"]
    )
    aggregated["abs_z"] = aggregated["z_residual"].abs()
    rows = []
    for method in ["random_forest", "mlp", "graphsage"]:
        local = aggregated.loc[aggregated["method"].eq(method)].sort_values(
            "abs_z", ascending=False
        )
        for rank, row in enumerate(local.head(10).itertuples(index=False), 1):
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "protocol": row.protocol,
                    "fold_id": row.fold_id,
                    "row_uid": row.row_uid,
                    "target_group": row.target_group,
                    "system_id": row.system_id,
                    "result": row.result,
                    "prediction": row.prediction,
                    "z_residual": row.z_residual,
                    "abs_z": row.abs_z,
                }
            )
        sse = float((local["z_residual"] ** 2).sum())
        top10 = float((local.head(10)["z_residual"] ** 2).sum())
        rows.append(
            {
                "method": method,
                "rank": "top10_sse_share",
                "z_residual": top10 / sse if sse else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "top_abs_z_residuals.csv", index=False)
    return result


def partition_diagnostics(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    aggregated = aggregate_predictions(predictions)
    methods = ["random_forest", "mlp", "graphsage", "gcn", "gat"]
    partition_rows = []
    system_rows = []
    for method in methods:
        local = aggregated.loc[aggregated["method"].eq(method)].copy()
        local["absolute_log_error"] = np.abs(
            local["signed_log_prediction"] - local["signed_log_result"]
        )
        for protocol, partition in local.groupby("protocol", sort=True):
            partition_rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "primary_error": float(
                        np.median(partition["absolute_log_error"])
                    ),
                    "rows": len(partition),
                }
            )
        system_partition = (
            local.groupby(["system_id", "protocol"], as_index=False)
            .agg(error=("absolute_log_error", "median"))
            .pivot(
                index="system_id", columns="protocol", values="error"
            )
        )
        system_partition["range"] = (
            system_partition.max(axis=1) - system_partition.min(axis=1)
        )
        system_partition["mean"] = system_partition[
            [f"cv5_partition_{index}" for index in range(5)]
        ].mean(axis=1)
        for system_id, row in system_partition.sort_values(
            "range", ascending=False
        ).head(10).iterrows():
            system_rows.append(
                {
                    "method": method,
                    "system_id": system_id,
                    "partition_range": row["range"],
                    "partition_mean_error": row["mean"],
                    **{
                        f"partition_{index}": row[f"cv5_partition_{index}"]
                        for index in range(5)
                    },
                }
            )
    partitions = pd.DataFrame(partition_rows)
    systems = pd.DataFrame(system_rows)
    partitions.to_csv(output / "partition_primary_errors.csv", index=False)
    systems.to_csv(output / "partition_system_drivers.csv", index=False)
    return partitions


def mixed_diagnostics(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    rows = []
    detail_rows = []
    for protocol in sorted(
        assignments.loc[
            assignments["protocol"].str.startswith("cv5_partition_"),
            "protocol",
        ].unique()
    ):
        for fold_id in sorted(
            assignments.loc[
                assignments["protocol"].eq(protocol), "fold_id"
            ].unique()
        ):
            train_systems, validation_systems, test_systems = fold_role_sets(
                assignments, protocol, fold_id
            )
            train = _frame_for_systems(frame, set(train_systems))
            validation = _frame_for_systems(frame, set(validation_systems))
            test_all = _frame_for_systems(frame, set(test_systems))
            scaler = StrictTargetScaler.fit(train)
            supported = set(scaler.supported_groups)
            train_labeled = train.loc[
                train["target_group"].isin(supported)
            ].copy()
            validation_labeled = validation.loc[
                validation["target_group"].isin(supported)
            ].copy()
            test = test_all.loc[
                test_all["target_group"].isin(supported)
            ].copy()
            _, diagnostics = fit_hierarchical_mixed_detailed(
                train_labeled, test, scaler
            )
            detail_rows.append(
                {
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "rows": len(test),
                    **{
                        key: value
                        for key, value in diagnostics.items()
                        if key
                        in {
                            "converged",
                            "optimizer",
                            "variance_components",
                        }
                    },
                }
            )
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(output / "mixed_model_fit_status.csv", index=False)
    specification = {
        "formula": (
            "y ~ C(target_group) + log_total + is_cpu + "
            "accelerator_memory"
        ),
        "groups": "accelerator family",
        "random_intercepts": [
            "accelerator family",
            "submitter",
            "system",
        ],
        "heldout_system_random_effect": 0.0,
        "heldout_system_random_effect_rationale": (
            "Prediction uses fixed effects only; unseen system contributes "
            "no random-intercept estimate and no variance-component update."
        ),
        "converged_all_folds": bool(detail["converged"].all()),
        "fits": len(detail),
    }
    (output / "mixed_model_specification.json").write_text(
        json.dumps(specification, indent=2, sort_keys=True) + "\n"
    )

    aggregated = aggregate_predictions(predictions)
    aggregated["absolute_log_error"] = np.abs(
        aggregated["signed_log_prediction"] - aggregated["signed_log_result"]
    )
    group_partition = (
        aggregated.loc[
            aggregated["method"].isin(["hierarchical_mixed", "ridge"])
        ]
        .groupby(["method", "protocol", "target_group"], as_index=False)
        .agg(error=("absolute_log_error", "median"))
        .groupby(["method", "target_group"], as_index=False)
        .agg(mean_error=("error", "mean"))
    )
    wide = group_partition.pivot(
        index="target_group", columns="method", values="mean_error"
    ).reset_index()
    wide["mixed_minus_ridge"] = (
        wide["hierarchical_mixed"] - wide["ridge"]
    )
    wide.to_csv(output / "mixed_vs_ridge_by_group.csv", index=False)
    return wide


def main() -> None:
    output = Path(__import__("sys").argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frame = load_phase2_frame()
    assignments = pd.read_csv(ROOT / "folds" / "phase2_fold_assignments.csv")
    predictions = pd.read_csv(
        CV_BUNDLE / "predictions_by_partition_seed.csv"
    )
    gat_diagnostics(frame, assignments, output)
    graphsage_regression(frame, assignments, output)
    spearman_diagnostics(predictions, output)
    residual_diagnostics(predictions, output)
    partition_diagnostics(predictions, output)
    mixed_diagnostics(frame, assignments, predictions, output)

    protocol_hash = sha256_file(ROOT / "protocol.md")
    pre_manifest = json.loads(
        (PRE_RUN_BUNDLE / "results_manifest.json").read_text()
    )
    if pre_manifest["protocol_sha256"] != protocol_hash:
        raise AssertionError("Protocol hash changed since the CV5 run")
    cv_manifest_path = CV_BUNDLE / "results_manifest.json"
    cv_manifest = json.loads(cv_manifest_path.read_text())
    cv_manifest["protocol_sha256"] = protocol_hash
    cv_manifest["protocol_introduced_commit"] = "0291baa"
    cv_manifest["protocol_hash_unchanged_since_cv5_run"] = True
    cv_manifest["diagnostic_manifest_commit"] = git_output(
        "rev-parse", "--short", "HEAD"
    )
    cv_manifest_path.write_text(
        json.dumps(cv_manifest, indent=2, sort_keys=True) + "\n"
    )

    coverage = pd.read_csv(CV_BUNDLE / "partition_metrics.csv")
    coverage = (
        coverage.loc[coverage["method"].eq("mlp"), ["protocol", "rows", "coverage"]]
        .sort_values("protocol")
        .reset_index(drop=True)
    )
    coverage.loc[len(coverage)] = {
        "protocol": "mean",
        "rows": coverage["rows"].mean(),
        "coverage": coverage["coverage"].mean(),
    }
    coverage.to_csv(output / "coverage_reconciliation.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "protocol_sha256": protocol_hash,
        "cv_bundle_manifest_sha256": sha256_file(
            CV_BUNDLE / "results_manifest.json"
        ),
        "cv_predictions_sha256": sha256_file(
            CV_BUNDLE / "predictions_by_partition_seed.csv"
        ),
        "wall_seconds": time.perf_counter() - started,
        "loso_run": False,
        "sensitivity_run": False,
        "loafo_run": False,
        "output_sha256": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "results_manifest.json"
        },
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
