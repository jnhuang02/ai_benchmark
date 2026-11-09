"""Repeated-CV aggregation and system-clustered paired inference."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from evaluation.metrics import group_metrics


METRICS = (
    "primary_error",
    "median_row_mape",
    "median_group_spearman",
    "normalized_rmse",
    "coverage",
)
TOTAL_ROWS = 476
PAIR_COMPARISONS = (
    ("best_gnn", "mlp", "primary"),
    ("graphsage", "random_forest", "secondary"),
    ("graphsage", "graphsage_shuffled", "secondary"),
    ("gcn", "graphsage", "secondary"),
    ("gat", "graphsage", "secondary"),
)


def _metric_values(
    prediction_log: np.ndarray,
    signed_log_prediction: np.ndarray,
    signed_log_result: np.ndarray,
    normalized_prediction: np.ndarray,
    normalized_target: np.ndarray,
    direction: np.ndarray,
    target_codes: np.ndarray,
) -> dict[str, float]:
    prediction = np.exp(direction * signed_log_prediction)
    result = np.exp(direction * signed_log_result)
    absolute_log = np.abs(signed_log_prediction - signed_log_result)
    ape = np.abs(prediction - result) / result
    group_scores = []
    for code in np.unique(target_codes):
        mask = target_codes == code
        if (
            mask.sum() >= 3
            and np.unique(result[mask]).size >= 2
            and np.unique(prediction[mask]).size >= 2
        ):
            rho = np.corrcoef(
                rankdata(result[mask]), rankdata(prediction[mask])
            )[0, 1]
            if np.isfinite(rho):
                group_scores.append(float(rho))
    return {
        "primary_error": float(np.median(absolute_log)),
        "median_row_mape": float(np.median(ape)),
        "median_group_spearman": (
            float(np.median(group_scores))
            if group_scores
            else float("nan")
        ),
        "normalized_rmse": float(
            np.sqrt(np.mean((normalized_prediction - normalized_target) ** 2))
        ),
        "coverage": float("nan"),
    }


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method",
        "protocol",
        "fold_id",
        "row_uid",
        "system_id",
        "target_group",
        "result",
        "direction",
        "prediction_log",
        "signed_log_prediction",
        "y",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    group_columns = [
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
        predictions.groupby(group_columns, as_index=False)
        .agg(
            prediction_log=("prediction_log", "mean"),
            signed_log_prediction=("signed_log_prediction", "mean"),
            y=("y", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    aggregated["signed_log_result"] = (
        aggregated["direction"] * np.log(aggregated["result"])
    )
    aggregated["prediction"] = np.exp(
        aggregated["direction"] * aggregated["signed_log_prediction"]
    )
    return aggregated


def partition_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = TOTAL_ROWS
    aggregated = aggregate_predictions(predictions)
    rows = []
    seed_rows = []
    for (method, protocol), local in predictions.groupby(
        ["method", "protocol"], sort=True
    ):
        partition_view = aggregate_predictions(local)
        _, target_codes = np.unique(
            partition_view["target_group"], return_inverse=True
        )
        metrics = _metric_values(
            partition_view["prediction_log"].to_numpy(),
            partition_view["signed_log_prediction"].to_numpy(),
            partition_view["signed_log_result"].to_numpy(),
            partition_view["prediction_log"].to_numpy(),
            partition_view["y"].to_numpy(),
            partition_view["direction"].to_numpy(),
            target_codes,
        )
        metrics.update(
            {
                "method": method,
                "protocol": protocol,
                "rows": len(partition_view),
                "coverage": len(partition_view) / total_rows,
                "seed_count": int(local["seed"].nunique()),
            }
        )
        if method == "group_median":
            metrics["median_group_spearman"] = float("nan")
        rows.append(metrics)
        for seed, seed_frame in local.groupby("seed"):
            _, seed_codes = np.unique(
                seed_frame["target_group"], return_inverse=True
            )
            seed_metrics = _metric_values(
                seed_frame["prediction_log"].to_numpy(),
                seed_frame["signed_log_prediction"].to_numpy(),
                (
                    seed_frame["direction"].to_numpy()
                    * np.log(seed_frame["result"].to_numpy())
                ),
                seed_frame["prediction_log"].to_numpy(),
                seed_frame["y"].to_numpy(),
                seed_frame["direction"].to_numpy(),
                seed_codes,
            )
            seed_metrics.update(
                {
                    "method": method,
                    "protocol": protocol,
                    "seed": seed,
                    "rows": len(seed_frame),
                    "coverage": len(seed_frame) / total_rows,
                }
            )
            if method == "group_median":
                seed_metrics["median_group_spearman"] = float("nan")
            seed_rows.append(seed_metrics)
    partition = pd.DataFrame(rows)
    partition_summary = (
        partition.groupby("method", as_index=False)
        .agg(
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_partition_sd": (metric, "std")
                for metric in METRICS
            },
            partitions=("protocol", "nunique"),
            rows=("rows", "first"),
        )
    )
    seed_by_partition = pd.DataFrame(seed_rows)
    seed_averaged = (
        seed_by_partition.groupby(
            ["method", "protocol", "seed"], as_index=False
        )
        .agg(
            **{
                metric: (metric, "mean")
                for metric in METRICS
            }
        )
        .groupby(["method", "seed"], as_index=False)
        .agg(
            **{
                metric: (metric, "mean")
                for metric in METRICS
            }
        )
    )
    seed_summary = (
        seed_averaged.groupby("method", as_index=False)
        .agg(
            **{
                f"{metric}_seed_mean": (metric, "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_seed_sd": (metric, "std")
                for metric in METRICS
            },
            neural_seeds=("seed", "nunique"),
        )
    )
    summary = partition_summary.merge(
        seed_summary, on="method", how="left"
    )
    return summary, partition


def _prepare_partition_arrays(
    aggregated: pd.DataFrame,
) -> dict[str, dict]:
    prepared = {}
    for (method, partition), local in aggregated.groupby(
        ["method", "protocol"], sort=True
    ):
        local = local.reset_index(drop=True)
        _, target_codes = np.unique(
            local["target_group"], return_inverse=True
        )
        positions = defaultdict(list)
        for index, system_id in enumerate(local["system_id"]):
            positions[str(system_id)].append(index)
        prepared[(method, partition)] = {
            "prediction_log": local["prediction_log"].to_numpy(),
            "signed_log_prediction": local[
                "signed_log_prediction"
            ].to_numpy(),
            "signed_log_result": local["signed_log_result"].to_numpy(),
            "y": local["y"].to_numpy(),
            "direction": local["direction"].to_numpy(),
            "target_codes": target_codes,
            "positions": {
                key: np.asarray(value, dtype=int)
                for key, value in positions.items()
            },
        }
    return prepared


def bootstrap_repeated_cv(
    predictions: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20261107,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregated = aggregate_predictions(predictions)
    prepared = _prepare_partition_arrays(aggregated)
    methods = sorted(aggregated["method"].unique())
    partitions = sorted(aggregated["protocol"].unique())
    coverage_by_method = {
        method: (
            aggregated.loc[
                aggregated["method"].eq(method), "signed_log_prediction"
            ].shape[0]
            / (TOTAL_ROWS * len(partitions))
        )
        for method in methods
    }
    systems = np.array(
        sorted(aggregated["system_id"].unique()), dtype=object
    )
    rng = np.random.default_rng(seed)
    draw_rows = []
    for replicate in range(replicates):
        sampled = rng.choice(systems, size=len(systems), replace=True)
        for method in methods:
            partition_values = {
                metric: [] for metric in METRICS
            }
            for partition in partitions:
                arrays = prepared[(method, partition)]
                selected = np.concatenate(
                    [
                        arrays["positions"][str(system)]
                        for system in sampled
                        if str(system) in arrays["positions"]
                    ]
                )
                metrics = _metric_values(
                    arrays["prediction_log"][selected],
                    arrays["signed_log_prediction"][selected],
                    arrays["signed_log_result"][selected],
                    arrays["prediction_log"][selected],
                    arrays["y"][selected],
                    arrays["direction"][selected],
                    arrays["target_codes"][selected],
                )
                if method == "group_median":
                    metrics["median_group_spearman"] = float("nan")
                for metric in METRICS:
                    partition_values[metric].append(metrics[metric])
            row = {
                "bootstrap_replicate": replicate,
                "method": method,
            }
            for metric in METRICS:
                values = np.asarray(partition_values[metric], dtype=float)
                row[metric] = (
                    float(np.nanmean(values))
                    if np.isfinite(values).any()
                    else float("nan")
                )
            # Coverage is a fixed property of the evaluated-set definition.
            row["coverage"] = float(coverage_by_method[method])
            draw_rows.append(row)
    draws = pd.DataFrame(draw_rows)
    summary_rows = []
    for method, local in draws.groupby("method", sort=True):
        row = {"method": method}
        for metric in METRICS:
            row[f"{metric}_95ci_low"] = float(
                local[metric].quantile(0.025)
            )
            row[f"{metric}_95ci_high"] = float(
                local[metric].quantile(0.975)
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    pair_rows = []
    for left, right, role in PAIR_COMPARISONS:
        wide = draws.pivot(
            index="bootstrap_replicate", columns="method", values="primary_error"
        )
        difference = wide[left] - wide[right]
        low = float(difference.quantile(0.025))
        high = float(difference.quantile(0.975))
        if high < -0.05:
            status = "better"
        elif low > 0.05:
            status = "worse"
        elif low >= -0.05 and high <= 0.05:
            status = "equivalent"
        else:
            status = "inconclusive"
        p_value = min(
            1.0,
            2.0
            * min(
                float((difference >= 0).mean()),
                float((difference <= 0).mean()),
            ),
        )
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "role": role,
                "mean_difference": float(difference.mean()),
                "difference_95ci_low": low,
                "difference_95ci_high": high,
                "probability_left_better": float((difference < 0).mean()),
                "bootstrap_p_value": p_value,
                "status": status,
                "secondary_holm_status": (
                    "pending_LOSO"
                    if role == "secondary"
                    else "not_applicable"
                ),
            }
        )
    return summary, draws, pd.DataFrame(pair_rows)


def protocol_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize one protocol using pooled OOF predictions."""
    total_rows = TOTAL_ROWS
    aggregated = aggregate_predictions(predictions)
    rows = []
    for method, local in aggregated.groupby("method", sort=True):
        _, codes = np.unique(local["target_group"], return_inverse=True)
        metrics = _metric_values(
            local["prediction_log"].to_numpy(),
            local["signed_log_prediction"].to_numpy(),
            local["signed_log_result"].to_numpy(),
            local["prediction_log"].to_numpy(),
            local["y"].to_numpy(),
            local["direction"].to_numpy(),
            codes,
        )
        metrics["coverage"] = len(local) / total_rows
        metrics["rows"] = len(local)
        if method == "group_median":
            metrics["median_group_spearman"] = float("nan")
        seed_metrics = []
        for _, seed_frame in predictions.loc[
            predictions["method"].eq(method)
        ].groupby("seed"):
            _, seed_codes = np.unique(
                seed_frame["target_group"], return_inverse=True
            )
            seed_metrics.append(
                _metric_values(
                    seed_frame["prediction_log"].to_numpy(),
                    seed_frame["signed_log_prediction"].to_numpy(),
                    (
                        seed_frame["direction"].to_numpy()
                        * np.log(seed_frame["result"].to_numpy())
                    ),
                    seed_frame["prediction_log"].to_numpy(),
                    seed_frame["y"].to_numpy(),
                    seed_frame["direction"].to_numpy(),
                    seed_codes,
                )
            )
        seed_table = pd.DataFrame(seed_metrics)
        if method == "group_median":
            seed_table["median_group_spearman"] = float("nan")
        row = {"method": method, **metrics}
        for metric in METRICS:
            row[f"{metric}_seed_mean"] = seed_table[metric].mean()
            row[f"{metric}_seed_sd"] = seed_table[metric].std(ddof=1)
        row["neural_seeds"] = int(
            predictions.loc[predictions["method"].eq(method), "seed"].nunique()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def group_appendix(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregated = aggregate_predictions(predictions)
    rows = []
    for (method, protocol), local in aggregated.groupby(
        ["method", "protocol"], sort=True
    ):
        for row in group_metrics(local).to_dict(orient="records"):
            if method == "group_median":
                row["spearman"] = float("nan")
            row.update({"method": method, "protocol": protocol})
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_mape_appendix(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregated = aggregate_predictions(predictions).copy()
    aggregated["ape"] = (
        np.abs(aggregated["prediction"] - aggregated["result"])
        / aggregated["result"]
    )
    by_partition = (
        aggregated.groupby(["method", "protocol"], as_index=False)
        .agg(pooled_mape=("ape", "mean"), rows=("row_uid", "size"))
    )
    return (
        by_partition.groupby("method", as_index=False)
        .agg(
            pooled_mape_mean=("pooled_mape", "mean"),
            pooled_mape_partition_sd=("pooled_mape", "std"),
            rows=("rows", "first"),
            partitions=("protocol", "nunique"),
        )
        .sort_values("method")
    )


def selected_gnn_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["method"].eq("best_gnn")]
    selections = selected.groupby(
        ["protocol", "fold_id", "seed"], as_index=False
    ).agg(selected_gnn=("selected_gnn", "first"))
    return (
        selections.groupby("selected_gnn", as_index=False)
        .size()
        .rename(columns={"size": "fold_seed_selections"})
    )
