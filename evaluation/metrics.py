"""Performance metrics and system-clustered bootstrap inference."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PRIMARY_METRIC = "median_absolute_log_ratio"
METRIC_COLUMNS = [
    PRIMARY_METRIC,
    "mape",
    "normalized_rmse",
    "median_group_absolute_log_ratio",
    "median_group_mape",
    "median_group_spearman",
]


def _safe_spearman(frame: pd.DataFrame, minimum_rows: int = 3) -> float:
    if (
        len(frame) < minimum_rows
        or frame["result"].nunique() < 2
        or frame["prediction"].nunique() < 2
    ):
        return float("nan")
    return float(frame["result"].rank().corr(frame["prediction"].rank()))


def group_metrics(
    predictions: pd.DataFrame, *, spearman_minimum_rows: int = 3
) -> pd.DataFrame:
    rows = []
    for target_group, group in predictions.groupby("target_group", sort=True):
        absolute_log_ratio = np.abs(
            np.log(group["prediction"].to_numpy(dtype=float))
            - np.log(group["result"].to_numpy(dtype=float))
        )
        absolute_percentage_error = (
            np.abs(group["prediction"] - group["result"]) / group["result"]
        )
        rows.append(
            {
                "target_group": target_group,
                "rows": len(group),
                "systems": group["system_id"].nunique(),
                "median_absolute_log_ratio": float(
                    np.median(absolute_log_ratio)
                ),
                "mape": float(np.mean(absolute_percentage_error)),
                "median_mape": float(np.median(absolute_percentage_error)),
                "spearman": _safe_spearman(
                    group, minimum_rows=spearman_minimum_rows
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_metrics(
    predictions: pd.DataFrame, *, spearman_minimum_rows: int = 3
) -> dict[str, float]:
    required = {"prediction", "result", "prediction_log", "y", "target_group"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {required - set(predictions)}")
    if predictions.empty:
        raise ValueError("Cannot calculate metrics on an empty prediction set")
    if predictions["prediction"].le(0).any() or predictions["result"].le(0).any():
        raise ValueError("Predictions and results must be positive")
    absolute_log_ratio = np.abs(
        np.log(predictions["prediction"].to_numpy(dtype=float))
        - np.log(predictions["result"].to_numpy(dtype=float))
    )
    absolute_percentage_error = (
        np.abs(predictions["prediction"] - predictions["result"])
        / predictions["result"]
    )
    groups = group_metrics(
        predictions, spearman_minimum_rows=spearman_minimum_rows
    )
    return {
        PRIMARY_METRIC: float(np.median(absolute_log_ratio)),
        "mape": float(np.mean(absolute_percentage_error)),
        "median_row_mape": float(np.median(absolute_percentage_error)),
        "normalized_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        predictions["prediction_log"].to_numpy(dtype=float)
                        - predictions["y"].to_numpy(dtype=float)
                    )
                    ** 2
                )
            )
        ),
        "median_group_absolute_log_ratio": float(
            groups["median_absolute_log_ratio"].median()
        ),
        "median_group_mape": float(groups["median_mape"].median()),
        "median_group_spearman": float(groups["spearman"].median()),
        "groups": float(len(groups)),
        "rows": float(len(predictions)),
        "systems": float(predictions["system_id"].nunique()),
    }


def metric_summary_by_seed(
    metrics_by_seed: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    value_columns = [
        column
        for column in METRIC_COLUMNS + ["median_row_mape", "groups", "rows", "systems"]
        if column in metrics_by_seed.columns
    ]
    summary = (
        metrics_by_seed.groupby(group_columns)[value_columns]
        .agg(["mean", "std", "min", "median", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in column if part])
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def _system_indices(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, system_id in enumerate(predictions["system_id"]):
        indices[str(system_id)].append(index)
    return {
        system_id: np.asarray(values, dtype=int)
        for system_id, values in indices.items()
    }


def clustered_bootstrap(
    predictions: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20261107,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample whole systems and compute all model metrics on each draw."""
    group_columns = ["method"]
    if "evaluation_policy" in predictions.columns:
        group_columns.insert(0, "evaluation_policy")
    identifiers = np.array(sorted(predictions["system_id"].unique()))
    model_groups = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.reset_index(drop=True)
        _, target_codes = np.unique(
            group["target_group"].to_numpy(), return_inverse=True
        )
        positions = _system_indices(group)
        model_groups.append(
            {
                "keys": keys,
                "prediction": group["prediction"].to_numpy(dtype=float),
                "result": group["result"].to_numpy(dtype=float),
                "prediction_log": group["prediction_log"].to_numpy(dtype=float),
                "y": group["y"].to_numpy(dtype=float),
                "target_codes": target_codes,
                "positions": positions,
                "systems": group["system_id"].to_numpy(),
            }
        )
    rng = np.random.default_rng(seed)
    draws = []
    for replicate in range(replicates):
        sampled_systems = rng.choice(identifiers, size=len(identifiers), replace=True)
        for model_group in model_groups:
            selected = np.concatenate(
                [
                    model_group["positions"][str(system)]
                    for system in sampled_systems
                    if str(system) in model_group["positions"]
                ]
            )
            row = dict(zip(group_columns, model_group["keys"]))
            row["bootstrap_replicate"] = replicate
            selected_result = model_group["result"][selected]
            selected_prediction = model_group["prediction"][selected]
            selected_log_error = np.abs(
                np.log(selected_prediction) - np.log(selected_result)
            )
            selected_ape = (
                np.abs(selected_prediction - selected_result) / selected_result
            )
            normalized_error = (
                model_group["prediction_log"][selected]
                - model_group["y"][selected]
            )
            row[PRIMARY_METRIC] = float(np.median(selected_log_error))
            row["mape"] = float(np.mean(selected_ape))
            row["normalized_rmse"] = float(
                np.sqrt(np.mean(normalized_error**2))
            )
            selected_codes = model_group["target_codes"][selected]
            group_log = []
            group_ape = []
            group_rho = []
            for code in np.unique(selected_codes):
                group_mask = selected_codes == code
                group_log.append(float(np.median(selected_log_error[group_mask])))
                group_ape.append(float(np.mean(selected_ape[group_mask])))
                truth = selected_result[group_mask]
                prediction = selected_prediction[group_mask]
                if len(truth) >= 3 and len(np.unique(truth)) > 1 and len(
                    np.unique(prediction)
                ) > 1:
                    rho = np.corrcoef(rankdata(truth), rankdata(prediction))[0, 1]
                    if np.isfinite(rho):
                        group_rho.append(float(rho))
            row["median_group_absolute_log_ratio"] = float(np.median(group_log))
            row["median_group_mape"] = float(np.median(group_ape))
            row["median_group_spearman"] = (
                float(np.median(group_rho)) if group_rho else float("nan")
            )
            draws.append(row)
    draws_frame = pd.DataFrame(draws)
    summary_rows = []
    for keys, group in draws_frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in METRIC_COLUMNS:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_95ci_low"] = (
                float(np.quantile(values, 0.025)) if len(values) else float("nan")
            )
            row[f"{metric}_95ci_high"] = (
                float(np.quantile(values, 0.975)) if len(values) else float("nan")
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), draws_frame


def paired_bootstrap_differences(
    draws: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Difference left minus right, paired within bootstrap systems draw."""
    subset = draws.loc[draws["method"].isin([left, right])]
    wide = subset.pivot_table(
        index=["bootstrap_replicate", "evaluation_policy"]
        if "evaluation_policy" in subset.columns
        else ["bootstrap_replicate"],
        columns="method",
        values=metric,
        aggfunc="first",
    ).reset_index()
    if left not in wide or right not in wide:
        raise ValueError(f"Missing paired model in bootstrap draws: {left}, {right}")
    wide["difference"] = wide[left] - wide[right]
    group_columns = (
        ["evaluation_policy"] if "evaluation_policy" in wide.columns else []
    )
    rows = []
    groups = (
        wide.groupby(group_columns, sort=True)
        if group_columns
        else [((), wide)]
    )
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "left_model": left,
                "right_model": right,
                "metric": metric,
                "mean_difference": float(group["difference"].mean()),
                "difference_95ci_low": float(
                    group["difference"].quantile(0.025)
                ),
                "difference_95ci_high": float(
                    group["difference"].quantile(0.975)
                ),
                "probability_left_better": float(
                    (group["difference"] < 0).mean()
                ),
                "bootstrap_replicates": len(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_policy_differences(
    draws: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Pair two evaluation policies within the same system-bootstrap draw."""
    subset = draws.loc[draws["evaluation_policy"].isin([left, right])]
    wide = subset.pivot_table(
        index="bootstrap_replicate",
        columns="evaluation_policy",
        values=metric,
        aggfunc="first",
    ).reset_index()
    if left not in wide or right not in wide:
        raise ValueError(f"Missing paired policy: {left}, {right}")
    wide["difference"] = wide[left] - wide[right]
    return pd.DataFrame(
        [
            {
                "left_policy": left,
                "right_policy": right,
                "metric": metric,
                "mean_difference": float(wide["difference"].mean()),
                "difference_95ci_low": float(
                    wide["difference"].quantile(0.025)
                ),
                "difference_95ci_high": float(
                    wide["difference"].quantile(0.975)
                ),
                "probability_left_better": float(
                    (wide["difference"] < 0).mean()
                ),
                "bootstrap_replicates": len(wide),
            }
        ]
    )
