#!/usr/bin/env python3
"""B1: saved-prediction paired bootstrap for GraphSAGE/GCN versus MLP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import sha256_file
from evaluation.phase2_metrics import partition_metrics, protocol_summary
from exploratory_manifest import ALLOWED_LABEL, write_manifest


ROOT = Path(__file__).resolve().parent
CV_BUNDLE = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
LOSO_BUNDLE = ROOT / "results" / "phase2-loso-final-a689ba5"
PAIRS = (("graphsage", "mlp"), ("gcn", "mlp"))
METHODS = ("graphsage", "gcn", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def observed_errors(
    cv_predictions: pd.DataFrame, loso_predictions: pd.DataFrame
) -> pd.DataFrame:
    cv_summary, _ = partition_metrics(cv_predictions)
    cv = (
        cv_summary.loc[cv_summary["method"].isin(METHODS), ["method", "primary_error_mean"]]
        .rename(columns={"primary_error_mean": "observed_error"})
        .assign(protocol="cv5_partitioned")
    )
    loso_summary = protocol_summary(loso_predictions)
    loso = (
        loso_summary.loc[loso_summary["method"].isin(METHODS), ["method", "primary_error"]]
        .rename(columns={"primary_error": "observed_error"})
        .assign(protocol="loso")
    )
    return pd.concat([cv, loso], ignore_index=True).sort_values(
        ["protocol", "method"]
    )


def paired_summary(
    draws: pd.DataFrame,
    *,
    protocol: str,
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    draw_rows = []
    wide = draws.pivot(
        index="bootstrap_replicate", columns="method", values="primary_error"
    )
    observed_local = observed.loc[observed["protocol"].eq(protocol)].set_index(
        "method"
    )["observed_error"]
    for left, right in PAIRS:
        difference = wide[left] - wide[right]
        observed_difference = float(observed_local[left] - observed_local[right])
        rows.append(
            {
                "protocol": protocol,
                "left": left,
                "right": right,
                "observed_left_error": float(observed_local[left]),
                "observed_right_error": float(observed_local[right]),
                "observed_difference": observed_difference,
                "bootstrap_mean_difference": float(difference.mean()),
                "difference_95ci_low": float(difference.quantile(0.025)),
                "difference_95ci_high": float(difference.quantile(0.975)),
                "probability_left_better": float((difference < 0).mean()),
                "bootstrap_replicates": int(len(difference)),
            }
        )
        draw_rows.append(
            pd.DataFrame(
                {
                    "protocol": protocol,
                    "left": left,
                    "right": right,
                    "bootstrap_replicate": difference.index,
                    "difference": difference.to_numpy(),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(draw_rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv_predictions_path = CV_BUNDLE / "predictions_by_partition_seed.csv"
    loso_predictions_path = LOSO_BUNDLE / "predictions_by_fold_seed.csv"
    cv_draws_path = CV_BUNDLE / "bootstrap_by_method.csv"
    loso_draws_path = LOSO_BUNDLE / "bootstrap_by_method.csv"
    cv_predictions = pd.read_csv(cv_predictions_path)
    loso_predictions = pd.read_csv(loso_predictions_path)
    cv_draws = pd.read_csv(cv_draws_path)
    loso_draws = pd.read_csv(loso_draws_path)

    observed = observed_errors(cv_predictions, loso_predictions)
    cv_summary, cv_pair_draws = paired_summary(
        cv_draws, protocol="cv5_partitioned", observed=observed
    )
    loso_summary, loso_pair_draws = paired_summary(
        loso_draws, protocol="loso", observed=observed
    )
    summary = pd.concat([cv_summary, loso_summary], ignore_index=True)
    pair_draws = pd.concat([cv_pair_draws, loso_pair_draws], ignore_index=True)
    observed.to_csv(output / "observed_model_errors.csv", index=False)
    summary.to_csv(output / "pair_summary.csv", index=False)
    pair_draws.to_csv(output / "paired_bootstrap_draws.csv", index=False)
    (output / "README.md").write_text(
        "\n".join(
            [
                "# B1: saved-prediction GNN-minus-MLP bootstrap",
                "",
                f"Label: **{ALLOWED_LABEL}**.",
                "",
                "This analysis does not refit any model. It replays paired "
                "system-clustered bootstrap draws already computed from the final "
                "saved CV5 and LOSO prediction bundles, and verifies the observed "
                "point estimates directly from those predictions.",
                "",
                "GraphSAGE and GCN are fixed architecture candidates; this "
                "comparison is not a replacement for the frozen best-GNN-versus-MLP "
                "co-primary hypothesis.",
            ]
        )
        + "\n"
    )
    write_manifest(
        output,
        root=ROOT,
        command=[sys.executable, str(Path(__file__)), *sys.argv[1:]],
        input_files={
            "cv5_predictions": cv_predictions_path,
            "loso_predictions": loso_predictions_path,
            "cv5_bootstrap_draws": cv_draws_path,
            "loso_bootstrap_draws": loso_draws_path,
            "cv5_manifest": CV_BUNDLE / "results_manifest.json",
            "loso_manifest": LOSO_BUNDLE / "results_manifest.json",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
