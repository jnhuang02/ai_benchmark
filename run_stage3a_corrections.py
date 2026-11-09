#!/usr/bin/env python3
"""Generate accepted Phase 2.5 corrections and Stage 3A evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file


ROOT = Path(__file__).resolve().parent
LOAFO = ROOT / "results" / "phase2-loafo-final-1a65664"
FAMILY_MAP = ROOT / "folds" / "loafo_family_mapping_proposed.csv"
SYSTEM_ASSIGNMENT = ROOT / "folds" / "loafo_system_assignment_proposed.csv"
PROTOCOL = ROOT / "protocol.md"
PROTOCOL_HISTORY = ROOT / "protocol_history.md"
HEADLINE = (
    "amd_cdna4_mi350x",
    "amd_cdna4_mi355x",
    "intel_xe2_battlemage_arc_pro_b70",
    "nvidia_blackwell_b200",
    "nvidia_blackwell_b300",
    "nvidia_blackwell_rtx_pro",
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def family_support(
    predictions: pd.DataFrame,
    main_table: pd.DataFrame,
    families: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for family_id, local in main_table.groupby("family_id", sort=True):
        prediction_rows = predictions.loc[
            predictions["fold_id"].eq(local["fold_id"].iloc[0])
        ]
        supported_systems = int(prediction_rows["system_id"].nunique())
        family = families.set_index("family_id").loc[family_id]
        rows.append(
            {
                "fold_id": local["fold_id"].iloc[0],
                "family_id": family_id,
                "device_class": family["device_class"],
                "holdout_disposition": family["holdout_disposition"],
                "systems": int(family["systems"]),
                "supported_systems": supported_systems,
                "rows": int(family["rows"]),
                "supported_rows": int(local["supported_rows"].iloc[0]),
                "interval_eligible": supported_systems >= 8,
            }
        )
    return pd.DataFrame(rows)


def system_coverage(
    predictions: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        predictions.groupby("system_id", as_index=False)
        .size()
        .rename(columns={"size": "prediction_rows"})
    )
    output = assignment.merge(
        counts, on="system_id", how="left", validate="one_to_one"
    )
    output["prediction_rows"] = output["prediction_rows"].fillna(0).astype(int)
    output["tested_in_loafo"] = output["prediction_rows"].gt(0)
    mixed = output["proposed_family_id"].isin(
        assignment.loc[
            assignment["proposed_family_id"].str.startswith("mixed_"),
            "proposed_family_id",
        ].unique()
    )
    output["coverage_status"] = "tested"
    output.loc[mixed, "coverage_status"] = "always_train_mixed"
    output.loc[
        ~mixed & ~output["tested_in_loafo"],
        "coverage_status",
    ] = "heldout_no_supported_rows"
    return output[
        [
            "system_id",
            "proposed_family_id",
            "suite",
            "submitter",
            "prediction_rows",
            "tested_in_loafo",
            "coverage_status",
        ]
    ].rename(columns={"proposed_family_id": "family_id"})


def group_median_comparisons(main_table: pd.DataFrame) -> pd.DataFrame:
    baseline = main_table.loc[
        main_table["method"].eq("group_median"),
        ["fold_id", "family_id", "primary_error"],
    ].rename(columns={"primary_error": "group_median_error"})
    output = main_table.merge(
        baseline,
        on=["fold_id", "family_id"],
        how="left",
        validate="many_to_one",
    )
    output["model_minus_group_median"] = (
        output["primary_error"] - output["group_median_error"]
    )
    output["model_better_than_group_median"] = output[
        "model_minus_group_median"
    ].lt(0)
    return output[
        [
            "fold_id",
            "family_id",
            "device_class",
            "holdout_disposition",
            "method",
            "primary_error",
            "group_median_error",
            "model_minus_group_median",
            "model_better_than_group_median",
        ]
    ]


def headline_counts(main_table: pd.DataFrame) -> pd.DataFrame:
    headline = main_table.loc[main_table["family_id"].isin(HEADLINE)]
    wide = headline.pivot(
        index="family_id", columns="method", values="primary_error"
    )
    best_gnn_better = (wide["best_gnn"] < wide["mlp"]).sort_index()
    return pd.DataFrame(
        [
            {
                "quantity": "headline_families",
                "value": len(HEADLINE),
                "families": ";".join(HEADLINE),
            },
            {
                "quantity": "best_gnn_point_better_than_mlp",
                "value": int(best_gnn_better.sum()),
                "families": ";".join(best_gnn_better.index[best_gnn_better]),
            },
        ]
    )


def partial_pooling_status() -> dict[str, object]:
    return {
        "analysis": "sparse_group_partial_pooling_secondary",
        "status": "not_run",
        "deviation_id": "S3A-SHRINK-001",
        "label": "secondary protocol item; not a primary analysis",
        "specified_in_hashed_protocol": True,
        "reason": (
            "Protocol v2.1 names a 'shrinkage posterior' but does not define "
            "the prior or posterior formula, the group-size shrinkage "
            "parameter, variance estimation, whether both center and scale "
            "are shrunk independently, or how the normalized primary error "
            "is evaluated after shrinkage."
        ),
        "decision": (
            "Do not invent an empirical-Bayes convention after results are "
            "known. Record the secondary analysis as not executable from the "
            "frozen specification and retain primary exclusion results."
        ),
    }


def run_test_suite(output: Path) -> tuple[str, pd.DataFrame, int, int]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = completed.stdout
    (output / "test_suite_output.txt").write_text(text)
    matches = re.findall(r"^test_.*? \((test_[a-zA-Z0-9_]+)\.", text, re.MULTILINE)
    if not matches:
        raise AssertionError("Could not parse verbose unittest output")
    counts = (
        pd.Series(matches, name="module")
        .value_counts()
        .rename_axis("module")
        .reset_index(name="tests")
        .sort_values("module")
    )
    total_match = re.search(r"Ran (\d+) tests", text)
    if total_match is None:
        raise AssertionError("Could not find unittest total")
    total = int(total_match.group(1))
    if completed.returncode != 0 or total != 21:
        raise AssertionError("Full test suite did not pass exactly 21 tests")
    targeted = int(
        counts.loc[
            counts["module"].isin(["test_phase2", "test_folds"]), "tests"
        ].sum()
    )
    counts.to_csv(output / "test_module_counts.csv", index=False)
    return text, counts, total, targeted


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(LOAFO / "predictions_by_fold_seed.csv")
    main_table = pd.read_csv(LOAFO / "main_table_by_family.csv")
    families = pd.read_csv(FAMILY_MAP)
    assignment = pd.read_csv(SYSTEM_ASSIGNMENT)

    support = family_support(predictions, main_table, families)
    support.to_csv(output / "loafo_family_support.csv", index=False)
    coverage = system_coverage(predictions, assignment)
    coverage.to_csv(output / "loafo_system_coverage.csv", index=False)
    comparisons = group_median_comparisons(main_table)
    comparisons.to_csv(
        output / "loafo_group_median_comparisons.csv", index=False
    )
    counts = headline_counts(main_table)
    counts.to_csv(output / "headline_claim_counts.csv", index=False)
    shrinkage = partial_pooling_status()
    (output / "partial_pooling_status.json").write_text(
        json.dumps(shrinkage, indent=2, sort_keys=True) + "\n"
    )
    test_text, test_counts, total_tests, targeted_tests = run_test_suite(output)
    if len(predictions["system_id"].unique()) != 97:
        raise AssertionError("LOAFO prediction file no longer contains 97 systems")
    if coverage["tested_in_loafo"].sum() != 97:
        raise AssertionError("System coverage does not reconcile to 97")
    if int(counts.loc[
        counts["quantity"].eq("best_gnn_point_better_than_mlp"), "value"
    ].iloc[0]) != 2:
        raise AssertionError("Expected two headline families with best GNN < MLP")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "protocol_sha256": sha256_file(PROTOCOL),
        "protocol_history_sha256": sha256_file(PROTOCOL_HISTORY),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "loafo_predictions_sha256": sha256_file(
            LOAFO / "predictions_by_fold_seed.csv"
        ),
        "loafo_main_table_sha256": sha256_file(
            LOAFO / "main_table_by_family.csv"
        ),
        "loafo_unique_tested_systems": int(
            predictions["system_id"].nunique()
        ),
        "loafo_total_systems_accounted_for": int(len(coverage)),
        "partial_pooling_status": shrinkage["status"],
        "partial_pooling_deviation_id": shrinkage["deviation_id"],
        "full_test_suite": {
            "tests": total_tests,
            "targeted_phase2_and_fold_tests": targeted_tests,
            "module_counts": dict(
                zip(test_counts["module"], test_counts["tests"])
            ),
            "output_sha256": sha256_file(output / "test_suite_output.txt"),
        },
        "output_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
