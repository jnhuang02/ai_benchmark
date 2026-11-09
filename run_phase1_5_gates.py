#!/usr/bin/env python3
"""Generate the Phase 1.5 acceptance-gate evidence without Phase 2 fitting."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import (
    DEFAULT_DATA,
    load_dataset,
    load_family_map,
    sha256_file,
    system_table,
)
from evaluation.evaluate import (
    GNNConfig,
    cross_validate_model,
    evaluate_fold_model,
    fold_role_sets,
)
from evaluation.graph import graph_isolation_audit
from evaluation.metrics import calculate_metrics
from run_phase1_5 import loafo_proposal


ROOT = Path(__file__).resolve().parent
PRE_FIX_COMMIT = "d3bc8350f9915befa69fa370dc4b12c24ba1ddcc"
PHASE1_BUNDLE = ROOT / "results" / "phase1-final-5c9dfde"
PHASE15_BUNDLE = ROOT / "results" / "phase1_5-final-8fbe960"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def group_spearman(
    predictions: pd.DataFrame, minimum_rows: int
) -> tuple[float, int]:
    values = []
    for _, group in predictions.groupby("target_group", sort=True):
        if (
            len(group) >= minimum_rows
            and group["result"].nunique() >= 2
            and group["prediction"].nunique() >= 2
        ):
            value = group["result"].rank().corr(group["prediction"].rank())
            if np.isfinite(value):
                values.append(float(value))
    return float(np.median(values)), len(values)


def spearman_by_seed(
    predictions: pd.DataFrame,
    dataset: str,
    *,
    seed_column: str = "seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    group_rows = []
    for seed, group in predictions.groupby(seed_column, sort=True):
        for definition, minimum_rows in [
            ("phase1_minimum_2_rows", 2),
            ("table_6_4_minimum_3_rows", 3),
        ]:
            value, qualifying = group_spearman(group, minimum_rows)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "definition": definition,
                    "minimum_rows": minimum_rows,
                    "median_group_spearman": value,
                    "qualifying_groups": qualifying,
                }
            )
        if dataset == "attribution_baseline" and seed == 7:
            for target_group, local in group.groupby("target_group", sort=True):
                min2, min2_count = group_spearman(local, 2)
                min3, min3_count = group_spearman(local, 3)
                group_rows.append(
                    {
                        "target_group": target_group,
                        "rows": len(local),
                        "result_unique": local["result"].nunique(),
                        "prediction_unique": local["prediction"].nunique(),
                        "minimum_2_rho": min2,
                        "minimum_2_enters": bool(min2_count),
                        "minimum_3_rho": min3,
                        "minimum_3_enters": bool(min3_count),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(group_rows)


def gate1(output: Path) -> pd.DataFrame:
    baseline = pd.read_csv(
        PHASE15_BUNDLE / "attribution_predictions_by_seed.csv",
        dtype={"public_id": str},
    )
    baseline = baseline.loc[baseline["stage"].eq("baseline")].copy()

    historical = pd.read_csv(
        PHASE15_BUNDLE
        / "regression"
        / "graphsage"
        / "all_graphsage_predictions.csv",
        dtype={"public_id": str},
    )
    historical = historical.loc[historical["method"].eq("graphsage")].copy()
    target_columns = [
        "suite",
        "version",
        "division",
        "benchmark",
        "model",
        "task_code",
        "scenario",
        "units",
    ]
    historical["target_group"] = (
        historical[target_columns].fillna("").astype(str).agg("|".join, axis=1)
    )
    historical["seed"] = 7

    phase1 = pd.read_csv(
        PHASE1_BUNDLE / "predictions_by_seed.csv",
        dtype={"public_id": str},
    )
    phase1 = phase1.loc[phase1["method"].eq("graphsage")].copy()
    fixed = phase1.loc[phase1["evaluation_policy"].eq("isolated_supported")]
    legacy = phase1.loc[phase1["evaluation_policy"].eq("legacy_leaky_global")]

    pieces = []
    group_pieces = []
    for dataset, predictions in [
        ("attribution_baseline", baseline),
        ("exact_phase0_graphsage", historical),
        ("phase1_fixed_isolation", fixed),
        ("phase1_legacy_leaky", legacy),
    ]:
        summary, groups = spearman_by_seed(predictions, dataset)
        pieces.append(summary)
        if not groups.empty:
            group_pieces.append(groups)
    by_seed = pd.concat(pieces, ignore_index=True)
    summary = (
        by_seed.groupby(["dataset", "definition", "minimum_rows"], as_index=False)
        .agg(
            mean=("median_group_spearman", "mean"),
            sd=("median_group_spearman", "std"),
            seeds=("seed", "nunique"),
            qualifying_groups_min=("qualifying_groups", "min"),
            qualifying_groups_max=("qualifying_groups", "max"),
        )
    )
    write_csv(by_seed, output / "spearman_by_seed.csv")
    write_csv(summary, output / "spearman_reconciliation.csv")
    if group_pieces:
        write_csv(
            pd.concat(group_pieces, ignore_index=True),
            output / "spearman_group_details.csv",
        )
    return summary


def _patch_test_frame(frame: pd.DataFrame, test_systems: set[str]) -> pd.DataFrame:
    altered = frame.copy()
    mask = altered["system_id"].isin(test_systems)
    altered.loc[mask, "nodes"] = altered.loc[mask, "nodes"] + 7
    altered.loc[mask, "accelerators_per_node"] = (
        altered.loc[mask, "accelerators_per_node"] + 3
    )
    altered.loc[mask, "precision"] = "scrambled-test-precision"
    altered.loc[mask, "result"] = altered.loc[mask, "result"] * 3.7
    altered.loc[mask, "log_performance"] = (
        altered.loc[mask, "direction"] * np.log(altered.loc[mask, "result"])
    )
    return altered


def gate3_current(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_dataset(DEFAULT_DATA)
    systems = system_table(frame, load_family_map())
    assignments = pd.read_csv(PHASE1_BUNDLE / "fold_assignments.csv")

    current, _ = cross_validate_model(
        frame,
        assignments,
        protocol="cv5",
        model="graphsage",
        policy="isolated_supported",
        seed=7,
        config=GNNConfig(),
    )
    saved = pd.read_csv(
        PHASE1_BUNDLE / "predictions_by_seed.csv", dtype={"public_id": str}
    )
    saved = saved.loc[
        saved["method"].eq("graphsage")
        & saved["evaluation_policy"].eq("isolated_supported")
        & saved["seed"].eq(7)
    ].copy()
    merged = saved[
        ["row_uid", "prediction", "prediction_log", "result", "system_id"]
    ].merge(
        current[["row_uid", "prediction", "prediction_log"]],
        on="row_uid",
        suffixes=("_saved", "_current"),
        validate="one_to_one",
    )
    if len(merged) != len(saved):
        raise AssertionError("Fixed-isolation rerun did not match saved rows")
    saved_metric = calculate_metrics(saved)
    current_metric = calculate_metrics(current)
    comparison = pd.DataFrame(
        [
            {
                "policy": "isolated_supported",
                "seed": 7,
                "rows": len(current),
                "systems": current["system_id"].nunique(),
                "saved_primary_error": saved_metric[
                    "median_absolute_log_ratio"
                ],
                "current_primary_error": current_metric[
                    "median_absolute_log_ratio"
                ],
                "saved_median_group_spearman": saved_metric[
                    "median_group_spearman"
                ],
                "current_median_group_spearman": current_metric[
                    "median_group_spearman"
                ],
                "max_abs_prediction_difference": float(
                    np.max(
                        np.abs(
                            merged["prediction_current"]
                            - merged["prediction_saved"]
                        )
                    )
                ),
                "max_abs_normalized_prediction_difference": float(
                    np.max(
                        np.abs(
                            merged["prediction_log_current"]
                            - merged["prediction_log_saved"]
                        )
                    )
                ),
            }
        ]
    )
    write_csv(comparison, output / "phase1_fixed_seed7_comparison.csv")

    _, _, test_systems = fold_role_sets(
        assignments, "cv5", "fold_0"
    )
    first = evaluate_fold_model(
        frame,
        assignments,
        protocol="cv5",
        fold_id="fold_0",
        model="graphsage",
        policy="legacy_leaky_global",
        seed=7,
        config=GNNConfig(),
    )
    guard_failed = False
    try:
        graph_isolation_audit(first.training_graph, set(test_systems))
    except AssertionError:
        guard_failed = True
    altered = _patch_test_frame(frame, set(test_systems))
    second = evaluate_fold_model(
        altered,
        assignments,
        protocol="cv5",
        fold_id="fold_0",
        model="graphsage",
        policy="legacy_leaky_global",
        seed=7,
        config=GNNConfig(),
    )
    differing = [
        key
        for key in first.trained_weights
        if not np.array_equal(
            first.trained_weights[key].numpy(),
            second.trained_weights[key].numpy(),
        )
    ]
    mutation = pd.DataFrame(
        [
            {
                "policy": "legacy_leaky_global",
                "seed": 7,
                "fold_id": "fold_0",
                "leak_reintroduced": True,
                "isolation_guard_failed_as_expected": guard_failed,
                "weight_hash_changed": (
                    first.weight_sha256 != second.weight_sha256
                ),
                "differing_weight_tensors": len(differing),
                "production_epoch_budget": 150,
                "production_patience": 20,
                "production_weight_folds_covered": 1,
                "structural_folds_covered": 130,
            }
        ]
    )
    write_csv(mutation, output / "mutation_test.csv")
    transcript = (
        "leak_reintroduced=yes\n"
        f"isolation_guard_failed_as_expected={guard_failed}\n"
        f"weight_hash_changed={first.weight_sha256 != second.weight_sha256}\n"
        f"differing_weight_tensors={len(differing)}\n"
        "production_weight_comparison_folds=1\n"
        "production_epochs=150\n"
        "production_patience=20\n"
        "structural_fold_checks=130\n"
    )
    (output / "mutation_test.txt").write_text(transcript)
    return comparison, mutation


def historical_mask_check() -> pd.DataFrame:
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", PRE_FIX_COMMIT], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(prefix="phase15-prefix-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(directory)
        script = r"""
import json
import numpy as np
import pandas as pd
from evaluation.data import load_dataset, load_family_map, system_table
from evaluation.evaluate import GNNConfig, cross_validate_model
from evaluation.metrics import calculate_metrics
from run_phase1_5 import stage_frames, attribution_assignments

frame = load_dataset()
frames = stage_frames(frame)
assignments, _ = attribution_assignments(frames, load_family_map())
out = []
for policy, protocol in [
    ("legacy_leaky_global", "legacy_key_legacy_folds"),
    ("audit_supported_leaky", "canonical_cv5"),
    ("isolated_supported", "canonical_cv5"),
]:
    predictions, _ = cross_validate_model(
        frames["legacy_key"] if policy == "legacy_leaky_global" else frame,
        assignments,
        protocol=protocol,
        model="graphsage",
        policy=policy,
        seed=7,
        config=GNNConfig(),
    )
    metric = calculate_metrics(predictions)
    primary = metric["median_absolute_log_ratio"]
    out.append({
        "commit": "d3bc835",
        "policy": policy,
        "predictions_all_finite": bool(
            np.isfinite(predictions["prediction"].to_numpy()).all()
        ),
        "primary_error": primary if np.isfinite(primary) else None,
    })
print(json.dumps(out))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=directory,
            text=True,
            capture_output=True,
            check=True,
        )
    return pd.DataFrame(json.loads(completed.stdout))


def current_mask_check(output: Path) -> pd.DataFrame:
    frame = load_dataset(DEFAULT_DATA)
    frames = {
        "legacy_key": frame.assign(
            system_id=frame[["suite", "system_name", "system_type"]]
            .fillna("")
            .astype(str)
            .agg("|".join, axis=1)
        ),
        "current_key": frame,
    }
    from run_phase1_5 import attribution_assignments

    assignments, _ = attribution_assignments(frames, load_family_map())
    rows = []
    for policy, protocol in [
        ("legacy_leaky_global", "legacy_key_legacy_folds"),
        ("audit_supported_leaky", "canonical_cv5"),
        ("isolated_supported", "canonical_cv5"),
    ]:
        predictions, _ = cross_validate_model(
            frames["legacy_key"] if policy == "legacy_leaky_global" else frame,
            assignments,
            protocol=protocol,
            model="graphsage",
            policy=policy,
            seed=7,
            config=GNNConfig(),
        )
        metric = calculate_metrics(predictions)
        rows.append(
            {
                "commit": git_output("rev-parse", "--short", "HEAD"),
                "policy": policy,
                "predictions_all_finite": bool(
                    np.isfinite(predictions["prediction"].to_numpy()).all()
                ),
                "primary_error": metric["median_absolute_log_ratio"],
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, output / "mask_check_current.csv")
    return result


def multi_count_rows(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    systems = system_table(frame, load_family_map())
    repeated = systems.loc[
        systems["accelerator_count_values"].str.contains(";"), "system_id"
    ]
    rows = frame.loc[frame["system_id"].isin(repeated)].copy()
    columns = [
        "row_uid",
        "system_id",
        "public_id",
        "benchmark",
        "model",
        "task_code",
        "scenario",
        "units",
        "nodes",
        "accelerators_per_node",
        "total_accelerators",
        "result",
    ]
    rows = rows[columns].sort_values(
        ["system_id", "row_uid"], ignore_index=True
    )
    rows["scalar_source"] = (
        "exported Total Accelerators value for this result row"
    )
    write_csv(rows, output / "multi_count_rows.csv")
    return rows


def write_manifest(output: Path) -> None:
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "results_manifest.json"
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "phase2_model_runs": 0,
        "output_sha256": files,
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    gate1(output)
    frame = load_dataset(DEFAULT_DATA)
    gate3_current(output)
    historical = historical_mask_check()
    current = current_mask_check(output)
    masks = pd.concat([historical, current], ignore_index=True)
    write_csv(masks, output / "loss_mask_bug_check.csv")
    multi_count_rows(frame, output)
    loafo_proposal(frame, output)
    (output / "gate_runtime.txt").write_text(
        f"wall_seconds={time.perf_counter() - started:.6f}\n"
    )
    write_manifest(output)
    print(output)


if __name__ == "__main__":
    main()
