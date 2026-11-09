#!/usr/bin/env python3
"""Generate pre-Phase-2 reconciliation, source, and test evidence."""

from __future__ import annotations

import json
import subprocess
import sys
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
)
from evaluation.evaluate import GNNConfig, cross_validate_model


ROOT = Path(__file__).resolve().parent
PHASE1_BUNDLE = ROOT / "results" / "phase1-final-5c9dfde"
PHASE15_BUNDLE = ROOT / "results" / "phase1_5-final-8fbe960"
SOURCE = (
    Path.home()
    / "Downloads"
    / "mlcommons_results_latest"
    / "Table - Inference_Full Data_data.csv"
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def primary(predictions: pd.DataFrame) -> float:
    return float(
        np.median(
            np.abs(
                np.log(predictions["prediction"].to_numpy(dtype=float))
                - np.log(predictions["result"].to_numpy(dtype=float))
            )
        )
    )


def reconciliation(output: Path) -> pd.DataFrame:
    saved = pd.read_csv(
        PHASE1_BUNDLE / "predictions_by_seed.csv",
        dtype={"public_id": str},
    )
    saved = saved.loc[saved["method"].eq("graphsage")].copy()
    attribution = pd.read_csv(
        PHASE15_BUNDLE / "attribution_predictions_by_seed.csv",
        dtype={"public_id": str},
    )
    attribution = attribution.loc[
        attribution["stage"].eq("baseline") & attribution["seed"].eq(7)
    ].copy()
    assignments = pd.read_csv(PHASE1_BUNDLE / "fold_assignments.csv")
    frame = load_dataset(DEFAULT_DATA)
    from run_phase1_5 import attribution_assignments, stage_frames

    stage_frames_by_key = stage_frames(frame)
    attribution_assignments_table, _ = attribution_assignments(
        stage_frames_by_key, load_family_map()
    )
    attribution_predictions, _ = cross_validate_model(
        stage_frames_by_key["legacy_key"],
        attribution_assignments_table,
        protocol="legacy_key_legacy_folds",
        model="graphsage",
        policy="legacy_leaky_global",
        seed=7,
        config=GNNConfig(),
    )
    rerun_rows = []
    rerun_rows.append(
        {
            "context": "rerun_Gate3_attribution_baseline_seed7_all_rows",
            "fold_scheme": "legacy 96-system random folds",
            "fold_seed": 7,
            "model_seed": 7,
            "prediction_rows": len(attribution_predictions),
            "evaluated_rows": len(attribution_predictions),
            "coverage": len(attribution_predictions) / len(frame),
            "primary_error": primary(attribution_predictions),
        }
    )
    for policy in ["legacy_leaky_global", "isolated_supported"]:
        predictions, _ = cross_validate_model(
            frame,
            assignments,
            protocol="cv5",
            model="graphsage",
            policy=policy,
            seed=7,
            config=GNNConfig(),
        )
        supported = predictions.loc[predictions["is_supported"]]
        rerun_rows.extend(
            [
                {
                    "context": f"rerun_{policy}_seed7_full_predictions",
                    "fold_scheme": "Phase 1 canonical cv5",
                    "fold_seed": 20261107,
                    "model_seed": 7,
                    "prediction_rows": len(predictions),
                    "evaluated_rows": len(predictions),
                    "coverage": len(predictions) / len(frame),
                    "primary_error": primary(predictions),
                },
                {
                    "context": f"rerun_{policy}_seed7_identical_supported",
                    "fold_scheme": "Phase 1 canonical cv5",
                    "fold_seed": 20261107,
                    "model_seed": 7,
                    "prediction_rows": len(predictions),
                    "evaluated_rows": len(supported),
                    "coverage": len(supported) / len(frame),
                    "primary_error": primary(supported),
                },
            ]
        )
    rows = [
        {
            "context": "Gate3_attribution_baseline_seed7_all_rows",
            "fold_scheme": "legacy 96-system random folds",
            "fold_seed": 7,
            "model_seed": 7,
            "prediction_rows": len(attribution),
            "evaluated_rows": len(attribution),
            "coverage": len(attribution) / len(frame),
            "primary_error": primary(attribution),
        }
    ]
    for policy, context_rows in [
        ("legacy_leaky_global", "Phase1_legacy"),
        ("isolated_supported", "Phase1_fixed"),
    ]:
        local = saved.loc[saved["evaluation_policy"].eq(policy)]
        for seed in [7]:
            seed_rows = local.loc[local["seed"].eq(seed)]
            supported = seed_rows.loc[seed_rows["is_supported"]]
            rows.append(
                {
                    "context": f"{context_rows}_seed{seed}_full_predictions",
                    "fold_scheme": "Phase 1 canonical cv5",
                    "fold_seed": 20261107,
                    "model_seed": seed,
                    "prediction_rows": len(seed_rows),
                    "evaluated_rows": len(seed_rows),
                    "coverage": len(seed_rows) / len(frame),
                    "primary_error": primary(seed_rows),
                }
            )
            rows.append(
                {
                    "context": f"{context_rows}_seed{seed}_identical_supported",
                    "fold_scheme": "Phase 1 canonical cv5",
                    "fold_seed": 20261107,
                    "model_seed": seed,
                    "prediction_rows": len(seed_rows),
                    "evaluated_rows": len(supported),
                    "coverage": len(supported) / len(frame),
                    "primary_error": primary(supported),
                }
            )
        seed_values = []
        for _, seed_rows in local.groupby("seed"):
            seed_values.append(primary(seed_rows.loc[seed_rows["is_supported"]]))
        rows.append(
            {
                "context": f"{context_rows}_seeds7to16_identical_supported",
                "fold_scheme": "Phase 1 canonical cv5",
                "fold_seed": 20261107,
                "model_seed": "7-16",
                "prediction_rows": int(
                    local.groupby("seed").size().iloc[0]
                ),
                "evaluated_rows": int(
                    local.loc[local["is_supported"]]
                    .groupby("seed")
                    .size()
                    .iloc[0]
                ),
                "coverage": 450 / len(frame),
                "primary_error": float(np.mean(seed_values)),
                "primary_error_sd": float(np.std(seed_values, ddof=1)),
            }
        )
    rows.extend(rerun_rows)
    result = pd.DataFrame(rows)
    result.to_csv(output / "legacy_fixed_reconciliation.csv", index=False)
    return result


def gb300_source_check(output: Path) -> pd.DataFrame:
    source = pd.read_csv(SOURCE, sep="\t", encoding="utf-16")
    source = source.loc[
        source["Public ID"].astype(str).isin(["6.1-0073", "6.1-0074"])
    ].copy()
    source["nodes_x_accelerators"] = (
        pd.to_numeric(source["# of Nodes"], errors="coerce")
        * pd.to_numeric(source["Accelerators per node"], errors="coerce")
    )
    source["reported_total_accelerators"] = pd.to_numeric(
        source["Total Accelerators1"], errors="coerce"
    )
    source["product_matches_total"] = source[
        "nodes_x_accelerators"
    ].eq(source["reported_total_accelerators"])
    source["system"] = "NVIDIA GB300 NVL72"
    source = source[
        [
            "system",
            "Public ID",
            "Benchmark",
            "Model MLC",
            "Scenario",
            "# of Nodes",
            "Accelerators per node",
            "nodes_x_accelerators",
            "reported_total_accelerators",
            "product_matches_total",
        ]
    ]
    source = source.rename(
        columns={
            "Public ID": "public_id",
            "Model MLC": "model",
            "Scenario": "scenario",
            "# of Nodes": "nodes",
            "Accelerators per node": "accelerators_per_node",
        }
    )
    if not source["product_matches_total"].all():
        raise AssertionError("GB300 source totals do not equal nodes * accelerators")
    source.to_csv(output / "gb300_source_check.csv", index=False)
    return source


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reconciliation(output)
    gb300_source_check(output)
    test_started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-m",
            "unittest",
            "tests.test_leakage.TestFiveFoldWeightBoundary",
            "-v",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    test_elapsed = time.perf_counter() - test_started
    (output / "five_fold_weight_test.txt").write_text(
        completed.stdout
        + completed.stderr
        + f"\nwall_seconds={test_elapsed:.6f}\n"
    )
    if completed.returncode:
        raise AssertionError("Five-fold production weight test failed")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "datacenter_source_sha256": sha256_file(SOURCE),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "loafo_family_mapping_sha256": sha256_file(
            ROOT / "folds" / "loafo_family_mapping_proposed.csv"
        ),
        "five_fold_weight_test_passed": True,
        "five_fold_weight_test_runtime_seconds": test_elapsed,
        "wall_seconds": time.perf_counter() - started,
        "phase2_model_runs": 0,
        "output_sha256": {
            str(path.relative_to(output)): sha256_file(path)
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
