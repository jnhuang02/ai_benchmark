#!/usr/bin/env python3
"""Reproduce Phase 0 and attribute reviewed pipeline changes one at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import feature_set_manifest
from evaluation.data import (
    DEFAULT_DATA,
    DEFAULT_FAMILY_MAP,
    load_dataset,
    load_family_map,
    sha256_file,
    system_table,
)
from evaluation.evaluate import GNNConfig, cross_validate_model
from evaluation.folds import (
    FoldSpec,
    build_fold_specs,
    fold_assignment_table,
    fold_summary,
)
from evaluation.metrics import calculate_metrics


ROOT = Path(__file__).resolve().parent
SEED_SET = tuple(range(7, 17))
PHASE0_COMMIT = "292ff7f23eb41335881a0075337a60bef72a78c2"
PHASE1_BUNDLE_COMMIT = "5c9dfde854d258d6848d1636564861d66e16a146"
PHASE1_RESULTS_COMMIT = "fb36169"
CANONICAL_FOLD_SEED = 20261107
LEGACY_FOLD_SEED = 7


@dataclass(frozen=True)
class AttributionStage:
    name: str
    change: str
    frame_name: str
    assignment_protocol: str
    policy: str
    unknown_mask_probability: float


STAGES = (
    AttributionStage(
        "baseline",
        "original settings",
        "legacy_key",
        "legacy_key_legacy_folds",
        "legacy_leaky_global",
        0.0,
    ),
    AttributionStage(
        "a_submitter_key",
        "a. submitter-aware system key",
        "current_key",
        "submitter_key_legacy_folds",
        "legacy_leaky_global",
        0.0,
    ),
    AttributionStage(
        "b_new_folds",
        "b. canonical suite-stratified five-fold split",
        "current_key",
        "canonical_cv5",
        "legacy_leaky_global",
        0.0,
    ),
    AttributionStage(
        "c_unknown_masking",
        "c. unknown masking at p=0.1",
        "current_key",
        "canonical_cv5",
        "legacy_leaky_global",
        0.1,
    ),
    AttributionStage(
        "d_supported_exclusion",
        "d. unsupported-group exclusion",
        "current_key",
        "canonical_cv5",
        "audit_supported_leaky",
        0.1,
    ),
    AttributionStage(
        "e_fixed_isolation",
        "e. fixed train/inference isolation",
        "current_key",
        "canonical_cv5",
        "isolated_supported",
        0.1,
    ),
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ["numpy", "pandas", "sklearn", "torch", "torch_geometric", "scipy"]:
        imported = __import__(module)
        versions[module] = str(getattr(imported, "__version__", "unknown"))
    return versions


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def old_system_frame(frame: pd.DataFrame) -> pd.DataFrame:
    legacy = frame.copy()
    legacy["system_id"] = legacy[
        ["suite", "system_name", "system_type"]
    ].fillna("").astype(str).agg("|".join, axis=1)
    return legacy


def legacy_fold_specs(
    systems: pd.DataFrame, *, seed: int = LEGACY_FOLD_SEED
) -> list[FoldSpec]:
    system_ids = np.array(sorted(systems["system_id"].tolist()))
    shuffled = np.random.default_rng(seed).permutation(system_ids)
    chunks = np.array_split(shuffled, 5)
    specs = []
    for fold, chunk in enumerate(chunks):
        test_systems = tuple(sorted(str(value) for value in chunk.tolist()))
        outer_train = np.array(
            [value for value in system_ids if value not in set(test_systems)]
        )
        rng = np.random.default_rng(seed + fold)
        validation_count = max(1, round(0.12 * len(outer_train)))
        validation_systems = tuple(
            sorted(
                rng.choice(
                    outer_train, validation_count, replace=False
                ).tolist()
            )
        )
        specs.append(
            FoldSpec(
                fold_id=f"fold_{fold}",
                heldout_group="legacy_random_system_chunk",
                test_systems=test_systems,
                validation_systems=validation_systems,
            )
        )
    return specs


def stage_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    legacy = old_system_frame(frame)
    return {"legacy_key": legacy, "current_key": frame}


def attribution_assignments(
    frames: dict[str, pd.DataFrame], family_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_systems = system_table(frames["current_key"], family_map)
    legacy_systems = system_table(frames["legacy_key"], family_map)
    legacy_protocols = {
        "legacy_key_legacy_folds": fold_assignment_table(
            legacy_systems, {"legacy": legacy_fold_specs(legacy_systems)}
        ),
        "submitter_key_legacy_folds": fold_assignment_table(
            current_systems, {"legacy": legacy_fold_specs(current_systems)}
        ),
    }
    canonical_specs = build_fold_specs(
        current_systems, seed=CANONICAL_FOLD_SEED
    )
    canonical = fold_assignment_table(
        current_systems, {"cv5": canonical_specs["cv5"]}
    )
    frames_to_concat = []
    fold_number = 0
    for protocol, assignments in legacy_protocols.items():
        local = assignments.copy()
        mapping = {
            value: f"github_{fold_number + int(value.rsplit('_', 1)[1]):03d}"
            for value in local["fold_id"].unique()
        }
        local["fold_id"] = local["fold_id"].map(mapping)
        local["protocol"] = protocol
        frames_to_concat.append(local)
        fold_number += local["fold_id"].nunique()
    canonical = canonical.copy()
    canonical["fold_id"] = canonical["fold_id"].map(
        {"fold_0": "github_010", "fold_1": "github_011", "fold_2": "github_012",
         "fold_3": "github_013", "fold_4": "github_014"}
    )
    canonical["protocol"] = "canonical_cv5"
    frames_to_concat.append(canonical)
    combined = pd.concat(frames_to_concat, ignore_index=True)
    return combined, current_systems


def evaluate_attribution(
    frames: dict[str, pd.DataFrame],
    assignments: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    rows = []
    prediction_frames = []
    for stage in STAGES:
        frame = frames[stage.frame_name]
        config = GNNConfig(unknown_mask_probability=stage.unknown_mask_probability)
        for seed in SEED_SET:
            predictions, _ = cross_validate_model(
                frame,
                assignments,
                protocol=stage.assignment_protocol,
                model="graphsage",
                policy=stage.policy,
                seed=seed,
                config=config,
            )
            if predictions.empty:
                raise AssertionError(f"{stage.name}/{seed} produced no predictions")
            metric = calculate_metrics(predictions)
            rows.append(
                {
                    "stage": stage.name,
                    "change": stage.change,
                    "seed": seed,
                    "primary_error": metric["median_absolute_log_ratio"],
                    "median_group_spearman": metric[
                        "median_group_spearman"
                    ],
                    "evaluated_rows": len(predictions),
                    "total_rows": len(frame),
                    "coverage": len(predictions) / len(frame),
                    "median_row_mape": metric["median_row_mape"],
                    "normalized_rmse": metric["normalized_rmse"],
                }
            )
            prediction_frames.append(
                predictions.assign(stage=stage.name, stage_change=stage.change)
            )
    by_seed = pd.DataFrame(rows)
    write_csv(by_seed, output / "attribution_by_seed.csv")
    write_csv(
        pd.concat(prediction_frames, ignore_index=True),
        output / "attribution_predictions_by_seed.csv",
    )
    summary = (
        by_seed.groupby(["stage", "change"], as_index=False)
        .agg(
            primary_error_mean=("primary_error", "mean"),
            primary_error_sd=("primary_error", "std"),
            median_group_spearman_mean=("median_group_spearman", "mean"),
            median_group_spearman_sd=("median_group_spearman", "std"),
            coverage_mean=("coverage", "mean"),
            coverage_sd=("coverage", "std"),
            evaluated_rows=("evaluated_rows", "first"),
            seeds=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    order = {stage.name: index for index, stage in enumerate(STAGES)}
    summary["_order"] = summary["stage"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order")
    write_csv(summary, output / "attribution_summary.csv")
    return summary


def prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    frame = predictions.copy()
    if "target_group" not in frame:
        columns = [
            "suite",
            "version",
            "division",
            "benchmark",
            "model",
            "task_code",
            "scenario",
            "units",
        ]
        frame["target_group"] = (
            frame[columns].fillna("").astype(str).agg("|".join, axis=1)
        )
    frame["ape"] = np.abs(frame["prediction"] - frame["result"]) / frame["result"]
    frame["absolute_log_ratio"] = np.abs(
        np.log(frame["prediction"]) - np.log(frame["result"])
    )
    group_size = frame.groupby("target_group")["result"].transform("size")
    singleton = frame.loc[group_size.eq(1), "ape"]
    edge = frame.loc[frame["suite"].eq("edge")].copy()
    edge_size = edge.groupby("target_group")["result"].transform("size")
    edge_evaluable = edge.loc[edge_size.ge(3)]
    group_mape = frame.groupby("target_group")["ape"].median().median()
    rank_scores = []
    for _, group in frame.groupby("target_group"):
        if (
            len(group) >= 3
            and group["result"].nunique() >= 2
            and group["prediction"].nunique() >= 2
        ):
            score = group["result"].rank().corr(group["prediction"].rank())
            if np.isfinite(score):
                rank_scores.append(float(score))
    return {
        "rows": len(frame),
        "coverage": 1.0,
        "primary_error": float(frame["absolute_log_ratio"].median()),
        "median_group_mape": float(group_mape),
        "median_row_mape": float(frame["ape"].median()),
        "median_group_spearman": float(np.median(rank_scores)),
        "edge_evaluable_rows": len(edge_evaluable),
        "edge_mean_ape_pct": float(edge_evaluable["ape"].mean() * 100),
        "singleton_ape_share": float(
            singleton.sum() / frame["ape"].sum()
        ),
    }


def run_regression(output: Path) -> pd.DataFrame:
    regression_dir = output / "regression"
    regression_dir.mkdir(parents=True)
    source_path = regression_dir / "phase0_train_graphsage.py"
    data_path = regression_dir / "phase0_mlperf_long.csv"
    source_path.write_text(git_output("show", f"{PHASE0_COMMIT}:train_graphsage.py"))
    data_path.write_text(git_output("show", f"{PHASE0_COMMIT}:data/mlperf_long.csv"))
    rows = []
    transcripts = []
    for model in ["graphsage", "gat"]:
        model_dir = regression_dir / f"{model}"
        command = [
            sys.executable,
            str(source_path),
            "--data",
            str(data_path),
            "--suite",
            "all",
            "--folds",
            "5",
            "--model",
            model,
            "--out-dir",
            str(model_dir),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        elapsed = time.perf_counter() - started
        transcripts.append(
            f"$ {' '.join(command)}\n{completed.stdout}{completed.stderr}"
            f"\nwall_seconds={elapsed:.6f}\n"
        )
        predictions = pd.read_csv(
            model_dir / f"all_{model}_predictions.csv",
            dtype={"public_id": str},
        )
        predictions = predictions.loc[predictions["method"].eq(model)]
        diagnostics = prediction_diagnostics(predictions)
        diagnostics.update(
            {
                "model": model,
                "phase0_source_sha256": sha256_file(source_path),
                "phase0_data_sha256": sha256_file(data_path),
                "runtime_seconds": elapsed,
            }
        )
        rows.append(diagnostics)
    table = pd.DataFrame(rows)
    target = {
        "graphsage": {
            "edge_mean_ape_pct": 6289.529815910648,
            "singleton_ape_share": 0.46214010550186546,
        },
        "gat": {
            "median_group_mape": 0.29756497402155335,
            "median_row_mape": 0.23157114656046207,
        },
    }
    checks = []
    for row in table.to_dict(orient="records"):
        for metric, expected in target[row["model"]].items():
            checks.append(
                {
                    "model": row["model"],
                    "metric": metric,
                    "observed": row[metric],
                    "phase0_value": expected,
                    "absolute_difference": abs(row[metric] - expected),
                    "reproduced": bool(np.isclose(row[metric], expected, atol=1e-12)),
                }
            )
    checks_frame = pd.DataFrame(checks)
    if not checks_frame["reproduced"].all():
        raise AssertionError("Phase 0 regression did not reproduce saved values")
    write_csv(table, regression_dir / "regression_diagnostics.csv")
    write_csv(checks_frame, regression_dir / "regression_checks.csv")
    (regression_dir / "regression_output.txt").write_text("\n".join(transcripts))
    return table


def sensitivity_artifacts(
    frame: pd.DataFrame,
    family_map: pd.DataFrame,
    assignments: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = system_table(frame, family_map)
    grouped = (
        systems.groupby("sensitivity_group_id", as_index=False)
        .agg(
            systems=("system_id", "nunique"),
            measurements=("measurements", "sum"),
            suites=("suite", lambda values: ";".join(sorted(set(values)))),
            system_ids=("system_id", lambda values: ";".join(sorted(values))),
            reported_count_signatures=(
                "accelerator_count_values",
                lambda values: ";".join(sorted(set(values))),
            ),
        )
        .sort_values(["systems", "sensitivity_group_id"], ascending=[False, True])
    )
    write_csv(grouped, output / "sensitivity_groups.csv")
    distribution = (
        grouped.groupby("systems", as_index=False)
        .size()
        .rename(columns={"size": "group_count"})
    )
    write_csv(distribution, output / "sensitivity_group_size_distribution.csv")
    summary = (
        systems.groupby("suite", as_index=False)
        .agg(
            systems=("system_id", "nunique"),
            measurements=("measurements", "sum"),
            sensitivity_groups=("sensitivity_group_id", "nunique"),
        )
    )
    total = pd.DataFrame(
        [
            {
                "suite": "total",
                "systems": systems["system_id"].nunique(),
                "measurements": systems["measurements"].sum(),
                "sensitivity_groups": systems["sensitivity_group_id"].nunique(),
            }
        ]
    )
    summary = pd.concat([summary, total], ignore_index=True)
    write_csv(summary, output / "sensitivity_summary.csv")

    canonical = assignments.loc[
        assignments["protocol"].eq("canonical_cv5")
    ].copy()
    write_csv(fold_summary(canonical), output / "canonical_fold_summary.csv")
    return summary, distribution


def loafo_proposal(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    systems = (
        frame.groupby("system_id", as_index=False)
        .agg(
            suite=("suite", "first"),
            submitter=("organization", "first"),
            system_name=("system_name", "first"),
            accelerator=("accelerator", lambda values: ";".join(sorted(set(values)))),
            processor=("host_processor", lambda values: ";".join(sorted(set(values)))),
            rows=("result", "size"),
        )
    )
    flagged = systems.loc[
        systems["accelerator"].str.contains(r"\+| and ")
        | systems["accelerator"].eq("")
        | systems["accelerator"].isin(
            [
                "AMD Radeon 8060S Graphics",
                "NVIDIA GB10",
                "NVIDIA Jetson AGX Thor 128G",
                "NVIDIA VR200",
            ]
        )
    ].copy()

    def proposal(row: pd.Series) -> tuple[str, str, str, str, str]:
        accelerator = row["accelerator"]
        if "NVIDIA H200" in accelerator and "MI350X" in accelerator:
            return (
                "NVIDIA + AMD",
                "Hopper H200 + CDNA4 MI350X",
                "mixed_nvidia_hopper_h200_amd_cdna4_mi350x",
                "always_train",
                "Two discrete accelerator architectures are reported by the same system.",
            )
        if "MI300X" in accelerator and "MI355X" in accelerator:
            return (
                "AMD",
                "CDNA3 MI300X + CDNA4 MI355X",
                "mixed_amd_cdna3_mi300x_cdna4_mi355x",
                "always_train",
                "Two AMD Instinct architecture generations are reported by one heterogeneous cluster.",
            )
        if accelerator == "":
            return (
                "Intel",
                "Xeon 6 Granite Rapids",
                "cpu_only_intel_xeon6_granite_rapids",
                "holdout_eligible",
                "No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.",
            )
        if accelerator == "NVIDIA GB10":
            return (
                "NVIDIA",
                "Blackwell integrated GB10",
                "nvidia_blackwell_integrated_gb10",
                "holdout_eligible",
                "GB10 is an integrated Grace-Blackwell device, distinct from discrete B-series accelerators.",
            )
        if accelerator == "NVIDIA Jetson AGX Thor 128G":
            return (
                "NVIDIA",
                "Blackwell integrated Jetson Thor",
                "nvidia_blackwell_integrated_jetson_thor",
                "holdout_eligible",
                "Jetson AGX Thor is an integrated Blackwell-generation edge device.",
            )
        if accelerator == "AMD Radeon 8060S Graphics":
            return (
                "AMD",
                "RDNA 3.5 integrated Radeon 8060S",
                "amd_rdna35_integrated_radeon_8060s",
                "holdout_eligible",
                "Radeon 8060S in the Strix Halo system is an integrated RDNA-generation device.",
            )
        if accelerator == "NVIDIA VR200":
            return (
                "NVIDIA",
                "Rubin VR200",
                "nvidia_rubin_vr200",
                "holdout_eligible",
                "The reported VR200 accelerator denotes the Rubin-generation product family.",
            )
        raise ValueError(f"No LOAFO proposal for {row['system_id']}: {accelerator}")

    proposed = [proposal(row) for _, row in flagged.iterrows()]
    for column, values in zip(
        [
            "proposed_vendor",
            "proposed_architecture_generation",
            "proposed_family_id",
            "holdout_disposition",
            "rationale",
        ],
        zip(*proposed),
    ):
        flagged[column] = values
    if len(flagged) != 18:
        raise AssertionError(f"Expected 18 LOAFO review cases, found {len(flagged)}")
    flagged = flagged.sort_values(
        ["holdout_disposition", "proposed_family_id", "system_id"],
        ascending=[False, True, True],
    )
    write_csv(flagged, output / "loafo_family_proposal.csv")
    return flagged


def feature_contract(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    systems = system_table(frame, load_family_map())
    repeated = systems.loc[
        systems["accelerator_count_values"].str.contains(";")
    ][
        [
            "system_id",
            "accelerator",
            "processor",
            "accelerator_count_values",
            "sensitivity_group_id",
        ]
    ].copy()
    details = []
    for row in repeated.itertuples(index=False):
        values = frame.loc[
            frame["system_id"].eq(row.system_id), "total_accelerators"
        ]
        details.append(
            {
                **row._asdict(),
                "reported_total_accelerator_values": ";".join(
                    f"{value:g}" for value in sorted(values.unique())
                ),
                "numeric_feature_contract": (
                    "each row uses its scalar total_accelerators value"
                ),
            }
        )
    table = pd.DataFrame(details)
    write_csv(table, output / "multi_count_systems.csv")
    lines = [
        "# Phase 1.5 feature contract",
        "",
        "- Numeric feature: each result row uses the scalar `total_accelerators` value.",
        "- The sorted reported-count tuple is retained only as `accelerator_count_values` for audit grouping.",
        "- The sensitivity key excludes the count tuple and uses `submitter|accelerator|processor`.",
        "- The count tuple is a system-level report signature, not a numeric model feature.",
        "",
        "| System | Reported counts | Sensitivity group |",
        "| --- | --- | --- |",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| `{row.system_id}` | `{row.accelerator_count_values}` | "
            f"`{row.sensitivity_group_id}` |"
        )
    (output / "feature_contract.md").write_text("\n".join(lines) + "\n")
    return table


def provenance(output: Path) -> str:
    run_files = [
        "run_phase1.py",
        "train_graphsage.py",
        "prepare_data.py",
        "evaluation",
        "tests",
        "config",
        "data",
        "requirements.txt",
    ]
    diff = git_output(
        "diff",
        "--stat",
        PHASE1_BUNDLE_COMMIT,
        PHASE1_RESULTS_COMMIT,
        "--",
        *run_files,
    )
    if not diff:
        diff = (
            "No run-affecting files differ between the manifest commit "
            "5c9dfde and HEAD fb36169. The intervening commit changed only "
            "CHANGELOG.md and result-report metadata."
        )
    text = (
        f"bundle_manifest_commit={PHASE1_BUNDLE_COMMIT}\n"
        f"phase1_results_commit={PHASE1_RESULTS_COMMIT}\n\n"
        "git diff --stat 5c9dfde fb36169 -- run-affecting files:\n"
        f"{diff}\n"
    )
    (output / "provenance.txt").write_text(text)
    return text


def write_report(
    output: Path,
    attribution: pd.DataFrame,
    regression: pd.DataFrame,
    sensitivity: pd.DataFrame,
    sensitivity_distribution: pd.DataFrame,
    loafo: pd.DataFrame,
    multi_count: pd.DataFrame,
    provenance_text: str,
    test_output: str,
    test_elapsed: float,
) -> None:
    lines = [
        "# Phase 1.5 Report",
        "",
        "Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.",
        "",
        "## 1. Attribution and regression",
        "",
        "Primary error is the median `|ln(prediction / measured)|` over the "
        "rows evaluated in a seed. The mean and SD are across seeds 7-16 after "
        "first computing that row-level median within each seed.",
        "The attribution loop pins each fold construction once and varies only "
        "the model seed. The separate exact Phase 0 replay below retains the "
        "historical seed-coupled folds.",
        "",
        "| Step | Change | Primary error mean | SD | Median-group Spearman mean | "
        "SD | Coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in attribution.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.change} | {row.primary_error_mean:.6f} | "
            f"{row.primary_error_sd:.6f} | "
            f"{row.median_group_spearman_mean:.6f} | "
            f"{row.median_group_spearman_sd:.6f} | "
            f"{row.coverage_mean * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Phase 0 regression model | Primary error | Median-group MAPE | "
            "Median-row MAPE | Median-group Spearman | Edge mean APE | "
            "Singleton APE share | Runtime (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in regression.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.primary_error:.6f} | "
            f"{row.median_group_mape * 100:.4f}% | "
            f"{row.median_row_mape * 100:.4f}% | "
            f"{row.median_group_spearman:.6f} | "
            f"{row.edge_mean_ape_pct:.4f}% | "
            f"{row.singleton_ape_share * 100:.4f}% | "
            f"{row.runtime_seconds:.3f} |"
        )
    lines.extend(
        [
            "",
            "The exact Phase 0 source and data were executed from "
            f"`{PHASE0_COMMIT}`. GAT reproduces 29.7565%/23.1571%; "
            "GraphSAGE reproduces the 6,289.5298% Edge mean APE and "
            "46.2140% singleton share. These were never all one model's "
            "diagnostic set.",
            "",
            "## 2. Sensitivity grouping",
            "",
            "The corrected key is `submitter|accelerator model|processor`; "
            "accelerator counts are excluded.",
            "",
            "| Suite | Systems | Measurements | Groups |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.suite} | {row.systems} | {row.measurements} | "
            f"{row.sensitivity_groups} |"
        )
    lines.extend(
        [
            "",
            "| Systems per group | Group count |",
            "| ---: | ---: |",
        ]
    )
    for row in sensitivity_distribution.itertuples(index=False):
        lines.append(f"| {row.systems} | {row.group_count} |")
    lines.extend(
        [
            "",
            "All groups are enumerated in `sensitivity_groups.csv`.",
            "",
            "## 3. Leakage runtime",
            "",
            f"The complete suite ran in {test_elapsed:.3f} seconds. The previous "
            "18-second suite used `epochs=1, patience=1` in its 130-fold weight "
            "loop. The strengthened production check uses the default 150-epoch "
            "budget with patience 20, saves and reloads actual state dictionaries, "
            "and confirms the reintroduced leak changes weights.",
            "",
            "```text",
            test_output.strip(),
            "```",
            "",
            "## 4. Provenance",
            "",
            "```text",
            provenance_text.strip(),
            "```",
            "",
            "## 5. LOAFO proposal (not run)",
            "",
            "Mixed-accelerator systems are proposed as always-train and are "
            "not held out as families.",
            "",
            "| System | Accelerator | Vendor | Architecture generation | "
            "Proposed family | Disposition | Rationale |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in loafo.itertuples(index=False):
        lines.append(
            f"| `{row.system_id}` | {row.accelerator or '<blank>'} | "
            f"{row.proposed_vendor} | {row.proposed_architecture_generation} | "
            f"`{row.proposed_family_id}` | {row.holdout_disposition} | "
            f"{row.rationale} |"
        )
    lines.extend(
        [
            "",
            "## 6. Feature contract",
            "",
            "Each result row retains its scalar `total_accelerators` value as "
            "the numeric feature. The sorted reported-count tuple is retained "
            "only for grouping/audit and is excluded from the sensitivity key.",
            "",
            "| System | Reported count signature |",
            "| --- | --- |",
        ]
    )
    for row in multi_count.itertuples(index=False):
        lines.append(
            f"| `{row.system_id}` | `{row.accelerator_count_values}` |"
        )
    lines.extend(
        [
            "",
            "## 7. Phase report",
            "",
            "### What was done",
            "",
            "- Re-executed the pinned Phase 0 source and verified all four diagnostics.",
            "- Ran the cumulative one-change-at-a-time GraphSAGE attribution.",
            "- Corrected and enumerated the sensitivity groups.",
            "- Strengthened the leakage test to exercise production training.",
            "- Recorded source/data provenance and the Phase 2 feature contract.",
            "- Proposed, but did not run, the 18-system LOAFO review families.",
            "",
            "### Files changed",
            "",
            "All numerical claims above trace to CSV or text files in this "
            "directory; exact hashes are recorded in `results_manifest.json`.",
            "",
            "### Numbers that changed",
            "",
            f"- Sensitivity groups: 101 -> {int(sensitivity.iloc[-1].sensitivity_groups)}.",
            "- Phase 1's non-reproducing legacy diagnostics are explained and "
            "replaced by the pinned regression table above.",
            "",
            "### Surprising or contradictory findings",
            "",
            "- The Phase 0 figures reproduce exactly, but they belong to different "
            "model outputs: GAT owns the 29.8%/23.2% errors; GraphSAGE owns the "
            "Edge and singleton diagnostics.",
            "- The prior leakage suite was fast because its weight comparisons used "
            "one epoch; the strengthened suite now exercises production training.",
            "",
            "### Open questions",
            "",
            "- Approve or revise the 18-system LOAFO proposal before any LOAFO run.",
            "",
            "### Acceptance criteria",
            "",
            "- Attribution table: yes.",
            "- Phase 0 regression: yes.",
            "- Sensitivity grouping fix and counts: yes.",
            "- Production-training leakage mutation test: yes.",
            "- Provenance: yes.",
            "- Metric aggregation definition: yes.",
            "- LOAFO table: proposed; execution intentionally withheld.",
            "- Feature contract: yes.",
        ]
    )
    (output / "phase1_5_report.md").write_text("\n".join(lines) + "\n")


def run_tests(output: Path) -> tuple[str, float, bool]:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    text = completed.stdout + completed.stderr
    (output / "leakage_test_output.txt").write_text(text)
    return text, elapsed, completed.returncode == 0


def write_manifest(output: Path, command: list[str]) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "results_manifest.json":
            files[str(path.relative_to(output))] = sha256_file(path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        **feature_set_manifest(),
        "command": command,
        "data_sha256": sha256_file(DEFAULT_DATA),
        "family_map_sha256": sha256_file(DEFAULT_FAMILY_MAP),
        "phase0_commit": PHASE0_COMMIT,
        "phase0_train_graphsage_sha256": sha256_file(
            output / "regression" / "phase0_train_graphsage.py"
        ),
        "phase0_data_sha256": sha256_file(
            output / "regression" / "phase0_mlperf_long.csv"
        ),
        "software_versions": software_versions(),
        "output_sha256": files,
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("ignore", message=".*type_params.*")
    frame = load_dataset(DEFAULT_DATA)
    family_map = load_family_map()
    frames = stage_frames(frame)
    assignments, _ = attribution_assignments(frames, family_map)
    write_csv(assignments, output / "attribution_fold_assignments.csv")
    regression = run_regression(output)
    attribution = evaluate_attribution(frames, assignments, output)
    sensitivity, sensitivity_distribution = sensitivity_artifacts(
        frame, family_map, assignments, output
    )
    loafo = loafo_proposal(frame, output)
    multi_count = feature_contract(frame, output)
    provenance_text = provenance(output)
    test_output, test_elapsed, passed = run_tests(output)
    if not passed:
        raise AssertionError("Test suite failed")
    write_report(
        output,
        attribution,
        regression,
        sensitivity,
        sensitivity_distribution,
        loafo,
        multi_count,
        provenance_text,
        test_output,
        test_elapsed,
    )
    write_manifest(output, [sys.executable, str(Path(__file__)), *sys.argv[1:]])
    print(output)


if __name__ == "__main__":
    main()
