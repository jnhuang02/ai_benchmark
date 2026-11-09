#!/usr/bin/env python3
"""Generate the complete Phase 2 report from saved result tables."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
FINAL = ROOT / "results" / "phase2-final-f51e57a"
CV = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
LOSO = ROOT / "results" / "phase2-loso-final-a689ba5"
SENS = ROOT / "results" / "phase2-sensitivity-final-a689ba5"
DIAG = ROOT / "results" / "phase2-cv5-diagnostics-77c4752"
PRE = ROOT / "results" / "phase2-prerun-1122699"


def fmt(value: float, digits: int = 6) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> None:
    decision = json.loads(
        (FINAL / "co_primary_decision.json").read_text()
    )
    cv = pd.read_csv(DIAG / "main_table_corrected.csv")
    loso = pd.read_csv(LOSO / "main_table.csv")
    sensitivity = pd.read_csv(SENS / "main_table.csv")
    cv_pairs = pd.read_csv(FINAL / "cv5_paired_differences.csv")
    primary = pd.read_csv(FINAL / "primary_decision.csv")
    secondary = pd.read_csv(FINAL / "secondary_holm.csv")
    gat = pd.read_csv(DIAG / "gat_training_summary.csv")
    gat_compare = pd.read_csv(DIAG / "gat_phase1_comparison.csv")
    regress = pd.read_csv(DIAG / "graphsage_regression_comparison.csv")
    spearman = pd.read_csv(DIAG / "spearman_group_summary.csv")
    residuals = pd.read_csv(DIAG / "top_abs_z_residuals.csv")
    partitions = pd.read_csv(DIAG / "partition_primary_errors.csv")
    drivers = pd.read_csv(DIAG / "partition_system_drivers.csv")
    mixed_status = pd.read_csv(DIAG / "mixed_model_fit_status.csv")
    mixed_groups = pd.read_csv(DIAG / "mixed_vs_ridge_by_group.csv")
    mixed_spec = json.loads(
        (DIAG / "mixed_model_specification.json").read_text()
    )
    coverage = pd.read_csv(DIAG / "coverage_reconciliation.csv")
    cv_manifest = json.loads(
        (CV / "results_manifest.json").read_text()
    )

    cv_by_method = cv.set_index("method")
    loso_by_method = loso.set_index("method")
    sens_by_method = sensitivity.set_index("method")
    all_methods = sorted(
        set(cv["method"]) | set(loso["method"]) | set(sensitivity["method"])
    )
    lines = [
        "# Phase 2 Complete Report",
        "",
        "Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.",
        "",
        "Canonical CV5: `mlperf_gnn/results/phase2-cv5-final-aeeb4b4/`  ",
        "LOSO: `mlperf_gnn/results/phase2-loso-final-a689ba5/`  ",
        "Sensitivity: `mlperf_gnn/results/phase2-sensitivity-final-a689ba5/`  ",
        "Combined inference: `mlperf_gnn/results/phase2-final-f51e57a/`  ",
        "Diagnostics: `mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/`",
        "",
        "## 1. What was done",
        "",
        "- Ran all requested CV5 diagnostics without changing CV5 predictions.",
        "- Repeated GAT training for all 25 partition-fold cells and all ten seeds.",
        "- Compared GAT with the Phase 1 numeric-feature path on the same fold.",
        "- Reran GraphSAGE on partition 20261107 with both Phase 2 and Phase 1",
        "  numeric features.",
        "- Reported group-median Spearman as unavailable and enumerated the",
        "  qualifying group count for every method.",
        "- Listed the ten largest normalized residuals for RF, MLP, and",
        "  GraphSAGE and identified partition/system drivers.",
        "- Reran mixed-model fits to capture convergence and compared it with",
        "  Ridge by target group.",
        "- Ran LOSO with neural seeds 7, 11, and 16.",
        "- Ran sensitivity with neural seeds 7-16.",
        "- Applied Holm correction across the 8 CV/LOSO secondary tests.",
        "- Did not run LOAFO.",
        "",
        "## 2. Files changed",
        "",
        "Key outputs:",
        "",
        "- [GAT training summary](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/gat_training_summary.csv)",
        "- [GAT by fold and seed](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/gat_training_by_fold_seed.csv)",
        "- [GraphSAGE regression](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/graphsage_regression_comparison.csv)",
        "- [Spearman counts](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/spearman_group_summary.csv)",
        "- [Top residuals](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/top_abs_z_residuals.csv)",
        "- [Partition errors](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/partition_primary_errors.csv)",
        "- [Mixed specification](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/mixed_model_specification.json)",
        "- [LOSO table](../mlperf_gnn/results/phase2-loso-final-a689ba5/main_table.csv)",
        "- [Sensitivity table](../mlperf_gnn/results/phase2-sensitivity-final-a689ba5/main_table.csv)",
        "- [Co-primary decision](../mlperf_gnn/results/phase2-final-f51e57a/co_primary_decision.json)",
        "- [Holm results](../mlperf_gnn/results/phase2-final-f51e57a/secondary_holm.csv)",
        "",
        "The pre-existing thesis source and bibliography were not edited.",
        "",
        "## 3. Numbers that changed",
        "",
        "### GAT diagnostic",
        "",
        "GAT does not collapse to a constant. Across all 25 partition-fold cells:",
        "",
        "- Normalized prediction variance ranged from `0.008526` to `0.126357`,",
        "  median `0.037296`.",
        "- Unique predictions per fold-seed ranged from 69 to 112.",
        "- Mean best epoch ranged by fold from 1 to 67;",
        "  mean epochs trained ranged from 21 to 87.",
        "- Mean final training Huber loss ranged from `0.282157` to `0.360742`.",
        "- Mean final validation loss ranged from `0.254338` to `1.506562`.",
        "",
        "The poor GAT rank correlation is therefore a ranking/outlier failure,",
        "not a near-constant-output failure.",
        "",
        "| Partition | Fold | Raw prediction variance | Normalized variance | Best epoch range | Epochs range | Final train loss | Final validation loss |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in gat.itertuples(index=False):
        lines.append(
            f"| {row.protocol.replace('cv5_partition_', '')} | "
            f"{row.fold_id.split('_')[-1]} | "
            f"{row.prediction_variance_mean:.6f} | "
            f"{row.normalized_prediction_variance_mean:.6f} | "
            f"{int(row.best_epoch_min)}-{int(row.best_epoch_max)} | "
            f"{int(row.epochs_run_min)}-{int(row.epochs_run_max)} | "
            f"{row.final_train_loss_mean:.6f} | "
            f"{row.final_validation_loss_mean:.6f} |"
        )
    lines.extend(
        [
            "",
            "Same-fold seed-7 comparison on `p0_fold_0`:",
            "",
            "| Run | Primary error | Raw variance | Normalized variance | Unique predictions | Best/epochs |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in gat_compare.itertuples(index=False):
        lines.append(
            f"| {row.run} | {row.primary_error:.6f} | "
            f"{row.prediction_variance:.6f} | "
            f"{row.normalized_prediction_variance_mean:.6f} | "
            f"{int(row.prediction_unique)} | "
            f"{int(row.best_epoch)}/{int(row.epochs_run)} |"
        )
    lines.extend(
        [
            "",
            "### GraphSAGE regression",
            "",
        ]
    )
    for row in regress.itertuples(index=False):
        lines.append(f"- {row.quantity}: `{row.value:.9f}`")
    lines.extend(
        [
            "",
            "The Phase 2 CPU-only indicator alone changes the partition-0 seed",
            "mean by `0.011535`; without it, the Phase 2 code reproduces the",
            "Phase 1.5 value `0.4623139872076701` exactly. Pooling is not the",
            "gap: both use the same five OOF folds and 450 rows per seed.",
            "",
            "### Spearman availability and group counts",
            "",
            "Group-median Spearman is reported as `NA`. Its predictions have no",
            "within-group rank variation. No other method was globally constant.",
            "The qualifying group count is 18-20 across the five partitions.",
            "",
            "| Method | Groups min | Groups max | Mean group Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in spearman.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {int(row.qualifying_groups_min)} | "
            f"{int(row.qualifying_groups_max)} | "
            f"{fmt(row.median_group_spearman_partition_mean)} |"
        )
    lines.extend(
        [
            "",
            "### Largest normalized residuals",
            "",
            "Top-ten SSE shares: RF `0.851005`, MLP `0.871018`, GraphSAGE",
            "`0.779098`. Two recurring rows dominate all three models:",
            "Llama 2 70B Interactive on the Dell_AMD XE7745 MI350P system and",
            "DeepSeek-R1 Interactive on the NVIDIA Vera Rubin NVL72 system.",
            "These extreme standardized predictions drive normalized RMSE above 1.5.",
            "",
            "| Method | Rank | Partition/fold | System | Target group | |z| |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for row in residuals.loc[
        residuals["rank"].astype(str).ne("top10_sse_share")
    ].itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.rank} | "
            f"{row.protocol}/{row.fold_id} | `{row.system_id}` | "
            f"`{row.target_group}` | {row.abs_z:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Partition sensitivity",
            "",
            "| Method | P0 | P1 | P2 | P3 | P4 | Partition SD |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in ["random_forest", "mlp", "graphsage", "gcn", "gat"]:
        local = partitions.loc[partitions["method"].eq(method)].set_index(
            "protocol"
        )
        values = [
            local.loc[f"cv5_partition_{index}", "primary_error"]
            for index in range(5)
        ]
        lines.append(
            f"| `{method}` | "
            + " | ".join(f"{value:.6f}" for value in values)
            + f" | {pd.Series(values).std(ddof=1):.6f} |"
        )
    lines.extend(
        [
            "",
            "GraphSAGE is highest on partition 3 (`0.489107`) and partition 0",
            "(`0.445144`), and lowest on partition 1 (`0.268147`). The largest",
            "cross-partition system ranges are shown below.",
            "",
            "| Method | System | Error range | Mean error |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for method in ["graphsage", "gcn", "mlp", "random_forest", "gat"]:
        for row in drivers.loc[drivers["method"].eq(method)].head(5).itertuples(
            index=False
        ):
            lines.append(
                f"| `{method}` | `{row.system_id}` | "
                f"{row.partition_range:.6f} | "
                f"{row.partition_mean_error:.6f} |"
            )
    lines.extend(
        [
            "",
            "Full top-10 driver tables are in `partition_system_drivers.csv`.",
            "",
            "### Mixed model",
            "",
            f"Formula: `{mixed_spec['formula']}`.",
            "Random intercepts: accelerator family, submitter, and system.",
            "Held-out system random effect is fixed at zero; predictions use fixed",
            "effects only, and held-out systems do not update variance components.",
            "",
            f"Convergence: {int(mixed_status['converged'].sum())}/"
            f"{len(mixed_status)} CV5 fold fits converged. Optimizers: "
            + ", ".join(
                f"`{key}` {value}"
                for key, value in mixed_status["optimizer"]
                .value_counts()
                .items()
            )
            + ". LOSO subsequently required the logged Powell fallback on one fold.",
            "",
            "Per-group mean absolute log error, mixed minus Ridge:",
            "",
            "| Target group | Mixed | Ridge | Difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in mixed_groups.itertuples(index=False):
        lines.append(
            f"| `{row.target_group}` | {row.hierarchical_mixed:.6f} | "
            f"{row.ridge:.6f} | {row.mixed_minus_ridge:+.6f} |"
        )
    lines.extend(
        [
            "",
            "### Protocol and coverage",
            "",
            "- Protocol SHA-256:",
            f"  `{cv_manifest['protocol_sha256']}`",
            "- Recorded as unchanged in the CV5 manifest:",
            f"  `{cv_manifest['protocol_hash_unchanged_since_cv5_run']}`",
            "- The CV5 average coverage is `94.495798%`, not `450/476 =",
            "  94.537815%`, because support differs by partition.",
            "",
            "| Partition | Evaluated rows | Coverage |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.rows:g} | {row.coverage * 100:.6f}% |"
        )
    lines.extend(
        [
            "",
            "The five partition coverages average to 449.8 rows and 94.495798%.",
            "",
            "### Primary co-primary decision",
            "",
            "| Protocol | Difference | 95% CI | Label |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.mean_difference:.6f} | "
            f"[{row.difference_95ci_low:.6f}, "
            f"{row.difference_95ci_high:.6f}] | {row.status} |"
        )
    lines.extend(
        [
            "",
            f"Combined primary label: **{decision['combined_label']}**. Both",
            "co-primary protocols return `inconclusive`; sensitivity is labeled",
            f"`{decision['sensitivity_label_not_co_primary']}` but is not co-primary.",
            "",
            "### Secondary comparisons after Holm",
            "",
            "| Protocol | Comparison | Difference [95% CI] | Raw label | Holm p | Significant |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in secondary.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | `{row.left} - {row.right}` | "
            f"{row.mean_difference:.6f} "
            f"[{row.difference_95ci_low:.6f}, "
            f"{row.difference_95ci_high:.6f}] | {row.status} | "
            f"{row.holm_adjusted_p_value:.6f} | "
            f"{row.significant_after_holm} |"
        )
    lines.extend(
        [
            "",
            "### Protocol tables",
            "",
            "| Method | CV5 primary | LOSO primary | Sensitivity primary |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in all_methods:
        lines.append(
            f"| `{method}` | "
            f"{fmt(cv_by_method.loc[method, 'primary_error_mean'])} | "
            f"{fmt(loso_by_method.loc[method, 'primary_error'])} | "
            f"{fmt(sens_by_method.loc[method, 'primary_error'])} |"
        )
    lines.extend(
        [
            "",
            "Full metric columns and intervals are in each protocol's",
            "`main_table.csv`.",
            "",
            "## 4. Anything surprising or contradicting the current narrative",
            "",
            "- GAT does not collapse numerically; it produces 69-112 distinct",
            "  predictions per fold but has near-zero rank correlation.",
            "- The GraphSAGE difference from Phase 1.5 is entirely attributable",
            "  to the added CPU-only indicator in the same-fold ablation.",
            "- Random forest remains best under CV5 and LOSO.",
            "- Best GNN minus MLP is inconclusive under both co-primary protocols",
            "  because the positive intervals cross the `+0.05` boundary.",
            "- Sensitivity is worse, materially stronger than its modest-group",
            "  warning suggested.",
            "- Two interactive-throughput rows dominate normalized RMSE.",
            "- The mixed model converges in every CV5 fold but is consistently",
            "  worse than Ridge on most target groups.",
            "",
            "## 5. Open questions",
            "",
            "- LOAFO remains unrun despite approval and requires explicit",
            "  instruction before execution.",
            "- Any model redesign after these results is exploratory.",
            "",
            "## 6. Acceptance criteria",
            "",
            "| Item | Status |",
            "| --- | --- |",
            "| GAT diagnostic and Phase 1 comparison | Complete |",
            "| CV5 partition-20261107 GraphSAGE regression | Complete |",
            "| Spearman NA and group counts | Complete |",
            "| Top-10 residual audit | Complete |",
            "| Per-partition GNN variance drivers | Complete |",
            "| Mixed-model specification and convergence | Complete |",
            "| Protocol hash and coverage reconciliation | Complete |",
            "| LOSO, seeds 7/11/16 | Complete |",
            "| Sensitivity, seeds 7-16 | Complete |",
            "| Combined co-primary label | Inconclusive |",
            "| LOAFO | Not run by instruction |",
            "",
            "Phase 2 protocol execution is complete. Work stops before LOAFO.",
            "",
        ]
    )
    path = THESIS / "phase2_complete_report.md"
    report_text = "\n".join(lines)
    report_text = re.sub(
        r"`([^`\n]*)`",
        lambda match: "`" + match.group(1).replace("|", r"\|") + "`",
        report_text,
    )
    path.write_text(report_text)
    print(path)


if __name__ == "__main__":
    main()
