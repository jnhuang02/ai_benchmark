#!/usr/bin/env python3
"""Generate the Phase 2 repeated-CV stop report from saved outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
CV_BUNDLE = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
PRE_RUN_BUNDLE = ROOT / "results" / "phase2-prerun-1122699"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def number(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    main_table = pd.read_csv(CV_BUNDLE / "main_table.csv").sort_values(
        "primary_error_mean"
    )
    pairs = pd.read_csv(CV_BUNDLE / "paired_differences.csv")
    selected = pd.read_csv(CV_BUNDLE / "selected_gnn_counts.csv")
    reconciliation = pd.read_csv(
        PRE_RUN_BUNDLE / "legacy_fixed_reconciliation.csv"
    )
    source_check = pd.read_csv(PRE_RUN_BUNDLE / "gb300_source_check.csv")
    family_map = pd.read_csv(
        ROOT / "folds" / "loafo_family_mapping_proposed.csv"
    )
    fold_manifest = json.loads(
        (ROOT / "folds" / "phase2_fold_manifest.json").read_text()
    )
    cv_manifest = json.loads(
        (CV_BUNDLE / "results_manifest.json").read_text()
    )
    pre_manifest = json.loads(
        (PRE_RUN_BUNDLE / "results_manifest.json").read_text()
    )
    partition_manifests = [
        json.loads(
            (
                ROOT
                / "results"
                / f"phase2-cv5-p{index}-8129f44"
                / "run_manifest.json"
            ).read_text()
        )
        for index in range(5)
    ]
    partition_runtime_sum = sum(
        manifest["total_runtime_seconds"]
        for manifest in partition_manifests
    )
    protocol_text = (ROOT / "protocol.md").read_text().rstrip()
    protocol_hash = sha256(ROOT / "protocol.md")
    if protocol_hash != "d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e":
        raise AssertionError("Protocol hash changed unexpectedly")
    lines = [
        "# Phase 2 Repeated-CV Stop Report",
        "",
        "Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.",
        "",
        "This report stops after the first complete model ladder on the five",
        "repeated five-fold partitions. LOSO and sensitivity have not run.",
        "",
        "## 1. What was done",
        "",
        "- Reconciled the seed-7 legacy/fixed table by model key, fold scheme,",
        "  evaluated rows, and coverage, then reran both policies on the same",
        "  supported rows.",
        "- Froze Protocol v2.1 before the successful Phase 2 run.",
        "- Verified the two 288-accelerator GB300 rows against the source export.",
        "- Extended production-weight equality to five selected folds.",
        "- Generated the proposed 15-family LOAFO map without fitting LOAFO.",
        "- Ran 5 partitions × 5 folds × 15 methods, with ten seeds for each",
        "  stochastic model, then computed system-clustered bootstrap summaries.",
        "- Stopped before LOSO and sensitivity.",
        "",
        "## 2. Files changed",
        "",
        "Code commit `0291baa` froze Protocol v2.1. The retained core predictions",
        "were generated at clean commit `8129f44`. The final aggregation bundle",
        "was generated at clean commit `aeeb4b4`.",
        "",
        "Canonical outputs:",
        "",
        "- Phase 2 core:",
        "  `mlperf_gnn/results/phase2-cv5-final-aeeb4b4/`",
        "- Pre-run checks:",
        "  `mlperf_gnn/results/phase2-prerun-1122699/`",
        "- Fold map: `mlperf_gnn/folds/phase2_fold_assignments.csv`",
        "- Protocol v2.1: `mlperf_gnn/protocol.md`",
        "- Protocol V2-to-V2.1 diff:",
        "  `mlperf_gnn/protocol_v2_to_v2_1.diff`",
        "- Proposed LOAFO families:",
        "  `mlperf_gnn/folds/loafo_family_mapping_proposed.csv`",
        "- Full proposed system map:",
        "  `mlperf_gnn/folds/loafo_system_assignment_proposed.csv`",
        "",
        "No thesis source (`main.tex`), PDF, or bibliography file was edited.",
        "",
        "## 3. Numbers that changed",
        "",
        "### A. Gate 3 seed-7 reconciliation",
        "",
        "The Gate 3 table mixed two fold contexts and one metric-filtering rule.",
        "",
        "| Context | Fold scheme | Fold seed | Model seed | Prediction rows | Evaluated rows | Coverage | Primary error |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    selected_contexts = [
        "rerun_Gate3_attribution_baseline_seed7_all_rows",
        "Phase1_legacy_seed7_full_predictions",
        "Phase1_legacy_seed7_identical_supported",
        "Phase1_legacy_seeds7to16_identical_supported",
        "Phase1_fixed_seed7_full_predictions",
        "Phase1_fixed_seeds7to16_identical_supported",
        "rerun_legacy_leaky_global_seed7_identical_supported",
        "rerun_isolated_supported_seed7_identical_supported",
    ]
    for context in selected_contexts:
        row = reconciliation.loc[reconciliation["context"].eq(context)].iloc[0]
        lines.append(
            f"| `{context}` | {row.fold_scheme} | {row.fold_seed} | "
            f"{row.model_seed} | {row.prediction_rows} | "
            f"{row.evaluated_rows} | {pct(row.coverage)} | "
            f"{row.primary_error:.9f} |"
        )
    lines.extend(
        [
            "",
            "`0.368149` comes from the Phase 1.5 attribution baseline: legacy",
            "96-system key, legacy seed-7 random folds, seed 7, all 476 rows,",
            "100% coverage. It is not the Phase 1 `legacy_leaky_global` result.",
            "",
            "The Phase 1 canonical split holds out 22/21/21/21/20 systems and",
            "produces 105/77/98/98/98 test rows; training-partition support leaves",
            "103/72/93/87/95 evaluated rows, 450 total. Model seeds are 7-16.",
            "",
            "Phase 1 `legacy_leaky_global` produced all 476 predictions per",
            "seed, but its reported primary metric was calculated after filtering",
            "to the same 450 supported rows. Thus its 10-seed mean `0.465427`",
            "was already on the identical evaluated set. Fixed isolation produced",
            "450 predictions and evaluated all of them. Its 10-seed mean was",
            "`0.462314`.",
            "",
            "Seed-7 identical-set rerun: legacy `0.485146`, fixed `0.510997`.",
            "The large-looking `0.368149` versus `0.510997` gap is therefore not",
            "a legitimate policy comparison.",
            "",
            "### B. Protocol v2.1",
            "",
            f"- SHA-256: `{protocol_hash}`",
            "- Frozen commit: `0291baa`",
            "- Supersedes V2 hash:",
            "  `9062c13cfe2ae50e9a5da2a0e92f48f8d2d2ef32191cd0d38942ac13b40296f4`",
            "- V2-to-V2.1 diff SHA-256:",
            "  `e8f87e52f920423c52247f6f8ef4357d382a4c847c1e65a955cbe3040f307b53`",
            "- Fold file SHA-256:",
            f"  `{fold_manifest['fold_assignments_sha256']}`",
            "",
            "The full protocol is pasted at the end of this report.",
            "",
            "The frozen compute plan was 17,610 fits: 11,875 for 5×5, 3,360",
            "for three-seed LOSO, and 2,375 for ten-seed sensitivity. At the",
            "documented 0.36 s/fit rate, the estimates were 71.25, 20.16, and",
            "14.25 minutes, totaling 105.66 minutes.",
            "",
            f"The completed 5×5 partition-fitting wall times summed to "
            f"{partition_runtime_sum:.6f} seconds "
            f"({partition_runtime_sum / 60:.6f} minutes). The five run",
            "manifests are retained in their partition directories.",
            "",
            "### C. GB300 NVL72 source verification",
            "",
            "Source file SHA-256:",
            f"`{pre_manifest['datacenter_source_sha256']}`.",
            "",
            "All 13 canonical GB300 rows match the source:",
            "`nodes × accelerators_per_node = Total Accelerators1`.",
            "",
            "| Public ID | Nodes | Accelerators/node | Product | Reported total |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    grouped_source = (
        source_check.groupby(
            ["public_id", "nodes", "accelerators_per_node"], as_index=False
        )
        .agg(
            rows=("scenario", "size"),
            product=("nodes_x_accelerators", "first"),
            reported=("reported_total_accelerators", "first"),
        )
        .sort_values(["product", "public_id"])
    )
    for row in grouped_source.itertuples(index=False):
        lines.append(
            f"| `{row.public_id}` | {row.nodes} | "
            f"{row.accelerators_per_node} | {row.product} | {row.reported} |"
        )
    lines.extend(
        [
            "",
            "The 72-versus-288 difference is the source-reported topology:",
            "public ID `6.1-0074` reports 18 nodes × 4 accelerators = 72, while",
            "`6.1-0073` reports 72 nodes × 4 accelerators = 288. Both rows in",
            "the latter submission (`row:0042`, `row:0343`) therefore use 288.",
            "",
            "### D. Production weight equality",
            "",
            "The production check covers five folds: CV Edge-containing fold",
            "`p0_fold_0`, CV fold `p0_fold_1`, LOSO Edge system `holdout_104`,",
            "LOSO Datacenter system `holdout_001`, and sensitivity Edge-containing",
            "`fold_0`. All use 150 epochs and patience 20.",
            "",
            "Result: PASS. The test ran in 2.99 seconds inside a 5.45-second",
            "process. For every fold, scrambling test metadata left every saved",
            "weight tensor exactly equal.",
            "",
            "The earlier 130-fold loop used `epochs=1, patience=1`. That loop was",
            "removed. All 130 folds retain structural isolation checks; the full",
            "production weight comparison covers the five selected folds.",
            "",
            "### E. LOAFO family map",
            "",
            "No LOAFO model was run. The proposed table has 15 families covering",
            "105 systems and 476 rows.",
            "",
            "| Family | Vendor | Architecture generation | Class | Systems | Rows | Supported rows if held out | Disposition |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in family_map.sort_values("family_id").itertuples(index=False):
        supported = (
            "NA"
            if pd.isna(row.supported_rows_if_held_out)
            else str(int(row.supported_rows_if_held_out))
        )
        lines.append(
            f"| `{row.family_id}` | {row.vendor} | "
            f"{row.architecture_generation} | {row.device_class} | "
            f"{row.systems} | {row.rows} | {supported} | "
            f"{row.holdout_disposition} |"
        )
    lines.extend(
        [
            "",
            "Headline LOAFO-eligible families with at least three systems are",
            "B300, B200, RTX PRO, MI355X, MI350X, and Intel Arc Pro B70.",
            "MI300X, H200, Radeon 8060S, GB10, Thor, and VR200 are reported",
            "separately because each has fewer than three systems. CPU-only is",
            "one labeled stress case outside the headline. The two mixed systems",
            "are always-train.",
            "",
            "### Phase 2 completed: repeated five-fold CV",
            "",
            "Point metrics pool all five folds within each partition, average the",
            "five partition metrics, and average neural seeds within row before",
            "partition metrics. Intervals use 2,000 system-cluster resamples with",
            "partition and seed variability averaged inside each draw.",
            "",
            "| Method | Primary error [95% CI] | Partition SD | Seed SD | Median row MAPE | Median-group Spearman | Normalized RMSE | Coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in main_table.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.primary_error_mean:.6f} "
            f"[{row.primary_error_95ci_low:.6f}, "
            f"{row.primary_error_95ci_high:.6f}] | "
            f"{number(row.primary_error_partition_sd)} | "
            f"{number(row.primary_error_seed_sd)} | "
            f"{row.median_row_mape_mean:.6f} | "
            f"{row.median_group_spearman_mean:.6f} | "
            f"{row.normalized_rmse_mean:.6f} | "
            f"{pct(row.coverage_mean)} |"
        )
    lines.extend(
        [
            "",
            "Pooled MAPE remains appendix-only. Full precision for every metric",
            "and interval is in `main_table.csv`; per-group results are in",
            "`per_group_results.csv`.",
            "",
            "Best-GNN validation selection counts across 250 fold-seed choices:",
            "",
            "| Selected GNN | Count |",
            "| --- | ---: |",
        ]
    )
    for row in selected.sort_values("selected_gnn").itertuples(index=False):
        lines.append(
            f"| `{row.selected_gnn}` | {row.fold_seed_selections} |"
        )
    lines.extend(
        [
            "",
            "### Paired differences",
            "",
            "| Left minus right | Role | Difference [95% CI] | Status | Holm |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| `{row.left} - {row.right}` | {row.role} | "
            f"{row.mean_difference:.6f} "
            f"[{row.difference_95ci_low:.6f}, "
            f"{row.difference_95ci_high:.6f}] | {row.status} | "
            f"{row.secondary_holm_status} |"
        )
    lines.extend(
        [
            "",
            "The primary interval excludes zero but crosses the `+0.05`",
            "equivalence boundary, so the repeated-CV label is `inconclusive`.",
            "The co-primary claim cannot be decided until LOSO. Holm-adjusted",
            "secondary results also remain pending because the frozen Holm family",
            "includes both co-primary protocols.",
            "",
            "## 4. Anything surprising or contradicting the current narrative",
            "",
            "- Random forest is the best primary-error method (`0.201853`).",
            "- MLP is second (`0.261071`) and outperforms every GNN.",
            "- GraphSAGE is `0.370271`, and the validation-selected best GNN is",
            "  `0.376393`.",
            "- GAT is the worst method (`0.632643`) and is selected least often",
            "  (13 of 250 fold-seed choices).",
            "- Real GraphSAGE and shuffled-edge GraphSAGE have an interval entirely",
            "  inside the equivalence margin (`0.024523`, 95% CI",
            "  `[0.004590, 0.045019]`), although the interval excludes zero.",
            "- Pooled MAPE ranks models differently because it is tail-sensitive;",
            "  it remains appendix-only as pre-registered.",
            "",
            "## 5. Open questions",
            "",
            "- LOSO must run before the two co-primary primary labels can be",
            "  combined.",
            "- Holm-adjusted secondary conclusions cannot be finalized until LOSO",
            "  results enter the eight-test family.",
            "- LOAFO remains unrun and awaits approval of the full 15-family table.",
            "",
            "## 6. Acceptance criteria",
            "",
            "| Item | Status |",
            "| --- | --- |",
            "| A. Seed-7 fold/evaluated-set reconciliation | Complete |",
            "| B. Protocol v2.1 with new hash and logged diff | Complete |",
            "| C. GB300 72/288 source verification | Complete |",
            "| D. Five-fold production weight equality | Complete; PASS |",
            "| E. Full 15-family LOAFO proposal | Complete; LOAFO not run |",
            "| First complete 15-method ladder on 5-fold × 5 partitions | Complete |",
            "| LOSO | Not started |",
            "| Sensitivity | Not started |",
            "",
            "Phase 2 core is complete. Work stops here pending approval to run LOSO.",
            "",
            "## Appendix: Full Protocol v2.1",
            "",
            "```markdown",
            protocol_text,
            "```",
            "",
        ]
    )
    report_path = THESIS / "phase2_cv5_ladder_report.md"
    report_path.write_text("\n".join(lines))
    print(report_path)
    print(f"cv_manifest_commit={cv_manifest['git_commit']}")
    print(f"pre_run_manifest_commit={pre_manifest['git_commit']}")


if __name__ == "__main__":
    main()
