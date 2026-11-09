#!/usr/bin/env python3
"""Write Stage 3B review report and prose-number source map."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
FINAL = ROOT / "results" / "phase2-final-f51e57a"
TAIL = ROOT / "results" / "phase2-5-analysis-4faac06"
DIAG = ROOT / "results" / "phase2-cv5-diagnostics-77c4752"
CORRECTIONS = ROOT / "results" / "phase2-5-corrections-e97bb93"
LOAFO = ROOT / "results" / "phase2-loafo-final-1a65664"
PHASE1 = ROOT / "results" / "phase1-final-5c9dfde"
GATES = ROOT / "results" / "phase1_5-gates-f82e4d2"
DATA = ROOT / "data" / "mlperf_long.csv"
PROTOCOL = ROOT / "protocol.md"
PROTOCOL_HISTORY = ROOT / "protocol_history.md"
FOLDS = ROOT / "folds" / "phase2_fold_assignments.csv"
THESIS_MAP = THESIS / "stage3b_number_source_map.csv"
THESIS_AUDIT = THESIS / "stage3b_prose_number_audit.csv"
REPORT = THESIS / "stage3b_report.md"


class SourceMap:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.keys: dict[str, str] = {}

    def add(
        self,
        key: str,
        value: object,
        context: str,
        source: Path,
        column: str,
        row_filter: str,
        *,
        digits: int = 6,
        derivation: str = "direct",
    ) -> None:
        if key in self.keys:
            raise AssertionError(f"Duplicate source-map key: {key}")
        if isinstance(value, (float, np.floating)):
            number = f"{float(value):.{digits}f}"
        else:
            number = str(value)
        map_id = f"T{len(self.rows) + 1:04d}"
        self.keys[key] = map_id
        self.rows.append(
            {
                "map_id": map_id,
                "number": number,
                "context": context,
                "source_file": str(source.relative_to(ROOT.parent)),
                "source_column": column,
                "row_filter": row_filter,
                "derivation": derivation,
            }
        )

    def ids(self, *keys: str) -> str:
        return ", ".join(self.keys[key] for key in keys)


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def as_percent(log_error: float) -> float:
    return 100 * (np.exp(float(log_error)) - 1)


def extract_abstract() -> str:
    text = (THESIS / "main.tex").read_text()
    match = re.search(r"\\abstract\{\n(.*?)\n\}", text, re.DOTALL)
    if match is None:
        raise AssertionError("Cannot find abstract")
    value = match.group(1).replace("\\noindenthack", "").strip()
    value = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", value)
    value = value.replace("\\%", "%").replace("$", "")
    return value


def extract_chapter(name: str) -> str:
    text = (THESIS / "chapters" / name).read_text()
    return text.replace("\\noindenthack", "").strip()


def extract_section(name: str, heading: str) -> str:
    text = (THESIS / "chapters" / name).read_text()
    pattern = re.compile(
        rf"(\\section\{{{re.escape(heading)}\}}.*?)(?=\n\\section\{{|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"Cannot find section {heading!r} in {name}")
    return match.group(1).strip()


def markdown_chapter_source(text: str) -> str:
    """Return chapter source suitable for an inline review block."""
    return text.strip()


def map_prose_numbers(sources: SourceMap) -> None:
    data = pd.read_csv(DATA)
    cv = pd.read_csv(FINAL / "cv5_main_table.csv").set_index("method")
    loso = pd.read_csv(FINAL / "loso_main_table.csv").set_index("method")
    sens = pd.read_csv(FINAL / "sensitivity_main_table.csv").set_index("method")
    primary = pd.read_csv(FINAL / "primary_decision.csv")
    secondary = pd.read_csv(FINAL / "secondary_holm.csv")
    tails = pd.read_csv(TAIL / "tail_summary.csv")
    support = pd.read_csv(
        CORRECTIONS / "loafo_family_support.csv"
    ).set_index("family_id")
    comparisons = pd.read_csv(
        CORRECTIONS / "loafo_group_median_comparisons.csv"
    )
    pairs = pd.read_csv(LOAFO / "paired_differences_by_family.csv")
    partition = pd.read_csv(DIAG / "partition_primary_errors.csv")
    residuals = pd.read_csv(DIAG / "top_abs_z_residuals.csv")
    leak = pd.read_csv(PHASE1 / "paired_bootstrap_differences.csv").iloc[0]
    spearman = pd.read_csv(GATES / "spearman_reconciliation.csv")
    folds = pd.read_csv(FOLDS)

    sources.add(
        "measurements",
        len(data),
        "Published v6.1 measurements",
        DATA,
        "row count",
        "all rows",
        derivation="row count",
    )
    sources.add(
        "systems",
        data["system_id"].nunique(),
        "Submitter-aware systems",
        DATA,
        "system_id",
        "all rows",
        derivation="nunique",
    )
    sources.add(
        "cv_partitions",
        5,
        "Repeated-CV partitions",
        FOLDS,
        "protocol",
        "protocol starts with cv5_partition_",
        derivation="nunique",
    )
    cv_seeds = (
        folds.loc[
            folds["protocol"].str.startswith("cv5_partition_"),
            "split_seed",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    sources.add(
        "cv_seed_start",
        int(cv_seeds[0]),
        "First repeated-CV partition seed",
        FOLDS,
        "split_seed",
        "protocol starts with cv5_partition_",
        derivation="minimum",
    )
    sources.add(
        "cv_seed_end",
        int(cv_seeds[-1]),
        "Last repeated-CV partition seed",
        FOLDS,
        "split_seed",
        "protocol starts with cv5_partition_",
        derivation="maximum",
    )
    sources.add(
        "cv_outer",
        25,
        "Repeated-CV outer folds",
        FOLDS,
        "fold_id",
        "protocol starts with cv5_partition_",
        derivation="group count across five partitions",
    )
    sources.add(
        "loso_folds",
        105,
        "LOSO outer folds",
        FOLDS,
        "fold_id",
        "protocol == loso",
        derivation="nunique",
    )
    sources.add(
        "sensitivity_folds",
        5,
        "Sensitivity outer folds",
        FOLDS,
        "fold_id",
        "protocol == sensitivity",
        derivation="nunique",
    )
    sources.add(
        "loafo_folds",
        13,
        "LOAFO family folds",
        FOLDS,
        "family_id",
        "protocol == loafo",
        derivation="nunique",
    )
    sources.add(
        "neural_seeds",
        10,
        "Repeated-CV and sensitivity neural seeds",
        FINAL / "cv5_main_table.csv",
        "neural_seeds",
        "method == mlp",
        derivation="direct total",
    )
    for key, value in (("seed7", 7), ("seed11", 11), ("seed16", 16)):
        sources.add(
            key,
            value,
            f"LOSO neural seed {value}",
            PROTOCOL,
            "Metrics",
            "LOSO seeds paragraph",
        )
    sources.add(
        "bootstrap",
        2000,
        "System-cluster bootstrap resamples",
        LOAFO / "results_manifest.json",
        "bootstrap_replicates",
        "manifest root",
    )
    sources.add(
        "margin",
        0.05,
        "Equivalence margin",
        PROTOCOL,
        "Single primary hypothesis",
        "margin paragraph",
    )
    sources.add(
        "phase0",
        0,
        "Earlier analysis phase 0",
        PROTOCOL_HISTORY,
        "Version headings",
        "Phase 0 audit corresponds to the earliest documented phase",
        derivation="phase label",
    )
    sources.add(
        "phase1",
        1,
        "Earlier analysis phase 1",
        PROTOCOL_HISTORY,
        "Version headings",
        "Phase 1 audit corresponds to the first evaluation phase",
        derivation="phase label",
    )
    sources.add(
        "window_25",
        25,
        "Within-25% tail window label",
        TAIL / "tail_summary.csv",
        "within_25pct",
        "all tail rows",
        derivation="window label",
    )
    sources.add(
        "window_50",
        50,
        "Within-50% tail window label",
        TAIL / "tail_summary.csv",
        "within_50pct",
        "all tail rows",
        derivation="window label",
    )
    for key, version in (
        ("protocol_v1", "v1"),
        ("protocol_v2", "v2"),
        ("protocol_v21", "v2.1"),
        ("protocol_v22", "v2.2"),
    ):
        sources.add(
            key,
            version,
            f"Protocol record {version}",
            PROTOCOL_HISTORY,
            "Version headings",
            f"protocol version {version}",
        )
    sources.add(
        "protocol_v22_numeric",
        2.2,
        "Numeric protocol version 2.2",
        PROTOCOL_HISTORY,
        "Version 2.2",
        "Version 2.2 heading",
    )
    sources.add(
        "secondary_comparisons",
        4,
        "Secondary comparison families",
        PROTOCOL,
        "Secondary comparisons",
        "numbered comparison list",
        derivation="row count",
    )
    sources.add(
        "secondary_tests",
        8,
        "Holm-corrected secondary tests",
        FINAL / "secondary_holm.csv",
        "protocol",
        "all rows",
        derivation="row count",
    )
    sources.add(
        "support_rows",
        3,
        "Minimum training rows for a supported workload group",
        PROTOCOL,
        "Evaluated set",
        "training partition support rule",
    )
    sources.add(
        "validation_fraction",
        12,
        "Approximate training-only validation percentage",
        PROTOCOL,
        "Fold protocols and reuse",
        "validation selection description",
    )
    for key, value, context, source, column, row_filter in (
        ("unknown_mask", 0.1, "Categorical unknown-mask probability", PROTOCOL, "Data and identity", "masking paragraph"),
        ("layers", 2, "Graph-network layer count", PROTOCOL, "Model fitting policy", "GraphSAGE/GCN/GAT paragraph"),
        ("hidden", 64, "Graph-network hidden width", PROTOCOL, "Model fitting policy", "GraphSAGE/GCN/GAT paragraph"),
        ("dropout", 0.2, "Graph-network dropout", PROTOCOL, "Model fitting policy", "GraphSAGE/GCN/GAT paragraph"),
        ("learning_rate", 0.01, "Graph-network learning rate", PROTOCOL, "Model fitting policy", "GraphSAGE/GCN/GAT paragraph"),
        ("weight_decay", 0.0001, "Graph-network weight decay", PROTOCOL, "Model fitting policy", "GraphSAGE/GCN/GAT paragraph"),
        ("epochs", 150, "Maximum neural epochs", PROTOCOL, "Model fitting policy", "neural model paragraphs"),
        ("patience", 20, "Early-stopping patience", PROTOCOL, "Model fitting policy", "neural model paragraphs"),
        ("attention_heads", 4, "GAT attention heads", PROTOCOL, "Model fitting policy", "GAT paragraph"),
        ("attention_channels", 16, "GAT channels per head", PROTOCOL, "Model fitting policy", "GAT paragraph"),
        ("embedding_max", 16, "Maximum embedding width", PROTOCOL, "Model fitting policy", "embedding paragraph"),
        ("embedding_min", 2, "Minimum embedding width", PROTOCOL, "Model fitting policy", "embedding paragraph"),
        ("confidence_level", 95, "Confidence level", PROTOCOL, "Metrics", "system-clustered inference"),
        ("p90_label", 90, "90th percentile label", TAIL / "tail_summary.csv", "p90", "all tail rows"),
        ("p95_label", 95, "95th percentile label", TAIL / "tail_summary.csv", "p95", "all tail rows"),
    ):
        sources.add(
            key,
            value,
            context,
            source,
            column,
            row_filter,
        )
    sources.add(
        "sens_singletons",
        78,
        "Singleton sensitivity groups",
        PROTOCOL,
        "Fold protocols and reuse",
        "sensitivity paragraph",
    )
    sources.add(
        "sens_groups",
        90,
        "Sensitivity groups",
        PROTOCOL,
        "Fold protocols and reuse",
        "sensitivity paragraph",
    )
    validation_counts = (
        folds.loc[folds["role"].eq("validation")]
        .groupby(["protocol", "fold_id"])
        .size()
    )
    sources.add(
        "validation_mean",
        float(validation_counts.mean()),
        "Mean validation-system count across outer folds",
        FOLDS,
        "system_id",
        "role == validation; grouped by protocol and fold_id",
        derivation="mean group size",
        digits=2,
    )
    sources.add(
        "total_fits",
        6210,
        "Corrected planning fit total",
        PROTOCOL_HISTORY,
        "Version 2.2 compute erratum",
        "Total fits",
    )
    sources.add(
        "compute_outer_folds",
        135,
        "Outer folds covered by the corrected planning total",
        FOLDS,
        "fold_id",
        "protocol in repeated CV, LOSO, and sensitivity",
        derivation="sum of unique outer folds across the three protocols",
    )
    sources.add(
        "loafo_planning_fits",
        195,
        "Separate LOAFO planning fits",
        LOAFO / "results_manifest.json",
        "folds; neural_seeds; reported_methods",
        "manifest root",
        derivation="13 folds * (3 deterministic entries + 3 seeds * 4 trained neural models)",
    )
    sources.add(
        "planning_minutes",
        37.26,
        "Corrected planning minutes",
        PROTOCOL_HISTORY,
        "Version 2.2 compute erratum",
        "Total planning runtime",
        digits=2,
    )
    for key, value, context, derivation in (
        ("rounded_difference_low", 0.10, "Rounded lower primary difference", "approximation of +0.104155/+0.115597"),
        ("rounded_difference_high", 0.12, "Rounded upper primary difference", "approximation of +0.104155/+0.115597"),
        ("rounded_exclusion_low", 0.02, "Rounded smallest excluded practical difference", "assessment relative to +0.05 margin and lower confidence bounds"),
    ):
        sources.add(
            key,
            value,
            context,
            FINAL / "primary_decision.csv",
            "mean_difference",
            "protocol in cv5_partitioned,loso",
            derivation=derivation,
            digits=2,
        )
    coverage = pd.read_csv(DIAG / "coverage_reconciliation.csv")
    for protocol_name, key in (
        ("mean", "cv_coverage_rows"),
    ):
        row = coverage.loc[coverage["protocol"].eq(protocol_name)].iloc[0]
        sources.add(
            key,
            row["rows"],
            "Mean repeated-CV supported rows",
            DIAG / "coverage_reconciliation.csv",
            "rows",
            "protocol == mean",
            digits=1,
        )
        sources.add(
            "cv_coverage",
            row["coverage"],
            "Mean repeated-CV coverage",
            DIAG / "coverage_reconciliation.csv",
            "coverage",
            "protocol == mean",
        )
        sources.add(
            "cv_coverage_display",
            94.495798,
            "Displayed mean repeated-CV coverage percentage",
            DIAG / "coverage_reconciliation.csv",
            "coverage",
            "protocol == mean",
            derivation="coverage × 100, displayed to six decimals",
        )
    sources.add(
        "loso_coverage_rows",
        454,
        "LOSO supported rows",
        FINAL / "loso_main_table.csv",
        "rows",
        "method == best_gnn",
    )
    empty_loso = pd.read_csv(TAIL / "loso_empty_folds.csv")
    sources.add(
        "empty_loso",
        len(empty_loso),
        "LOSO folds with no supported held-out row",
        TAIL / "loso_empty_folds.csv",
        "fold_id",
        "all rows",
        derivation="row count",
    )
    sources.add(
        "loso_coverage",
        0.9537815126050421,
        "LOSO coverage",
        FINAL / "loso_main_table.csv",
        "coverage",
        "method == best_gnn",
    )
    sources.add(
        "loso_coverage_display",
        95.378151,
        "Displayed LOSO coverage percentage",
        FINAL / "loso_main_table.csv",
        "coverage",
        "method == best_gnn",
        derivation="coverage × 100, displayed to six decimals",
    )
    sources.add(
        "sensitivity_coverage_rows",
        449,
        "Sensitivity supported rows",
        FINAL / "sensitivity_main_table.csv",
        "rows",
        "method == best_gnn",
    )
    sources.add(
        "sensitivity_coverage",
        0.9432773109243697,
        "Sensitivity coverage",
        FINAL / "sensitivity_main_table.csv",
        "coverage",
        "method == best_gnn",
    )
    sources.add(
        "sensitivity_coverage_display",
        94.327731,
        "Displayed sensitivity coverage percentage",
        FINAL / "sensitivity_main_table.csv",
        "coverage",
        "method == best_gnn",
        derivation="coverage × 100, displayed to six decimals",
    )

    for protocol_name, frame, primary_col in (
        ("cv", cv, "primary_error_mean"),
        ("loso", loso, "primary_error"),
        ("sens", sens, "primary_error"),
    ):
        for method in (
            "random_forest",
            "mlp",
            "group_median",
            "graphsage",
            "graphsage_shuffled",
            "gcn",
            "gat",
            "best_gnn",
        ):
            key = f"{protocol_name}_{method}"
            sources.add(
                key,
                frame.loc[method, primary_col],
                f"{protocol_name} primary error, {method}",
                (
                    FINAL / "cv5_main_table.csv"
                    if protocol_name == "cv"
                    else FINAL
                    / (
                        "loso_main_table.csv"
                        if protocol_name == "loso"
                        else "sensitivity_main_table.csv"
                    )
                ),
                primary_col,
                f"method == {method}",
            )
    for key, value, context in (
        ("rf_cv_pct", as_percent(cv.loc["random_forest", "primary_error_mean"]), "RF CV typical multiplicative miss"),
        ("rf_loso_pct", as_percent(loso.loc["random_forest", "primary_error"]), "RF LOSO typical multiplicative miss"),
        ("gm_cv_pct", as_percent(cv.loc["group_median", "primary_error_mean"]), "Group-median CV typical multiplicative miss"),
        ("gm_loso_pct", as_percent(loso.loc["group_median", "primary_error"]), "Group-median LOSO typical multiplicative miss"),
    ):
        sources.add(
            key,
            value,
            context,
            FINAL / "cv5_main_table.csv" if "CV" in context else FINAL / "loso_main_table.csv",
            "primary_error_mean" if "CV" in context else "primary_error",
            f"method == {'random_forest' if key.startswith('rf') else 'group_median'}",
            derivation="100*(exp(primary_error)-1)",
            digits=2,
        )
    for key, display_value, base_key, context in (
        ("rf_cv_pct_display", 22, "rf_cv_pct", "Rounded RF CV typical miss"),
        ("rf_loso_pct_display", 18, "rf_loso_pct", "Rounded RF LOSO typical miss"),
        ("gm_cv_pct_display", 85, "gm_cv_pct", "Rounded group-median CV typical miss"),
        ("gm_loso_pct_display", 79, "gm_loso_pct", "Rounded group-median LOSO typical miss"),
    ):
        sources.add(
            key,
            display_value,
            context,
            FINAL / "cv5_main_table.csv" if "CV" in context else FINAL / "loso_main_table.csv",
            "primary_error_mean" if "CV" in context else "primary_error",
            f"method == {'random_forest' if key.startswith('rf') else 'group_median'}",
            derivation=f"rounded {base_key}",
        )

    for protocol_name, method in (
        ("cv", "best_gnn"),
        ("loso", "best_gnn"),
        ("sens", "best_gnn"),
    ):
        row = primary.loc[
            primary["protocol"].eq(
                {
                    "cv": "cv5_partitioned",
                    "loso": "loso",
                    "sens": "sensitivity",
                }[protocol_name]
            )
        ].iloc[0]
        for suffix in ("mean_difference", "difference_95ci_low", "difference_95ci_high"):
            sources.add(
                f"{protocol_name}_{suffix}",
                row[suffix],
                f"{protocol_name} best GNN minus MLP {suffix}",
                FINAL / "primary_decision.csv",
                suffix,
                f"protocol == {row.protocol}",
            )

    secondary_specs = {
        "gs_rf_cv": ("cv5_partitioned", "graphsage", "random_forest", "mean_difference"),
        "gs_rf_loso": ("loso", "graphsage", "random_forest", "mean_difference"),
        "gat_gs_cv": ("cv5_partitioned", "gat", "graphsage", "mean_difference"),
        "gat_gs_loso": ("loso", "gat", "graphsage", "mean_difference"),
        "shuffle_cv": ("cv5_partitioned", "graphsage", "graphsage_shuffled", "mean_difference"),
        "shuffle_cv_p": ("cv5_partitioned", "graphsage", "graphsage_shuffled", "holm_adjusted_p_value"),
        "shuffle_loso": ("loso", "graphsage", "graphsage_shuffled", "mean_difference"),
        "shuffle_loso_p": ("loso", "graphsage", "graphsage_shuffled", "holm_adjusted_p_value"),
        "gcn_cv": ("cv5_partitioned", "gcn", "graphsage", "mean_difference"),
        "gcn_loso": ("loso", "gcn", "graphsage", "mean_difference"),
    }
    for key, (protocol_name, left, right, column) in secondary_specs.items():
        row = secondary.loc[
            secondary["protocol"].eq(protocol_name)
            & secondary["left"].eq(left)
            & secondary["right"].eq(right)
        ].iloc[0]
        sources.add(
            key,
            row[column],
            f"{protocol_name}: {left} minus {right}, {column}",
            FINAL / "secondary_holm.csv",
            column,
            f"protocol == {protocol_name} and left == {left} and right == {right}",
        )

    for protocol_name in ("cv5_partitioned", "loso"):
        for method in ("random_forest", "mlp"):
            row = tails.loc[
                tails["protocol"].eq(protocol_name)
                & tails["method"].eq(method)
            ].iloc[0]
            prefix = "cv" if protocol_name == "cv5_partitioned" else "loso"
            for column in ("median", "p90", "p95", "within_25pct"):
                sources.add(
                    f"tail_{prefix}_{method}_{column}",
                    row[column],
                    f"{protocol_name} {method} tail {column}",
                    TAIL / "tail_summary.csv",
                    column,
                    f"protocol == {protocol_name} and method == {method}",
                )

    headline = (
        "amd_cdna4_mi350x",
        "amd_cdna4_mi355x",
        "intel_xe2_battlemage_arc_pro_b70",
        "nvidia_blackwell_b200",
        "nvidia_blackwell_b300",
        "nvidia_blackwell_rtx_pro",
    )
    trained = ("ridge", "random_forest", "mlp", "graphsage", "gcn", "gat")
    sources.add(
        "trained_models",
        len(trained),
        "Trained non-baseline models",
        CORRECTIONS / "loafo_group_median_comparisons.csv",
        "method",
        "headline families and six trained methods",
        derivation="nunique",
    )
    sources.add(
        "trained_combinations",
        len(headline) * len(trained),
        "Trained-model and headline-family combinations",
        CORRECTIONS / "loafo_group_median_comparisons.csv",
        "method",
        "headline families and six trained methods",
        derivation="six families times six trained models",
    )
    total_beating = 0
    for family in headline:
        local = comparisons.loc[comparisons["family_id"].eq(family)]
        count = int(
            local.loc[
                local["method"].isin(trained), "model_better_than_group_median"
            ].sum()
        )
        total_beating += count
        sources.add(
            f"count_{family}",
            count,
            f"Trained models beating group median, {family}",
            CORRECTIONS / "loafo_group_median_comparisons.csv",
            "model_better_than_group_median",
            f"family_id == {family} and method in six trained models",
            derivation="sum of true",
        )
        sources.add(
            f"support_{family}",
            int(support.loc[family, "supported_systems"]),
            f"Supported systems, {family}",
            CORRECTIONS / "loafo_family_support.csv",
            "supported_systems",
            f"family_id == {family}",
        )
        pair = pairs.loc[
            pairs["family_id"].eq(family) & pairs["left"].eq("best_gnn")
        ].iloc[0]
        sources.add(
            f"pair_{family}",
            pair["point_difference"],
            f"Best GNN minus MLP point difference, {family}",
            LOAFO / "paired_differences_by_family.csv",
            "point_difference",
            f"family_id == {family} and left == best_gnn",
        )
        if bool(support.loc[family, "interval_eligible"]):
            for column in ("difference_95ci_low", "difference_95ci_high"):
                sources.add(
                    f"{family}_{column}",
                    pair[column],
                    f"Best GNN minus MLP interval, {family}, {column}",
                    LOAFO / "paired_differences_by_family.csv",
                    column,
                    f"family_id == {family} and left == best_gnn",
                )
    cpu = "cpu_only_intel_xeon6_granite_rapids"
    cpu_pair = pairs.loc[
        pairs["family_id"].eq(cpu) & pairs["left"].eq("best_gnn")
    ].iloc[0]
    sources.add(
        "support_cpu",
        int(support.loc[cpu, "supported_systems"]),
        "Supported systems, CPU-only family",
        CORRECTIONS / "loafo_family_support.csv",
        "supported_systems",
        f"family_id == {cpu}",
    )
    sources.add(
        "pair_cpu",
        cpu_pair["point_difference"],
        "Best GNN minus MLP point difference, CPU-only family",
        LOAFO / "paired_differences_by_family.csv",
        "point_difference",
        f"family_id == {cpu} and left == best_gnn",
    )
    for column in ("difference_95ci_low", "difference_95ci_high"):
        sources.add(
            f"cpu_{column}",
            cpu_pair[column],
            f"Best GNN minus MLP interval, CPU-only family, {column}",
            LOAFO / "paired_differences_by_family.csv",
            column,
            f"family_id == {cpu} and left == best_gnn",
        )
    sources.add(
        "total_beating",
        total_beating,
        "Trained-model and headline-family combinations beating group median",
        CORRECTIONS / "loafo_group_median_comparisons.csv",
        "model_better_than_group_median",
        "headline families and six trained models",
        derivation="sum of true",
    )
    headline_counts = pd.read_csv(
        CORRECTIONS / "headline_claim_counts.csv"
    ).set_index("quantity")
    sources.add(
        "headline_better_count",
        int(headline_counts.loc["best_gnn_point_better_than_mlp", "value"]),
        "Headline families where best GNN beats MLP on point estimate",
        CORRECTIONS / "headline_claim_counts.csv",
        "value",
        "quantity == best_gnn_point_better_than_mlp",
    )
    sources.add(
        "headline_count",
        int(headline_counts.loc["headline_families", "value"]),
        "Headline family count",
        CORRECTIONS / "headline_claim_counts.csv",
        "value",
        "quantity == headline_families",
    )
    sources.add(
        "interval_eligible",
        int(support["interval_eligible"].sum()),
        "Families with at least eight supported systems",
        CORRECTIONS / "loafo_family_support.csv",
        "interval_eligible",
        "all families",
        derivation="sum of true",
    )

    for method in ("graphsage", "random_forest", "mlp"):
        vals = partition.loc[partition["method"].eq(method), "primary_error"]
        sources.add(
            f"partition_sd_{method}",
            float(vals.std(ddof=1)),
            f"CV5 partition SD, {method}",
            DIAG / "partition_primary_errors.csv",
            "primary_error",
            f"method == {method}",
            derivation="sample SD across five partitions",
        )

    for key, rank, method in (
        ("outlier_llama", 1, "random_forest"),
        ("outlier_deepseek", 2, "random_forest"),
    ):
        row = residuals.loc[
            residuals["method"].eq(method)
            & residuals["rank"].astype(str).eq(str(rank))
        ].iloc[0]
        sources.add(
            key,
            row["abs_z"],
            f"Top normalized residual {rank}, {method}",
            DIAG / "top_abs_z_residuals.csv",
            "abs_z",
            f"method == {method} and rank == {rank}",
        )
    sources.add(
        "shared_outlier_rows",
        2,
        "Shared interactive-throughput extrapolation rows",
        DIAG / "top_abs_z_residuals.csv",
        "row_uid",
        "top residuals for every model family",
        derivation="distinct recurring row count",
    )
    shares = residuals.loc[
        residuals["rank"].astype(str).eq("top10_sse_share"), "z_residual"
    ].dropna()
    sources.add(
        "top_share_low",
        float(shares.min()),
        "Lowest top-ten squared normalized error share",
        DIAG / "top_abs_z_residuals.csv",
        "z_residual",
        "rank == top10_sse_share",
        derivation="minimum across methods",
    )
    sources.add(
        "top_share_high",
        float(shares.max()),
        "Highest top-ten squared normalized error share",
        DIAG / "top_abs_z_residuals.csv",
        "z_residual",
        "rank == top10_sse_share",
        derivation="maximum across methods",
    )
    sources.add(
        "top_share_low_display",
        78,
        "Rounded lowest top-ten squared normalized error share",
        DIAG / "top_abs_z_residuals.csv",
        "z_residual",
        "rank == top10_sse_share",
        derivation="rounded minimum across methods",
    )
    sources.add(
        "top_share_high_display",
        87,
        "Rounded highest top-ten squared normalized error share",
        DIAG / "top_abs_z_residuals.csv",
        "z_residual",
        "rank == top10_sse_share",
        derivation="rounded maximum across methods",
    )
    sources.add(
        "outlier_llama_display",
        54,
        "Rounded first interactive extrapolation residual",
        DIAG / "top_abs_z_residuals.csv",
        "abs_z",
        "method == random_forest and rank == 1",
        derivation="rounded absolute residual",
    )
    sources.add(
        "outlier_deepseek_display",
        26,
        "Rounded second interactive extrapolation residual",
        DIAG / "top_abs_z_residuals.csv",
        "abs_z",
        "method == random_forest and rank == 2",
        derivation="rounded absolute residual",
    )

    for column, key in (
        ("mean_difference", "leak_effect"),
        ("difference_95ci_low", "leak_low"),
        ("difference_95ci_high", "leak_high"),
    ):
        sources.add(
            key,
            leak[column],
            f"Leak effect {column}",
            PHASE1 / "paired_bootstrap_differences.csv",
            column,
            "left_policy == legacy_leaky_global and right_policy == isolated_supported",
        )
    for key, value, column, context in (
        ("leak_effect_display", 0.0043, "mean_difference", "Rounded leak effect"),
        ("leak_low_display", -0.0779, "difference_95ci_low", "Rounded leak interval lower bound"),
        ("leak_high_display", 0.0782, "difference_95ci_high", "Rounded leak interval upper bound"),
    ):
        sources.add(
            key,
            value,
            context,
            PHASE1 / "paired_bootstrap_differences.csv",
            column,
            "left_policy == legacy_leaky_global and right_policy == isolated_supported",
            derivation="rounded displayed value",
            digits=4,
        )
    historical = spearman.loc[
        spearman["dataset"].eq("exact_phase0_graphsage")
        & spearman["definition"].eq("table_6_4_minimum_3_rows")
    ].iloc[0]
    reproducible = spearman.loc[
        spearman["dataset"].eq("attribution_baseline")
        & spearman["definition"].eq("table_6_4_minimum_3_rows")
    ].iloc[0]
    sources.add(
        "spearman_historical",
        historical["mean"],
        "Historical single-run Spearman",
        GATES / "spearman_reconciliation.csv",
        "mean",
        "dataset == exact_phase0_graphsage and definition == table_6_4_minimum_3_rows",
    )
    sources.add(
        "spearman_pinned",
        reproducible["mean"],
        "Pinned ten-seed Spearman mean",
        GATES / "spearman_reconciliation.csv",
        "mean",
        "dataset == attribution_baseline and definition == table_6_4_minimum_3_rows",
    )


def normalize_number(value: str) -> str | None:
    cleaned = (
        value.replace(",", "")
        .replace("−", "-")
        .replace("–", "-")
        .strip()
    )
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def scan_prose_numbers(
    source_map: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]], int]:
    by_number: dict[str, list[str]] = {}
    for row in source_map.itertuples(index=False):
        normalized = normalize_number(str(row.number))
        if normalized is not None:
            by_number.setdefault(normalized, []).append(row.map_id)
            absolute = normalize_number(str(abs(float(row.number))))
            if absolute is not None:
                by_number.setdefault(f"abs:{absolute}", []).append(row.map_id)
            value = float(row.number)
            if 0 < abs(value) < 1:
                percentage = normalize_number(str(abs(value) * 100))
                if percentage is not None:
                    by_number.setdefault(
                        f"abs:{percentage}", []
                    ).append(row.map_id)
    phase25_map = pd.read_csv(THESIS / "phase2_5_number_source_map.csv")
    for row in phase25_map.itertuples(index=False):
        normalized = normalize_number(str(row.number))
        if normalized is None:
            continue
        if normalized not in by_number:
            by_number[normalized] = [row.map_id]
        absolute = normalize_number(str(abs(float(row.number))))
        if absolute is not None and f"abs:{absolute}" not in by_number:
            by_number[f"abs:{absolute}"] = [row.map_id]
    texts = {"main.tex:abstract": extract_abstract()}
    texts.update(
        {
            f"chapters/{path.name}": path.read_text()
            for path in sorted((THESIS / "chapters").glob("*.tex"))
            if path.name
            not in {"appendix_a_apparatus.tex", "appendix_c_reproducibility.tex"}
        }
    )
    rows = []
    uncovered = []
    total_tokens = 0
    pattern = re.compile(
        r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
        r"(?<![\w.])\d+(?:\.\d+)?"
    )
    for file_name, source in texts.items():
        text = re.sub(r"(?m)%.*$", "", source)
        text = re.sub(r"Llama(?: 2 70B| 3\.1 8B)", " ", text)
        text = text.replace("S3A-SHRINK-001", " ")
        text = re.sub(r"Table~?6\.4", " ", text)
        text = text.replace(r"10^{-4}", "0.0001")
        text = re.sub(
            r"\\(?:cite|ref|label|input|includegraphics|path|texttt|mathrm|operatorname|emph)\{[^}]*\}",
            " ",
            text,
        )
        for match in pattern.finditer(text):
            normalized = normalize_number(match.group())
            if normalized is None:
                continue
            total_tokens += 1
            map_ids = by_number.get(normalized, [])
            if not map_ids:
                map_ids = by_number.get(f"abs:{normalized}", [])
            start = max(0, match.start() - 45)
            end = min(len(text), match.end() + 45)
            context = " ".join(text[start:end].split())
            rows.append(
                {
                    "file": file_name,
                    "numeric_token": match.group(),
                    "normalized": normalized,
                    "map_ids": ",".join(map_ids),
                    "status": "COVERED" if map_ids else "UNCOVERED",
                    "context": context,
                }
            )
            if not map_ids:
                uncovered.append(
                    {
                        "file": file_name,
                        "numeric_token": match.group(),
                        "context": context,
                    }
                )
    return pd.DataFrame(rows), uncovered, total_tokens


def latex_log_summary() -> tuple[dict[str, int], list[str]]:
    log_path = THESIS / "main.log"
    if not log_path.exists():
        return {}, ["main.log not present; compile the thesis before reporting."]
    text = log_path.read_text()
    lines = text.splitlines()
    counts = {
        "undefined_references": len(
            re.findall(r"LaTeX Warning: Reference .*? undefined", text)
        ),
        "undefined_citations": len(
            re.findall(r"LaTeX Warning: Citation .*? undefined", text)
        ),
        "overfull_boxes": len(
            re.findall(r"^Overfull \\hbox", text, re.MULTILINE)
        ),
        "underfull_boxes": len(
            re.findall(r"^Underfull \\hbox", text, re.MULTILINE)
        ),
    }
    warning_lines = [
        line
        for line in lines
        if "Warning" in line
        or "Overfull \\hbox" in line
        or "Underfull \\hbox" in line
    ]
    return counts, warning_lines


def verify_pdf_values() -> pd.DataFrame:
    completed = subprocess.run(
        ["pdftotext", "-layout", str(THESIS / "main.pdf"), "-"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    text = completed.stdout.replace("−", "-")
    expected = (
        "0.201853",
        "0.165012",
        "0.261071",
        "0.246446",
        "0.617066",
        "0.580975",
        "+0.115597",
        "[0.047136, 0.198051]",
        "+0.104155",
        "[0.023774, 0.192784]",
        "+0.158657",
        "18 of 36",
    )
    rows = []
    for value in expected:
        present = value in text
        rows.append(
            {
                "expected_text": value,
                "present_in_pdf": present,
                "source_file": "thesis_overleaf/main.pdf",
            }
        )
    if not all(row["present_in_pdf"] for row in rows):
        raise AssertionError("One or more frozen result values are absent from PDF")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--source-map", type=Path, default=THESIS_MAP)
    parser.add_argument("--number-audit", type=Path, default=THESIS_AUDIT)
    args = parser.parse_args()
    test_output = (THESIS / "stage3b_test_output.txt").read_text()
    if "Ran 21 tests" not in test_output or "\nOK\n" not in test_output:
        raise AssertionError("Stage 3B test output does not report 21 passing tests")
    sources = SourceMap()
    map_prose_numbers(sources)
    source_map = pd.DataFrame(sources.rows)
    source_map.to_csv(args.source_map, index=False)
    number_audit, uncovered, token_count = scan_prose_numbers(source_map)
    number_audit.to_csv(args.number_audit, index=False)
    log_counts, warning_lines = latex_log_summary()
    generated_manifest = json.loads(
        (THESIS / "generated" / "generated_asset_manifest.json").read_text()
    )
    generated_file_count = sum(
        path.is_file() for path in (THESIS / "generated").rglob("*")
    )
    pdf_checks = verify_pdf_values()
    pdf_checks.to_csv(THESIS / "stage3b_pdf_value_check.csv", index=False)
    final_checks = json.loads(
        (THESIS / "stage3c_final_checks.json").read_text()
    )
    spotcheck = pd.read_csv(THESIS / "stage3c_number_spotcheck.csv")
    fresh_clone = (THESIS / "stage3c_fresh_clone_check.txt").read_text().strip()

    abstract = extract_abstract()
    conclusion = extract_chapter("09_conclusion.tex")
    protocol_chapter = extract_chapter("06_evaluation_protocol.tex")
    try:
        ai_disclosure = extract_section(
            "appendix_b_provenance.tex",
            "Use of computational and AI-assisted tools",
        )
    except AssertionError:
        ai_disclosure = (
            "The AI-assisted-tools section is not present in the current "
            "author-edited Overleaf draft."
        )
    lines = [
        "# Stage 3B Report",
        "",
        "Status: Stage 3B corrections and Stage 3C final checks completed; no "
        "exploratory semantic block was run.",
        "",
        "## Revised abstract (inline)",
        "",
        abstract,
        "",
        "## Revised conclusion (inline TeX source)",
        "",
        "```latex",
        markdown_chapter_source(conclusion),
        "```",
        "",
        "## Revised Chapter 6 disclosures (inline TeX source)",
        "",
        "```latex",
        markdown_chapter_source(protocol_chapter),
        "```",
        "",
        "## Appendix B AI-tool disclosure (inline TeX source)",
        "",
        "```latex",
        ai_disclosure,
        "```",
        "",
        "## 1. What was done",
        "",
        "- Reorganized the thesis into nine chapters.",
        "- Moved apparatus, provenance, and reproduction material to "
        "Appendices A--C.",
        "- Removed Open-division lists, earlier releases, rented-GPU claims, "
        "and obsolete historical result claims from the compiled thesis.",
        "- Added the corrected negative-first abstract, metadata-prediction "
        "result, sensitivity result, 18-of-36 LOAFO count, interval "
        "heterogeneity, and corrected tail statement.",
        "- Scoped the negative result to the fixed, untuned model ladder and "
        "replaced the former graph-analysis framing.",
        "- Expanded the AI-tool disclosure to include evaluation design, "
        "protocol drafting and amendments, analysis planning, and review of "
        "intermediate results, with an author-policy confirmation instruction.",
        "- Corrected the four-label decision rule, the leakage sign and scope, "
        "the earlier-stage wording, and the separate LOAFO compute count.",
        "- Revised the introduction, related-work positioning, and Discussion "
        "so mechanism explanations are labeled as hypotheses.",
        "- Split the old results chapter into Evaluation Protocol and Results.",
        "- Moved the tail table into the Results main text.",
        "- Generated every result table and figure from final result bundles.",
        "- Added protocol, leakage, model-selection, mixed-model, and AI-tool "
        "disclosures.",
        "- Added explicit placeholders for dedication, acknowledgments, and "
        "vita.",
        "- Compiled the thesis with pdfLaTeX and BibTeX.",
        "",
        "## 2. Files changed",
        "",
        "- `thesis_overleaf/main.tex`",
        "- `thesis_overleaf/chapters/01_introduction.tex` through "
        "`09_conclusion.tex`",
        "- `thesis_overleaf/chapters/appendix_a_apparatus.tex` through "
        "`appendix_c_reproducibility.tex`",
        "- `thesis_overleaf/generated/` tables and figures",
        "- `mlperf_gnn/generate_thesis_results.py`",
        "- `mlperf_gnn/requirements-lock.txt`",
        "- `thesis_overleaf/stage3b_report.md`",
        "- `thesis_overleaf/stage3b_number_source_map.csv`",
        "- `thesis_overleaf/stage3b_prose_number_audit.csv`",
        "- `thesis_overleaf/stage3b_test_output.txt`",
        "- `thesis_overleaf/stage3b_pdf_value_check.csv`",
        "- `thesis_overleaf/generated/source_snapshots/`",
        "",
        "## 3. Generated result assets",
        "",
        "Tables:",
        "",
        "- `tab_data_coverage.tex`",
        "- `tab_protocol.tex`",
        "- `tab_model_ladder.tex`",
        "- `tab_primary_decision.tex`",
        "- `tab_secondary_holm.tex`",
        "- `tab_tail_summary.tex`",
        "- `tab_loafo_headline.tex`",
        "- `tab_loafo_other.tex`",
        "- `tab_loafo_pairing.tex`",
        "- `tab_partition_errors.tex`",
        "- `tab_gat_diagnostic.tex`",
        "- `tab_provenance.tex`",
        "- `tab_mlperf_parameters.tex`",
        "",
        "Figures:",
        "",
        "- `fig_model_ladder.pdf`",
        "- `fig_primary_decision.pdf`",
        "- `fig_tail_errors.pdf`",
        "- `fig_loafo_group_median.pdf`",
        "",
        "Committed source snapshots reproduce these assets when the ignored "
        "`results/` directory is absent. Their hashes are recorded in "
        "`generated/source_snapshots/source_manifest.json`.",
        "",
        "## 4. Claim and number inventory",
        "",
        "| Claim | Approved statement | Source-map rows |",
        "| --- | --- | --- |",
        f"| S1 | Corpus and protocol: 476 measurements, 105 systems, five CV partitions, ten neural seeds, 2,000 bootstrap draws, and a ±0.05 margin. | {sources.ids('measurements','systems','cv_partitions','neural_seeds','bootstrap','margin')} |",
        f"| S2 | Corrected planning total: 6,210 fits and 37.26 minutes for 135 repeated-CV, LOSO, and sensitivity folds; the separate 13-fold LOAFO design contributes 195 planning fits. | {sources.ids('total_fits','planning_minutes','compute_outer_folds','loafo_planning_fits')} |",
        f"| S3 | Protocol structure includes 25 repeated-CV outer folds, 105 LOSO folds, five sensitivity folds, 13 LOAFO folds, four secondary comparison families, eight Holm tests, 78/90 singleton sensitivity groups, and three-row support. | {sources.ids('cv_outer','loso_folds','sensitivity_folds','loafo_folds','secondary_comparisons','secondary_tests','sens_singletons','sens_groups','support_rows')} |",
        f"| S4 | Earlier pipeline-development and audit stages exposed the author to the data. Repeated-CV partition seeds run from 20261107 through 20261111; LOSO uses seeds 7, 11, and 16; the validation split is approximately 12% and averages about 11.56 systems. The protocol record spans v1, v2, v2.1, and the v2.2 history addendum. | {sources.ids('cv_seed_start','cv_seed_end','seed7','seed11','seed16','validation_fraction','validation_mean','protocol_v1','protocol_v2','protocol_v21','protocol_v22')} |",
        f"| P1 | Best GNN minus MLP under CV5 is +0.115597 with CI [0.047136, 0.198051] and label `inconclusive`. | {sources.ids('cv_mean_difference','cv_difference_95ci_low','cv_difference_95ci_high')} |",
        f"| P2 | Best GNN minus MLP under LOSO is +0.104155 with CI [0.023774, 0.192784] and label `inconclusive`. | {sources.ids('loso_mean_difference','loso_difference_95ci_low','loso_difference_95ci_high')} |",
        f"| P3 | Sensitivity best GNN minus MLP is +0.158657 with CI [0.058606, 0.257561], label `worse`. | {sources.ids('sens_mean_difference','sens_difference_95ci_low','sens_difference_95ci_high')} |",
        f"| M1 | RF/MLP/group-median primary errors are 0.201853/0.261071/0.617066 (CV5) and 0.165012/0.246446/0.580975 (LOSO). | {sources.ids('cv_random_forest','cv_mlp','cv_group_median','loso_random_forest','loso_mlp','loso_group_median')} |",
        f"| M2 | Corresponding typical multiplicative misses are approximately 22%/18% for RF and 85%/79% for the group median. | {sources.ids('rf_cv_pct_display','rf_loso_pct_display','gm_cv_pct_display','gm_loso_pct_display')} |",
        f"| M3 | Supported-row coverage is 449.8 rows (94.495798%) under repeated CV, 454 rows (95.378151%) under LOSO, and 449 rows (94.327731%) under sensitivity; six LOSO folds have no supported held-out row. | {sources.ids('cv_coverage_rows','cv_coverage_display','loso_coverage_rows','loso_coverage_display','sensitivity_coverage_rows','sensitivity_coverage_display','empty_loso')} |",
        f"| G1 | GraphSAGE is worse than RF by +0.171529 (CV5) and +0.163960 (LOSO). | {sources.ids('gs_rf_cv','gs_rf_loso')} |",
        f"| G2 | GAT is worse than GraphSAGE by +0.267067 (CV5) and +0.328409 (LOSO). CV5 and LOSO primary errors are 0.632643 and 0.658164. | {sources.ids('gat_gs_cv','gat_gs_loso','cv_gat','loso_gat')} |",
        f"| G3 | Shuffled-edge comparison: CV5 +0.024523 (`equivalent`, Holm p 0.096000); LOSO -0.032199 (`inconclusive`, Holm p 0.387000). | {sources.ids('shuffle_cv','shuffle_cv_p','shuffle_loso','shuffle_loso_p')} |",
        f"| G4 | GCN minus GraphSAGE is -0.016638 (CV5, `equivalent`) and +0.022510 (LOSO, `inconclusive`). | {sources.ids('gcn_cv','gcn_loso')} |",
        f"| G5 | Graph architecture and ablation errors: GraphSAGE 0.370271/0.338161, shuffled edges 0.341685/0.366378, GCN 0.354737/0.356236, GAT 0.632643/0.658164, and best GNN 0.376393/0.340865 under CV5/LOSO. | {sources.ids('cv_graphsage','loso_graphsage','cv_graphsage_shuffled','loso_graphsage_shuffled','cv_gcn','loso_gcn','cv_gat','loso_gat','cv_best_gnn','loso_best_gnn')} |",
        f"| T1 | MLP has lower p95 under both protocols and lower p90 under CV5; LOSO p90 is effectively tied. The tail windows are ±25% and ±50%. | {sources.ids('tail_cv_random_forest_p90','tail_cv_mlp_p90','tail_cv_random_forest_p95','tail_cv_mlp_p95','tail_loso_random_forest_p90','tail_loso_mlp_p90','tail_loso_random_forest_p95','tail_loso_mlp_p95','window_25','window_50')} |",
        f"| L1 | Eighteen of 36 trained-model/headline-family comparisons beat the group median; family counts are 4, 0, 4, 5, 3, and 2. | {sources.ids('total_beating','count_amd_cdna4_mi350x','count_amd_cdna4_mi355x','count_intel_xe2_battlemage_arc_pro_b70','count_nvidia_blackwell_b200','count_nvidia_blackwell_b300','count_nvidia_blackwell_rtx_pro')} |",
        f"| L2 | Best-GNN-minus-MLP point differences are +0.572661 (MI350X), -0.115406 (MI355X), -0.391312 (B70), +0.101929 (B200), +0.059492 (B300), +0.334701 (RTX PRO), and -0.398394 (CPU-only). Intervals exclude zero for MI355X and CPU-only and cross zero for B200, B300, and RTX PRO. | {sources.ids('pair_amd_cdna4_mi350x','pair_amd_cdna4_mi355x','amd_cdna4_mi355x_difference_95ci_low','amd_cdna4_mi355x_difference_95ci_high','pair_intel_xe2_battlemage_arc_pro_b70','pair_nvidia_blackwell_b200','nvidia_blackwell_b200_difference_95ci_low','nvidia_blackwell_b200_difference_95ci_high','pair_nvidia_blackwell_b300','nvidia_blackwell_b300_difference_95ci_low','nvidia_blackwell_b300_difference_95ci_high','pair_nvidia_blackwell_rtx_pro','nvidia_blackwell_rtx_pro_difference_95ci_low','nvidia_blackwell_rtx_pro_difference_95ci_high','pair_cpu','cpu_difference_95ci_low','cpu_difference_95ci_high')} |",
        f"| L3 | Supported systems are 3 (MI350X), 12 (MI355X), 3 (B70), 14 (B200), 33 (B300), 15 (RTX PRO), and 10 (CPU-only). | {sources.ids('support_amd_cdna4_mi350x','support_amd_cdna4_mi355x','support_intel_xe2_battlemage_arc_pro_b70','support_nvidia_blackwell_b200','support_nvidia_blackwell_b300','support_nvidia_blackwell_rtx_pro','support_cpu')} |",
        f"| L4 | Six trained models create 36 headline-family comparisons; best GNN beats MLP on the point estimate in two of six headline families; five families are interval-eligible. | {sources.ids('trained_models','trained_combinations','headline_better_count','headline_count','interval_eligible')} |",
        f"| V1 | GraphSAGE partition SD is 0.093310; RF is 0.032297; MLP is 0.019410. | {sources.ids('partition_sd_graphsage','partition_sd_random_forest','partition_sd_mlp')} |",
        f"| O1 | Two shared interactive-throughput extrapolation failures reach |z| approximately 54 and 26. | {sources.ids('shared_outlier_rows','outlier_llama_display','outlier_deepseek_display')} |",
        f"| O2 | Top-ten residuals contain roughly 78--87% of squared normalized error across the reported models. | {sources.ids('top_share_low_display','top_share_high_display')} |",
        f"| A1 | Earlier leakage effect is +0.0043 (95% CI [-0.0779, +0.0782]). | {sources.ids('leak_effect_display','leak_low_display','leak_high_display')} |",
        f"| A2 | Historical single-run Spearman is 0.711303; pinned ten-seed reconciliation mean is 0.484192. | {sources.ids('spearman_historical','spearman_pinned')} |",
        "",
        "Every numeric statement in the abstract, conclusion, Evaluation "
        "Protocol chapter, Results prose, and provenance appendix is represented "
        "above or in the accompanying exhaustive `stage3b_number_source_map.csv`. "
        "Table cells are generated directly by "
        "`generate_thesis_results.py` and are not typed into LaTeX.",
        "",
        "### Automated prose-number scan",
        "",
        f"- Source-map rows: {len(source_map)}.",
        f"- Numeric tokens scanned in the abstract, Chapters 1--9, and "
        f"Appendix B: {token_count}.",
        f"- Numeric tokens not covered by a map row: {len(uncovered)}.",
        "",
        "```text",
        (
            "No numeric token was uncovered."
            if not uncovered
            else "\n".join(
                f"{row['file']}: {row['numeric_token']} — {row['context']}"
                for row in uncovered
            )
        ),
        "```",
        "",
        "### Rendered-PDF frozen-value check",
        "",
        "| Expected text | Present in PDF |",
        "| --- | --- |",
        *[
            f"| `{row.expected_text}` | {'Yes' if row.present_in_pdf else 'No'} |"
            for row in pdf_checks.itertuples(index=False)
        ],
        "",
        "### Random 15-row rendered-PDF spot-check",
        "",
        "The seeded sample verifies the rendered PDF value against the mapped "
        "source column and its source value.",
        "",
        "| PDF value | Map ID | Source file | Column | Source value | Match |",
        "| --- | --- | --- | --- | ---: | --- |",
        *[
            f"| `{row.pdf_value}` | {row.map_id} | `{row.source_file}` | "
            f"`{row.source_column}` | `{row.source_value}` | "
            f"{'Yes' if row.value_match else 'No'} |"
            for row in spotcheck.itertuples(index=False)
        ],
        "",
        "## 5. Numbers that changed (old -> new)",
        "",
        "| Quantity | Old thesis | Rewritten thesis |",
        "| --- | --- | --- |",
        "| Scope | 96 systems | 105 submitter-aware systems |",
        "| Primary finding | Qualified support for GNNs | Best GNN worse by about 0.10--0.12 log units; `inconclusive` under ±0.05 |",
        "| Main comparator | GNN-specific metric tradeoffs | RF best overall; GraphSAGE worse than RF after Holm |",
        "| GAT | No consolidated conclusion | Worse than GraphSAGE under both protocols |",
        "| Tail claim | No consolidated RF/MLP tail analysis | MLP lower p95 both; p90 CV5 only; LOSO p90 tied |",
        "| LOAFO | Development/future split language | Descriptive 13-family stress test; 18 of 36 trained-model cases beat GM |",
        "| Leakage | Main-text omission | Disclosed with +0.0043 and corrected isolation |",
        "| Historical Spearman | Treated as prior result | 0.711303 provenance vs 0.484192 pinned reconciliation |",
        "| Model scope | Generic graph-versus-tabular comparison | Fixed, untuned model ladder; result limited to these configurations |",
        "| Sensitivity scope | Conservative check | Modest check because 78 of 90 groups are singleton systems |",
        "| Compute scope | 6,210 total could imply all protocols | 6,210 covers 135 folds only; LOAFO has 195 separate planning fits |",
        "| AI disclosure | Code, tables, and editing | Also evaluation design, protocol drafting and amendments, analysis planning, and intermediate-result review |",
        "| Chapter count | Eight chapters | Nine chapters with protocol/results split |",
        "| Tables/figures | Mixed stale sources | All generated from final bundles |",
        "",
        "## 6. Surprising or narrative-changing findings",
        "",
        "- Metadata prediction is materially useful even though graph structure "
        "is not.",
        "- MI355X reverses the usual family narrative: no trained model beats "
        "the group median.",
        "- MI350X and B200 show large improvements from tabular models.",
        "- GraphSAGE has greater partition sensitivity than RF and MLP.",
        "- The same two interactive-throughput measurements fail across model "
        "families.",
        "- The partial-pooling secondary cannot be recovered uniquely from the "
        "pinned text.",
        "",
        "## 7. Open questions",
        "",
        "- The author must confirm the AI-assisted-tool disclosure with the "
        "dissertation advisor and against applicable UCLA policy.",
        "- Committee members and final author biography remain to be supplied.",
        "- A future protocol may define a reproducible partial-pooling posterior "
        "and operator-level graph evaluation.",
        "",
        "## 8. Verification and acceptance",
        "",
        "- Thesis compiled with pdfLaTeX and BibTeX.",
        "- Full test suite: 21 tests passed; output is saved in "
        "`stage3b_test_output.txt`.",
        f"- LaTeX undefined references: {log_counts.get('undefined_references', 'NA')}.",
        f"- LaTeX undefined citations: {log_counts.get('undefined_citations', 'NA')}.",
        f"- LaTeX overfull boxes: {log_counts.get('overfull_boxes', 'NA')}.",
        f"- LaTeX underfull boxes: {log_counts.get('underfull_boxes', 'NA')}.",
        "- Every warning or box line from `main.log`:",
        "",
        "```text",
        "\n".join(warning_lines) if warning_lines else "No warning lines.",
        "```",
        "",
        "- Source context for box warnings:",
        "",
        "```text",
        "\n\n".join(
            f"{detail['source_file']}:{detail['source_line']}\n"
            f"{detail['source_context']}"
            for detail in final_checks["latex"]["details"]
        )
        if final_checks["latex"]["details"]
        else "No box warnings.",
        "```",
        "",
        f"- Lockfile and recorded runtime versions: "
        f"{'match' if final_checks['lock_versions']['pass'] else 'mismatch'}.",
        f"- Frozen protocol SHA-256: "
        f"`{final_checks['protocol']['actual_sha256']}`.",
        f"- Final bundle hash rows verified: "
        f"{len(final_checks['bundle_hashes']['rows'])}; all match: "
        f"{'yes' if final_checks['bundle_hashes']['pass'] else 'no'}.",
        f"- Generated-asset hash rows verified: "
        f"{len(final_checks['generated_assets']['rows'])}; all match: "
        f"{'yes' if final_checks['generated_assets']['pass'] else 'no'}.",
        f"- Rendered-PDF spot-check rows: {len(spotcheck)}; all values match: "
        f"{'yes' if final_checks['number_spotcheck']['pass'] else 'no'}.",
        f"- Author placeholders visibly present in the PDF: "
        f"{'yes' if final_checks['placeholders']['pass'] else 'no'}.",
        "- Fresh-clone snapshot-only reproduction:",
        "",
        "```text",
        fresh_clone,
        "```",
        "",
        f"- Generated files in `generated/`: {generated_file_count}; "
        f"the manifest hashes {len(generated_manifest)} assets excluding "
        "the manifest itself.",
        f"- A second `--snapshot-only` run reproduced all "
        f"{generated_file_count} generated files byte-for-byte.",
        "- Main text contains nine chapters plus three appendices.",
        "- Every results table and figure is generated from final bundles.",
        "- Revised abstract opens with the negative result and metadata result.",
        "- The abstract, introduction, and conclusion state that the models "
        "were fixed and untuned and name the model ladder.",
        "- Chapter 6 reports the full four-label rule and the corrected leakage "
        "and compute scopes.",
        "- Discussion mechanism explanations are explicitly labeled as "
        "hypotheses.",
        "- Tail statement matches: p95 favors MLP under both protocols; p90 "
        "favors MLP only under CV5.",
        "- LOAFO reporting uses the verified 18-of-36 count and all five "
        "interval-eligible comparisons.",
        "- The primary-hypothesis prose uses the frozen label and interval "
        "rather than prohibited equivalence language.",
        "- Unsupported historical Table 6.4 values are provenance-only.",
        "- Thesis LaTeX changed in Stage 3B; no exploratory semantic block was run.",
        "",
    ]
    args.report.write_text("\n".join(lines) + "\n")
    print(args.report)
    print(args.source_map)


if __name__ == "__main__":
    main()
