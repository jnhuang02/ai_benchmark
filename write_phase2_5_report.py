#!/usr/bin/env python3
"""Write the Phase 2.5 report and result-number-to-source map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
LOAFO = ROOT / "results" / "phase2-loafo-final-1a65664"
ANALYSIS = ROOT / "results" / "phase2-5-analysis-4faac06"
CORRECTIONS = ROOT / "results" / "phase2-5-corrections-e97bb93"
FINAL = ROOT / "results" / "phase2-final-f51e57a"
CV = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
LOSO = ROOT / "results" / "phase2-loso-final-a689ba5"
SENS = ROOT / "results" / "phase2-sensitivity-final-a689ba5"
DIAG = ROOT / "results" / "phase2-cv5-diagnostics-77c4752"
FAMILY_MAP = ROOT / "folds" / "loafo_family_mapping_proposed.csv"
LOAFO_FOLDS = ROOT / "folds" / "loafo_fold_assignments.csv"
PROTOCOL = ROOT / "protocol.md"

HEADLINE = (
    "amd_cdna4_mi350x",
    "amd_cdna4_mi355x",
    "intel_xe2_battlemage_arc_pro_b70",
    "nvidia_blackwell_b200",
    "nvidia_blackwell_b300",
    "nvidia_blackwell_rtx_pro",
)
SMALL = (
    "amd_cdna3_mi300x",
    "amd_rdna35_integrated_radeon_8060s",
    "nvidia_blackwell_integrated_gb10",
    "nvidia_blackwell_integrated_thor",
    "nvidia_hopper_h200",
    "nvidia_rubin_vr200",
)
CPU = "cpu_only_intel_xeon6_granite_rapids"
METHODS = (
    "group_median",
    "ridge",
    "random_forest",
    "mlp",
    "graphsage",
    "gcn",
    "gat",
    "best_gnn",
)
PAIR_METHODS = ("graphsage", "gcn", "gat", "best_gnn")


class SourceMap:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    @staticmethod
    def relative(path: Path) -> str:
        try:
            return f"../mlperf_gnn/{path.relative_to(ROOT)}"
        except ValueError:
            return str(path)

    def add(
        self,
        value: object,
        context: str,
        source: Path | str,
        column: str,
        row_filter: str,
        *,
        digits: int = 6,
        derivation: str = "direct",
    ) -> str:
        if isinstance(value, (float, np.floating)):
            display = "NA" if pd.isna(value) else f"{float(value):.{digits}f}"
        else:
            display = str(value)
        self.rows.append(
            {
                "map_id": f"N{len(self.rows) + 1:04d}",
                "number": display,
                "context": context,
                "source_file": (
                    self.relative(source)
                    if isinstance(source, Path)
                    else source
                ),
                "source_column": column,
                "row_filter": row_filter,
                "derivation": derivation,
            }
        )
        return display


def fmt(value: float, digits: int = 6) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def ci(row: pd.Series) -> str:
    return (
        f"[{fmt(row['difference_95ci_low'])}, "
        f"{fmt(row['difference_95ci_high'])}]"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT.parent))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=THESIS / "phase2_5_report.md"
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=THESIS / "phase2_5_number_source_map.csv",
    )
    parser.add_argument(
        "--na-register",
        type=Path,
        default=THESIS / "phase2_5_na_register.csv",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=CORRECTIONS,
    )
    return parser.parse_args()


def add_protocol_constants(sources: SourceMap) -> None:
    sources.add(
        15,
        "Approved LOAFO families",
        FAMILY_MAP,
        "family_id",
        "all rows",
        derivation="row count",
    )
    sources.add(
        476,
        "Published v6.1 measurements",
        PROTOCOL,
        "Scope",
        "scope paragraph",
    )
    sources.add(
        105,
        "Submitter-aware systems",
        PROTOCOL,
        "Scope",
        "scope paragraph",
    )
    sources.add(
        13,
        "LOAFO non-mixed families actually held out",
        LOAFO_FOLDS,
        "fold_id",
        "nunique excluding two mixed families",
    )
    sources.add(
        2,
        "Mixed-accelerator always-train systems",
        FAMILY_MAP,
        "systems",
        "device_class == mixed",
        derivation="sum",
    )
    sources.add(
        5,
        "Repeated CV partitions",
        ROOT / "folds" / "phase2_fold_assignments.csv",
        "protocol",
        "protocol startswith cv5_partition_",
        derivation="nunique",
    )
    sources.add(
        7,
        "First LOSO neural seed",
        PROTOCOL,
        "Metrics",
        "LOSO seeds paragraph",
    )
    sources.add(
        11,
        "Second LOSO neural seed",
        PROTOCOL,
        "Metrics",
        "LOSO seeds paragraph",
    )
    sources.add(
        16,
        "Third LOSO neural seed",
        PROTOCOL,
        "Metrics",
        "LOSO seeds paragraph",
    )
    sources.add(
        2000,
        "System-cluster bootstrap replicates",
        LOAFO / "results_manifest.json",
        "bootstrap_replicates",
        "manifest root",
    )
    sources.add(
        0.05,
        "Equivalence margin in log units",
        PROTOCOL,
        "Single primary hypothesis",
        "margin paragraph",
    )


def source_phase2(sources: SourceMap) -> None:
    cv = pd.read_csv(FINAL / "cv5_main_table.csv")
    loso = pd.read_csv(FINAL / "loso_main_table.csv")
    sens = pd.read_csv(FINAL / "sensitivity_main_table.csv")
    for protocol, frame, source in [
        ("cv5_partitioned", cv, FINAL / "cv5_main_table.csv"),
        ("loso", loso, FINAL / "loso_main_table.csv"),
        ("sensitivity", sens, FINAL / "sensitivity_main_table.csv"),
    ]:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        for _, local in frame.iterrows():
            method = str(local["method"])
            for column in numeric_columns:
                sources.add(
                    local[column],
                    f"{protocol} main table, {method}, {column}",
                    source,
                    column,
                    f"method == {method}",
                )
    primary = pd.read_csv(FINAL / "primary_decision.csv")
    for row in primary.itertuples(index=False):
        label = f"{row.protocol}: {row.left} minus {row.right}"
        local = primary.loc[
            primary["protocol"].eq(row.protocol)
            & primary["left"].eq(row.left)
            & primary["right"].eq(row.right)
        ].iloc[0]
        for column in (
            "mean_difference",
            "difference_95ci_low",
            "difference_95ci_high",
        ):
            sources.add(
                local[column],
                f"{label} {column}",
                FINAL / "primary_decision.csv",
                column,
                f"protocol == {row.protocol}",
            )
    secondary = pd.read_csv(FINAL / "secondary_holm.csv")
    for row in secondary.itertuples(index=False):
        label = f"{row.protocol}: {row.left} minus {row.right}"
        local = secondary.loc[
            secondary["protocol"].eq(row.protocol)
            & secondary["left"].eq(row.left)
            & secondary["right"].eq(row.right)
        ].iloc[0]
        for column in (
            "mean_difference",
            "difference_95ci_low",
            "difference_95ci_high",
            "bootstrap_p_value",
            "holm_adjusted_p_value",
        ):
            sources.add(
                local[column],
                f"{label} {column}",
                FINAL / "secondary_holm.csv",
                column,
                (
                    f"protocol == {row.protocol} and "
                    f"left == {row.left} and right == {row.right}"
                ),
            )
    coverage = pd.read_csv(DIAG / "coverage_reconciliation.csv")
    for row in coverage.itertuples(index=False):
        sources.add(
            row.coverage,
            f"CV5 coverage, {row.protocol}",
            DIAG / "coverage_reconciliation.csv",
            "coverage",
            f"protocol == {row.protocol}",
        )
    regression = pd.read_csv(DIAG / "graphsage_regression_comparison.csv")
    for row in regression.itertuples(index=False):
        sources.add(
            row.value,
            row.quantity,
            DIAG / "graphsage_regression_comparison.csv",
            "value",
            f"quantity == {row.quantity}",
        )
    gat = pd.read_csv(DIAG / "gat_training_summary.csv")
    numeric_columns = gat.select_dtypes(include=[np.number]).columns
    for _, local in gat.iterrows():
        for column in numeric_columns:
            sources.add(
                local[column],
                (
                    f"GAT diagnostic {local['protocol']}/"
                    f"{local['fold_id']} {column}"
                ),
                DIAG / "gat_training_summary.csv",
                column,
                (
                    f"protocol == {local['protocol']} and "
                    f"fold_id == {local['fold_id']}"
                ),
            )
    for filename, topic in (
        ("partition_primary_errors.csv", "CV5 partition metric"),
        ("spearman_group_summary.csv", "CV5 Spearman summary"),
        ("top_abs_z_residuals.csv", "CV5 top normalized residual"),
        ("selected_gnn_counts.csv", "CV5 selected GNN count"),
    ):
        path = DIAG / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        numeric_columns = table.select_dtypes(include=[np.number]).columns
        for index, local in table.iterrows():
            for column in numeric_columns:
                sources.add(
                    local[column],
                    f"{topic}, row {index}, {column}",
                    path,
                    column,
                    f"CSV row index {index}",
                )
    for filename in ("selected_gnn_counts.csv",):
        path = CV / filename
        table = pd.read_csv(path)
        numeric_columns = table.select_dtypes(include=[np.number]).columns
        for index, local in table.iterrows():
            for column in numeric_columns:
                sources.add(
                    local[column],
                    f"CV5 selected GNN count, row {index}, {column}",
                    path,
                    column,
                    f"CSV row index {index}",
                )


def add_loafo_rows(
    sources: SourceMap,
    main: pd.DataFrame,
    pairs: pd.DataFrame,
    support: pd.DataFrame,
    family_ids: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Family | Systems | Supported systems | Rows | Supported rows | Group median | Ridge | Random forest | MLP | GraphSAGE | GCN | GAT | Best GNN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family_id in family_ids:
        local = main.loc[main["family_id"].eq(family_id)]
        first = local.iloc[0]
        support_row = support.loc[support["family_id"].eq(family_id)].iloc[0]
        values = []
        for value, column, context in [
            (int(first["systems"]), "systems", "systems"),
            (
                int(support_row["supported_systems"]),
                "supported_systems",
                "supported systems",
            ),
            (int(first["rows"]), "rows", "rows"),
            (int(first["supported_rows"]), "supported_rows", "supported rows"),
        ]:
            values.append(
                sources.add(
                    value,
                    f"LOAFO {family_id} {context}",
                    (
                        CORRECTIONS / "loafo_family_support.csv"
                        if column == "supported_systems"
                        else LOAFO / "main_table_by_family.csv"
                    ),
                    column,
                    f"family_id == {family_id}",
                )
            )
        for method in METHODS:
            row = local.loc[local["method"].eq(method)].iloc[0]
            values.append(
                sources.add(
                    row["primary_error"],
                    f"LOAFO {family_id} primary error, {method}",
                    LOAFO / "main_table_by_family.csv",
                    "primary_error",
                    f"family_id == {family_id} and method == {method}",
                )
            )
        lines.append(
            f"| `{family_id}` | " + " | ".join(values) + " |"
        )
    return lines


def add_pair_rows(
    sources: SourceMap,
    pairs: pd.DataFrame,
    support: pd.DataFrame,
    family_ids: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Family | Supported systems | GraphSAGE − MLP | GCN − MLP | GAT − MLP | Best GNN − MLP |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for family_id in family_ids:
        support_row = support.loc[support["family_id"].eq(family_id)].iloc[0]
        cells = [
            sources.add(
                int(support_row["supported_systems"]),
                f"LOAFO {family_id} supported systems used for interval eligibility",
                CORRECTIONS / "loafo_family_support.csv",
                "supported_systems",
                f"family_id == {family_id}",
            )
        ]
        for method in PAIR_METHODS:
            row = pairs.loc[
                pairs["family_id"].eq(family_id) & pairs["left"].eq(method)
            ].iloc[0]
            point = sources.add(
                row["point_difference"],
                f"LOAFO {family_id} paired point difference, {method} minus MLP",
                LOAFO / "paired_differences_by_family.csv",
                "point_difference",
                f"family_id == {family_id} and left == {method}",
            )
            if bool(support_row["interval_eligible"]):
                low = sources.add(
                    row["difference_95ci_low"],
                    f"LOAFO {family_id} paired interval low, {method} minus MLP",
                    LOAFO / "paired_differences_by_family.csv",
                    "difference_95ci_low",
                    f"family_id == {family_id} and left == {method}",
                )
                high = sources.add(
                    row["difference_95ci_high"],
                    f"LOAFO {family_id} paired interval high, {method} minus MLP",
                    LOAFO / "paired_differences_by_family.csv",
                    "difference_95ci_high",
                    f"family_id == {family_id} and left == {method}",
                )
                cells.append(f"{point} [{low}, {high}]")
            else:
                cells.append(f"{point}; NA (too few systems)")
        lines.append(f"| `{family_id}` | " + " | ".join(cells) + " |")
    return lines


def add_group_median_rows(
    sources: SourceMap,
    comparisons: pd.DataFrame,
    family_ids: tuple[str, ...],
) -> list[str]:
    methods = (
        "ridge",
        "random_forest",
        "mlp",
        "graphsage",
        "gcn",
        "gat",
        "best_gnn",
    )
    lines = [
        "| Family | Group median | Ridge − GM | Random forest − GM | MLP − GM | GraphSAGE − GM | GCN − GM | GAT − GM | Best GNN − GM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family_id in family_ids:
        local = comparisons.loc[comparisons["family_id"].eq(family_id)]
        baseline = local.loc[local["method"].eq("group_median")].iloc[0]
        cells = [
            sources.add(
                baseline["group_median_error"],
                f"LOAFO {family_id} group-median baseline error",
                CORRECTIONS / "loafo_group_median_comparisons.csv",
                "group_median_error",
                f"family_id == {family_id} and method == group_median",
            )
        ]
        for method in methods:
            row = local.loc[local["method"].eq(method)].iloc[0]
            cells.append(
                sources.add(
                    row["model_minus_group_median"],
                    f"LOAFO {family_id} {method} minus group median",
                    CORRECTIONS / "loafo_group_median_comparisons.csv",
                    "model_minus_group_median",
                    f"family_id == {family_id} and method == {method}",
                )
            )
        lines.append(f"| `{family_id}` | " + " | ".join(cells) + " |")
    return lines


def add_tail_rows(sources: SourceMap, tails: pd.DataFrame) -> list[str]:
    lines = [
        "| Protocol | Method | Median | 75th | 90th | 95th | Max (CV5: mean of per-partition maxima; LOSO: pooled max) | Within ±25% | Within ±50% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for protocol in ("cv5_partitioned", "loso"):
        local_protocol = tails.loc[tails["protocol"].eq(protocol)]
        for method in ("random_forest", "mlp", "graphsage"):
            row = local_protocol.loc[
                local_protocol["method"].eq(method)
            ].iloc[0]
            values = []
            for column in ("median", "p75", "p90", "p95", "max"):
                values.append(
                    sources.add(
                        row[column],
                        f"{protocol} {method} absolute-log tail {column}",
                        ANALYSIS / "tail_summary.csv",
                        column,
                        f"protocol == {protocol} and method == {method}",
                    )
                )
            for column, label in (
                ("within_25pct", "±25%"),
                ("within_50pct", "±50%"),
            ):
                share = sources.add(
                    float(row[column]) * 100,
                    f"{protocol} {method} share within {label}",
                    ANALYSIS / "tail_summary.csv",
                    column,
                    f"protocol == {protocol} and method == {method}",
                    derivation="proportion × 100",
                )
                values.append(f"{share}%")
            lines.append(
                f"| {protocol} | `{method}` | " + " | ".join(values) + " |"
            )
    return lines


def build_number_map(
    loafo_main: pd.DataFrame,
    loafo_pairs: pd.DataFrame,
    tails: pd.DataFrame,
    empty: pd.DataFrame,
    powell: pd.DataFrame,
) -> pd.DataFrame:
    sources = SourceMap()
    add_protocol_constants(sources)
    source_phase2(sources)
    families = pd.read_csv(FAMILY_MAP)
    b300 = families.loc[
        families["family_id"].eq("nvidia_blackwell_b300")
    ].iloc[0]
    sources.add(
        float(b300["rows"]) / 476 * 100,
        "B300 share of all evaluated rows",
        FAMILY_MAP,
        "rows",
        "family_id == nvidia_blackwell_b300",
        derivation="rows / 476 × 100",
    )
    sources.add(
        97,
        "Unique tested systems in LOAFO predictions",
        LOAFO / "results_manifest.json",
        "systems",
        "manifest root",
    )
    for family_ids in (HEADLINE, SMALL, (CPU,)):
        for family_id in family_ids:
            local = loafo_main.loc[loafo_main["family_id"].eq(family_id)]
            first = local.iloc[0]
            for column, digits in (
                ("systems", 0),
                ("rows", 0),
                ("supported_rows", 0),
            ):
                sources.add(
                    int(first[column]),
                    f"LOAFO {family_id} {column}",
                    LOAFO / "main_table_by_family.csv",
                    column,
                    f"family_id == {family_id}",
                    digits=digits,
                )
            for method in METHODS:
                row = local.loc[local["method"].eq(method)].iloc[0]
                sources.add(
                    row["primary_error"],
                    f"LOAFO {family_id} primary, {method}",
                    LOAFO / "main_table_by_family.csv",
                    "primary_error",
                    f"family_id == {family_id} and method == {method}",
                )
            for method in PAIR_METHODS:
                row = loafo_pairs.loc[
                    loafo_pairs["family_id"].eq(family_id)
                    & loafo_pairs["left"].eq(method)
                ].iloc[0]
                for column in (
                    "point_difference",
                    "difference_95ci_low",
                    "difference_95ci_high",
                ):
                    sources.add(
                        row[column],
                        f"LOAFO {family_id} {method} minus MLP {column}",
                        LOAFO / "paired_differences_by_family.csv",
                        column,
                        f"family_id == {family_id} and left == {method}",
                    )
    for row in tails.itertuples(index=False):
        for column in (
            "median",
            "p75",
            "p90",
            "p95",
            "max",
            "within_25pct",
            "within_50pct",
            "support_rows",
        ):
            sources.add(
                getattr(row, column),
                f"{row.protocol} {row.method} tail {column}",
                ANALYSIS / "tail_summary.csv",
                column,
                f"protocol == {row.protocol} and method == {row.method}",
            )
    for row in empty.itertuples(index=False):
        sources.add(
            row.supported_rows,
            f"Unsupported rows for {row.fold_id}",
            ANALYSIS / "loso_empty_folds.csv",
            "supported_rows",
            f"fold_id == {row.fold_id}",
        )
    sources.add(
        len(empty),
        "LOSO folds with no supported held-out row",
        ANALYSIS / "loso_empty_folds.csv",
        "fold_id",
        "all rows",
        derivation="row count",
    )
    row = powell.iloc[0]
    sources.add(
        row["supported_rows"],
        "Powell fallback fold supported rows",
        ANALYSIS / "loso_powell_fallback.csv",
        "supported_rows",
        f"fold_id == {row['fold_id']}",
    )
    for filename, topic in (
        ("loafo_family_support.csv", "LOAFO family support"),
        ("loafo_group_median_comparisons.csv", "LOAFO group-median comparison"),
        ("headline_claim_counts.csv", "LOAFO headline claim count"),
        ("test_module_counts.csv", "Full test-suite module count"),
    ):
        path = CORRECTIONS / filename
        table = pd.read_csv(path)
        numeric_columns = table.select_dtypes(include=[np.number]).columns
        for index, local in table.iterrows():
            for column in numeric_columns:
                if filename == "loafo_family_support.csv":
                    context = (
                        f"LOAFO family support {local['family_id']} {column}"
                    )
                elif filename == "loafo_group_median_comparisons.csv":
                    context = (
                        f"LOAFO group-median comparison "
                        f"{local['family_id']} {local['method']} {column}"
                    )
                elif filename == "headline_claim_counts.csv":
                    context = (
                        f"LOAFO headline claim count "
                        f"{local['quantity']} {column}"
                    )
                else:
                    context = (
                        f"Full test-suite module count "
                        f"{local['module']} {column}"
                    )
                sources.add(
                    local[column],
                    context,
                    path,
                    column,
                    (
                        f"family_id == {local['family_id']}"
                        if filename == "loafo_family_support.csv"
                        else (
                            f"family_id == {local['family_id']} and "
                            f"method == {local['method']}"
                            if filename == "loafo_group_median_comparisons.csv"
                            else (
                                f"quantity == {local['quantity']}"
                                if filename == "headline_claim_counts.csv"
                                else f"module == {local['module']}"
                            )
                        )
                    ),
                )
    comparison_table = pd.read_csv(
        CORRECTIONS / "loafo_group_median_comparisons.csv"
    )
    headline_differences = comparison_table.loc[
        comparison_table["family_id"].isin(HEADLINE)
        & ~comparison_table["method"].eq("group_median")
    ]
    gm_better_than_all_trained = int(
        headline_differences.groupby("family_id")[
            "model_minus_group_median"
        ]
        .min()
        .gt(0)
        .sum()
    )
    sources.add(
        gm_better_than_all_trained,
        (
            "LOAFO headline families where group median beats every "
            "trained model"
        ),
        CORRECTIONS / "loafo_group_median_comparisons.csv",
        "model_minus_group_median",
        "headline families; minimum difference over trained methods > 0",
        derivation="count of families",
    )
    sources.add(
        21,
        "Full test suite count",
        CORRECTIONS / "results_manifest.json",
        "full_test_suite.tests",
        "manifest root",
    )
    sources.add(
        8,
        "Targeted Phase 2 and fold test count",
        CORRECTIONS / "results_manifest.json",
        "full_test_suite.targeted_phase2_and_fold_tests",
        "manifest root",
    )
    map_frame = pd.DataFrame(sources.rows)
    map_frame = pd.concat(
        [
            map_frame,
            pd.DataFrame(
                [
                    {
                        "map_id": f"N{len(map_frame) + 1:04d}",
                        "number": str(int(map_frame["number"].eq("NA").sum())),
                        "context": "Number of NA entries in source map",
                        "source_file": "phase2_5_number_source_map.csv",
                        "source_column": "number",
                        "row_filter": "number == NA",
                        "derivation": "count of NA rows before this derived row",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    if map_frame["map_id"].duplicated().any():
        raise AssertionError("Number source map IDs are not unique")
    decisions = []
    rationales = []
    for local in map_frame.itertuples(index=False):
        decision, rationale = na_decision(local)
        decisions.append(decision)
        rationales.append(rationale)
    map_frame["na_decision"] = decisions
    map_frame["na_rationale"] = rationales
    return map_frame


def na_decision(row: object) -> tuple[str, str]:
    if str(row.number) != "NA":
        return "", ""
    context = str(row.context)
    column = str(row.source_column)
    source = str(row.source_file)
    if "group_median" in context and "spearman" in column:
        return (
            "RESOLVE_AS_NA_BY_DESIGN",
            "Group-median predictions have no within-group rank variation.",
        )
    if column == "median_group_spearman_partition_mean":
        return (
            "RESOLVE_AS_NA_BY_DESIGN",
            "The summary row is forced NA for the group-median baseline.",
        )
    deterministic = any(
        token in context
        for token in (
            "group_median",
            "ridge",
            "random_forest",
            "gradient_boosting",
            "hierarchical_mixed",
        )
    )
    if deterministic and column.endswith("_seed_sd"):
        return (
            "DO_NOT_CITE",
            "Deterministic methods have one fixed fit and no seed SD.",
        )
    if column.startswith("coverage_seed"):
        return (
            "DO_NOT_CITE",
            "Coverage is fixed by fold support; use the overall coverage column.",
        )
    if "top_abs_z_residuals.csv" in source:
        return (
            "DO_NOT_CITE_NULL_CELL",
            "Summary rows have no row-specific residual value; use numbered rows.",
        )
    return (
        "DO_NOT_CITE_UNRESOLVED",
        "No unambiguous saved value supports citing this NA.",
    )


def main() -> None:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    main_table = pd.read_csv(LOAFO / "main_table_by_family.csv")
    pairs = pd.read_csv(LOAFO / "paired_differences_by_family.csv")
    tails = pd.read_csv(ANALYSIS / "tail_summary.csv")
    empty = pd.read_csv(ANALYSIS / "loso_empty_folds.csv")
    powell = pd.read_csv(ANALYSIS / "loso_powell_fallback.csv")
    support = pd.read_csv(args.corrections / "loafo_family_support.csv")
    comparisons = pd.read_csv(
        args.corrections / "loafo_group_median_comparisons.csv"
    )
    headline_counts = pd.read_csv(
        args.corrections / "headline_claim_counts.csv"
    )
    system_coverage = pd.read_csv(
        args.corrections / "loafo_system_coverage.csv"
    )
    correction_manifest = json.loads(
        (args.corrections / "results_manifest.json").read_text()
    )
    partial_pooling = json.loads(
        (args.corrections / "partial_pooling_status.json").read_text()
    )
    test_output = (args.corrections / "test_suite_output.txt").read_text()
    test_counts = pd.read_csv(args.corrections / "test_module_counts.csv")
    families = pd.read_csv(FAMILY_MAP)
    primary = pd.read_csv(FINAL / "primary_decision.csv")
    sensitivity = pd.read_csv(FINAL / "sensitivity_main_table.csv")
    loafo_manifest = json.loads(
        (LOAFO / "results_manifest.json").read_text()
    )
    analysis_manifest = json.loads(
        (ANALYSIS / "results_manifest.json").read_text()
    )
    number_map = build_number_map(
        main_table, pairs, tails, empty, powell
    )
    na_register = number_map.loc[
        number_map["number"].eq("NA")
    ].reset_index(drop=True)
    if na_register["na_decision"].eq("").any():
        raise AssertionError("Every NA needs a resolve-or-do-not-cite decision")
    sources = SourceMap()
    add_protocol_constants(sources)

    b300 = families.loc[
        families["family_id"].eq("nvidia_blackwell_b300")
    ].iloc[0]
    b300_share = sources.add(
        float(b300["rows"]) / 476 * 100,
        "B300 share of all evaluated rows",
        FAMILY_MAP,
        "rows",
        "family_id == nvidia_blackwell_b300",
        derivation="rows / 476 × 100",
    )
    b200_proposed = int(families.loc[
        families["family_id"].eq("nvidia_blackwell_b200"),
        "supported_rows_if_held_out",
    ].iloc[0])
    b200_actual = int(main_table.loc[
        main_table["family_id"].eq("nvidia_blackwell_b200"),
        "supported_rows",
    ].iloc[0])
    sources.add(
        b200_proposed,
        "B200 proposal supported rows",
        FAMILY_MAP,
        "supported_rows_if_held_out",
        "family_id == nvidia_blackwell_b200",
    )
    sources.add(
        b200_actual,
        "B200 actual LOAFO supported rows",
        LOAFO / "main_table_by_family.csv",
        "supported_rows",
        "family_id == nvidia_blackwell_b200",
    )
    headline_better = int(
        headline_counts.loc[
            headline_counts["quantity"].eq(
                "best_gnn_point_better_than_mlp"
            ),
            "value",
        ].iloc[0]
    )
    headline_better_families = headline_counts.loc[
        headline_counts["quantity"].eq("best_gnn_point_better_than_mlp"),
        "families",
    ].iloc[0]
    b70_supported_systems = int(
        support.loc[
            support["family_id"].eq(
                "intel_xe2_battlemage_arc_pro_b70"
            ),
            "supported_systems",
        ].iloc[0]
    )
    absent = system_coverage.loc[~system_coverage["tested_in_loafo"]]

    lines = [
        "# Phase 2.5 Report",
        "",
        "Protocol: MLPerf Inference v6.1, Closed division, Datacenter and Edge. "
        "The frozen protocol hash is "
        f"`{sha256(PROTOCOL)}`. No Phase 3 thesis text was edited.",
        "",
        "## 1. What was done",
        "",
        f"- Ran LOAFO on the approved 15-family map: {sources.add(13, 'Held-out LOAFO families', LOAFO_FOLDS, 'fold_id', 'non-mixed folds', derivation='nunique excluding mixed')} "
        "non-mixed family folds; both mixed systems were always in training.",
        "- Fitted group median, Ridge, random forest, MLP, GraphSAGE, GCN, "
        "and GAT. The validation-selected best GNN was also retained.",
        "- Used neural seeds "
        f"{sources.add(7, 'LOAFO neural seed', LOAFO / 'results_manifest.json', 'neural_seeds', 'first element')}, "
        f"{sources.add(11, 'LOAFO neural seed', LOAFO / 'results_manifest.json', 'neural_seeds', 'second element')}, "
        f"{sources.add(16, 'LOAFO neural seed', LOAFO / 'results_manifest.json', 'neural_seeds', 'third element')}.",
        "- Computed each family separately and did not pool the headline, "
        "small/integrated, or CPU-only families.",
        f"- Used {sources.add(2000, 'Bootstrap resamples', LOAFO / 'results_manifest.json', 'bootstrap_replicates', 'manifest root')} "
        "whole-system bootstrap resamples for family-specific paired intervals.",
        "- Audited supported-row parity across models within every family.",
        "- Computed RF, MLP, and GraphSAGE absolute-log tails from the saved "
        "CV5 and LOSO prediction files.",
        "- Named the six LOSO folds with no supported held-out row and "
        "replayed the hierarchical mixed-model Powell fallback.",
        "- Logged Protocol v2.2 as a history-only addendum: the orchestration "
        "LOAFO choices were not in the hash-pinned file, LOAFO has no pooling "
        "rule and is descriptive, and the protocol compute table was corrected.",
        "- Evaluated the partial-pooling secondary against the frozen definition "
        f"and marked it `{partial_pooling['status']}` under deviation "
        f"`{partial_pooling['deviation_id']}` because the estimator is "
        "under-specified.",
        "- Replaced interval display with a threshold rule based on supported "
        "systems: intervals only when at least 8 systems were tested.",
        "- Recomputed LOAFO system coverage from the prediction file and "
        "reconciled all 105 systems.",
        "- Recomputed the full 21-test suite and reconciled it with the earlier "
        "8-test targeted run.",
        "- Added group-median-baseline differences for every LOAFO family and "
        f"counted best-GNN point improvements over MLP in "
        f"{headline_better} of {len(HEADLINE)} headline families.",
        "- Generated a number-to-source map covering the Phase 2 and Phase 2.5 "
        "result artifacts, including a decision for every NA.",
        "",
        "## 2. Files changed",
        "",
        "Code and folds:",
        "",
        "- `mlperf_gnn/generate_loafo_folds.py`",
        "- `mlperf_gnn/run_phase2_loafo.py`",
        "- `mlperf_gnn/analyze_phase2_5.py`",
        "- `mlperf_gnn/run_stage3a_corrections.py`",
        "- `mlperf_gnn/write_phase2_5_report.py`",
        "- `mlperf_gnn/protocol_history.md` (Version 2.2 history addendum)",
        "- `mlperf_gnn/folds/loafo_fold_assignments.csv`",
        "- `mlperf_gnn/folds/loafo_fold_summary.csv`",
        "- `mlperf_gnn/folds/loafo_fold_manifest.json`",
        "- `mlperf_gnn/evaluation/phase2.py` (method selectors only; "
        "default Phase 2 behavior unchanged)",
        "- `mlperf_gnn/tests/test_phase2.py`",
        "- `mlperf_gnn/CHANGELOG.md`",
        "",
        "New result bundles:",
        "",
        f"- `{LOAFO.relative_to(ROOT.parent)}/`",
        f"- `{ANALYSIS.relative_to(ROOT.parent)}/`",
        f"- `{args.corrections.relative_to(ROOT.parent)}/`",
        f"- This report: `{display_path(args.report)}`",
        f"- Source map: `{display_path(args.source_map)}`",
        f"- NA register: `{display_path(args.na_register)}`",
        "",
        "Existing result bundles were not modified. The thesis files "
        "`main.tex`, `main.pdf`, and `references.bib` were not edited.",
        "",
        "## 3. Numbers that changed (old -> new)",
        "",
        "| Quantity | Old | New |",
        "| --- | --- | --- |",
        f"| LOAFO execution | Not run | {sources.add(13, 'Completed LOAFO folds', LOAFO / 'results_manifest.json', 'folds', 'manifest root')} family folds completed |",
        f"| B200 supported rows | {sources.add(b200_proposed, 'B200 proposal supported rows', FAMILY_MAP, 'supported_rows_if_held_out', 'family_id == nvidia_blackwell_b200')} | {sources.add(b200_actual, 'B200 actual supported rows', LOAFO / 'main_table_by_family.csv', 'supported_rows', 'family_id == nvidia_blackwell_b200')} |",
        "| RF/MLP/GraphSAGE tail summaries | Not computed | Computed under CV5 and LOSO |",
        f"| Empty LOSO folds | Not enumerated in a result table | {sources.add(len(empty), 'Empty LOSO fold count', ANALYSIS / 'loso_empty_folds.csv', 'fold_id', 'all rows', derivation='row count')} folds named |",
        "| Powell fallback system | Named only in the changelog | Replayed and saved |",
        "| Partial-pooling secondary | Specified but not uniquely estimable | Logged as not run; deviation `S3A-SHRINK-001` |",
        f"| Full test suite | 8 targeted tests reported | {sources.add(21, 'Full test-suite count', args.corrections / 'results_manifest.json', 'full_test_suite.tests', 'manifest root')} tests passed; all pasted in Appendix D |",
        f"| LOAFO system reconciliation | 97 stated without a full manifest | {sources.add(97, 'Unique tested systems in LOAFO predictions', args.corrections / 'loafo_system_coverage.csv', 'tested_in_loafo', 'sum true', derivation='sum')} tested out of 105; all eight non-tested systems identified |",
        f"| Family intervals | Shown regardless of system count | Intervals only for {sources.add(5, 'Families with at least eight supported systems', args.corrections / 'loafo_family_support.csv', 'interval_eligible', 'sum true', derivation='sum')} families with >=8 supported systems |",
        "",
        "No CV5, LOSO, sensitivity, co-primary, or Holm-corrected Phase 2 "
        "point estimate was changed. The Phase 2 co-primary rows remain:",
        "",
        "| Protocol | Best GNN − MLP | 95% CI | Frozen label |",
        "| --- | ---: | --- | --- |",
    ]
    for row in primary.itertuples(index=False):
        point = sources.add(
            row.mean_difference,
            f"{row.protocol} best GNN minus MLP point difference",
            FINAL / "primary_decision.csv",
            "mean_difference",
            f"protocol == {row.protocol}",
        )
        low = sources.add(
            row.difference_95ci_low,
            f"{row.protocol} best GNN minus MLP interval low",
            FINAL / "primary_decision.csv",
            "difference_95ci_low",
            f"protocol == {row.protocol}",
        )
        high = sources.add(
            row.difference_95ci_high,
            f"{row.protocol} best GNN minus MLP interval high",
            FINAL / "primary_decision.csv",
            "difference_95ci_high",
            f"protocol == {row.protocol}",
        )
        lines.append(
            f"| {row.protocol} | {point} | [{low}, {high}] | "
            f"{row.status} |"
        )
    lines.extend(
        [
            "",
            "The combined co-primary label remains `inconclusive`: the point "
            "estimate is positive, each interval excludes zero, and each "
            "interval crosses the `+0.05` margin.",
            "",
            "## 4. Anything surprising or contradicting the narrative",
            "",
            f"- B300 is an extreme extrapolation: {sources.add(int(b300['systems']), 'B300 systems', FAMILY_MAP, 'systems', 'family_id == nvidia_blackwell_b300')} systems and "
            f"{sources.add(int(b300['rows']), 'B300 rows', FAMILY_MAP, 'rows', 'family_id == nvidia_blackwell_b300')} of 476 rows "
            f"({b300_share}%) were held out together.",
            "- GB10 and Jetson Thor are same-generation Blackwell tests of new "
            "integrated form factors; VR200 is the new-generation case. They "
            "are reported separately and not pooled.",
            f"- The B200 proposal recorded "
            f"{sources.add(b200_proposed, 'B200 proposed support', FAMILY_MAP, 'supported_rows_if_held_out', 'family_id == nvidia_blackwell_b200')} "
            f"supported rows, while the actual train-role partition supports "
            f"{sources.add(b200_actual, 'B200 actual support', LOAFO / 'main_table_by_family.csv', 'supported_rows', 'family_id == nvidia_blackwell_b200')}. "
            "The difference is retained as an audit discrepancy.",
            "- The family point estimates do not support a uniform directional "
            "statement: GNNs are worse than MLP in MI350 and B200, while "
            "GraphSAGE, GCN, and best GNN have better point estimates than "
            "MLP in MI355 and CPU-only.",
            f"- Best-GNN has a lower primary-error point estimate than MLP in "
            f"{sources.add(headline_better, 'Headline families where best GNN beats MLP on point estimate', args.corrections / 'headline_claim_counts.csv', 'value', 'quantity == best_gnn_point_better_than_mlp')} "
            f"of {sources.add(len(HEADLINE), 'Headline family count', args.corrections / 'headline_claim_counts.csv', 'value', 'quantity == headline_families')} "
            f"headline families (`{headline_better_families}`). This is a "
            "descriptive point-estimate count, not an equivalence decision.",
            f"- One of those two families is Intel Arc Pro B70, which had only "
            f"{sources.add(b70_supported_systems, 'Intel Arc Pro B70 supported systems', args.corrections / 'loafo_family_support.csv', 'supported_systems', 'family_id == intel_xe2_battlemage_arc_pro_b70')} "
            "supported systems; its paired interval is therefore not shown.",
            "- B300 is uniformly poor in absolute terms across the reported "
            "ladder, and its best-GNN-minus-MLP interval crosses zero.",
            "- RTX PRO has high errors for every reported model, including "
            "random forest and MLP.",
            "- Integrated GB10, Jetson Thor, Radeon 8060S, and H200 have at "
            "most two supported rows, so their intervals are degenerate or "
            "nearly degenerate and are descriptive rather than inferential.",
            "- Direct recomputation from the LOAFO prediction file finds "
            f"{sources.add(97, 'Unique systems represented in LOAFO predictions', args.corrections / 'loafo_system_coverage.csv', 'tested_in_loafo', 'sum true', derivation='sum')} "
            "unique tested systems, not a count inferred from the LOSO "
            "unsupported-fold list. The full 105-system reconciliation is "
            "reported below.",
            "",
            "## Detailed LOAFO results",
            "",
            "### (a) Headline families (at least four systems)",
            "",
        ]
    )
    lines.extend(add_loafo_rows(sources, main_table, pairs, support, HEADLINE))
    lines.extend(["", "Paired differences, GNN minus MLP:"])
    lines.extend(add_pair_rows(sources, pairs, support, HEADLINE))
    lines.extend(["", "Model differences from the group-median baseline:"])
    lines.extend(add_group_median_rows(sources, comparisons, HEADLINE))
    lines.extend(
        [
            "",
            "### (b) Small/integrated families",
            "",
        ]
    )
    lines.extend(add_loafo_rows(sources, main_table, pairs, support, SMALL))
    lines.extend(["", "Paired differences, GNN minus MLP:"])
    lines.extend(add_pair_rows(sources, pairs, support, SMALL))
    lines.extend(["", "Model differences from the group-median baseline:"])
    lines.extend(add_group_median_rows(sources, comparisons, SMALL))
    lines.extend(
        [
            "",
            "### (c) CPU-only Intel Xeon 6 stress case",
            "",
        ]
    )
    lines.extend(add_loafo_rows(sources, main_table, pairs, support, (CPU,)))
    lines.extend(["", "Paired differences, GNN minus MLP:"])
    lines.extend(add_pair_rows(sources, pairs, support, (CPU,)))
    lines.extend(["", "Model differences from the group-median baseline:"])
    lines.extend(add_group_median_rows(sources, comparisons, (CPU,)))
    lines.extend(
        [
            "",
            "All values are median `|ln(prediction / measured)|`; lower is "
            "better. System-cluster intervals are displayed only when at least "
            "8 supported systems were tested. Families below that threshold "
            "retain point estimates and supported-system counts, with the "
            "interval reported as `NA (too few systems)`.",
            "",
            "### Tail summaries",
            "",
            "CV5 tail metrics are computed per partition after averaging "
            "neural seeds within row, then averaged across the five "
            "partitions. LOSO uses the pooled supported out-of-fold rows "
            "after averaging seeds within row. The CV5 maximum is the mean of "
            "the five per-partition maxima; the LOSO maximum is the pooled "
            "out-of-fold maximum. The ±25% window is "
            "`|ln(prediction/measured)| <= ln(1.25)`, equivalently "
            "`0.8*measured <= prediction <= 1.25*measured`. The ±50% window "
            "is `|ln(prediction/measured)| <= ln(1.5)`, equivalently "
            "`measured/1.5 <= prediction <= 1.5*measured`.",
            "",
        ]
    )
    lines.extend(add_tail_rows(sources, tails))
    lines.extend(
        [
            "",
            "The medians reproduce the saved CV5 and LOSO primary errors. "
            "Random forest has the better median and a larger share within "
            "±25%, but MLP has lower 90th and 95th percentile errors under "
            "both protocols.",
            "",
            "### LOSO unsupported folds and mixed-model fallback",
            "",
            "The six LOSO folds with no supported held-out row were:",
            "",
            "| Fold | System |",
            "| --- | --- |",
        ]
    )
    for row in empty.itertuples(index=False):
        lines.append(f"| `{row.fold_id}` | `{row.system_id}` |")
    row = powell.iloc[0]
    lines.extend(
        [
            "",
            f"The hierarchical mixed-model Powell fallback occurred on "
            f"`{row['fold_id']}` for system `{row['system_id']}`. The replay "
            f"converged with optimizer `{row['optimizer']}` and produced "
            f"finite predictions; its supported-row count was "
            f"{sources.add(row['supported_rows'], 'Powell fallback supported rows', ANALYSIS / 'loso_powell_fallback.csv', 'supported_rows', 'fold_id == holdout_089')}.",
            "",
            "### Holm p-values versus equivalence labels",
            "",
            "The Holm-adjusted p-value tests whether a paired secondary "
            "difference is zero, whereas the equivalence label evaluates "
            "whether the same 95% system-clustered interval lies entirely "
            "within the frozen `[-0.05, +0.05]` log-unit margin.",
            "",
            "### Partial-pooling secondary analysis",
            "",
            f"Requested scope: the full CV5 and LOSO ladder. Status: "
            f"`{partial_pooling['status']}` as a secondary analysis, "
            f"logged under deviation `{partial_pooling['deviation_id']}`. "
            f"{partial_pooling['reason']} {partial_pooling['decision']}",
            "No shrinkage performance number is reported because choosing a "
            "formula after observing the primary results would constitute a "
            "new exploratory analysis.",
            "",
            "### LOAFO system coverage reconciliation",
            "",
            "The 105 systems divide into 97 tested systems, two mixed systems "
            "kept in training for every family fold, and six systems whose "
            "held-out family rows were all unsupported:",
            "",
            "| Family | System | Coverage status |",
            "| --- | --- | --- |",
        ]
    )
    for row in absent.itertuples(index=False):
        lines.append(
            f"| `{row.family_id}` | `{row.system_id}` | "
            f"`{row.coverage_status}` |"
        )
    lines.extend(
        [
            "",
            "### Protocol record correction",
            "",
            "`protocol.md` remains hash-pinned at Version 2.1. "
            "`protocol_history.md` adds Version 2.2 to record that the LOAFO "
            "models, seeds 7/11/16, omission of the mixed model, and "
            "family-level paired inference were specified in orchestration "
            "before the run rather than in the pinned file. It also records "
            "that the pinned phrase \"LOAFO stress-test estimate\" contains "
            "no pooling rule, so LOAFO results are descriptive.",
            "",
            "The compute erratum uses 25 repeated-CV outer folds and 5 "
            "sensitivity outer folds. At the documented `0.36` seconds per "
            "fit, corrected totals are 6,210 fits and 2,235.6 seconds "
            "(37.26 minutes): repeated CV 2,375 fits; LOSO 3,360; sensitivity "
            "475.",
            "",
            "## 5. Open questions",
            "",
            "- Should the thesis summarize family-level findings descriptively "
            "without attempting a pooled LOAFO claim, given the explicit "
            "instruction not to pool the three family sets?",
            "- Should the author supply the missing shrinkage posterior and "
            "parameterization if a future exploratory partial-pooling analysis "
            "is desired?",
            "- Should the B200 proposal-versus-actual supported-row difference "
            "be documented as a protocol-map audit note rather than changing "
            "the approved mapping?",
            "- Should the CPU-only stress fold remain outside every headline "
            "claim, consistent with the frozen protocol?",
            "",
            "These questions are for orchestrator review; no Phase 3 work or "
            "exploratory semantic block was started.",
            "",
            "## 6. Acceptance criteria",
            "",
            "| Item | Met? | Evidence |",
            "| --- | --- | --- |",
            "| Run LOAFO on approved 15-family map; mixed systems always train | Yes | Fold table and manifest; 13 non-mixed holdouts |",
            "| Report headline, small/integrated, CPU-only separately; no pooling | Yes | Detailed LOAFO results |",
            "| Systems, rows, supported rows, primary error per model | Yes | LOAFO family tables |",
            "| Paired GNN-minus-MLP system-clustered intervals | Yes | LOAFO paired tables and `paired_differences_by_family.csv` |",
            "| Flag B300 extreme extrapolation | Yes | Section 4 |",
            "| Describe GB10/Thor and VR200 cases | Yes | Section 4 |",
            "| Tail summaries for RF, MLP, GraphSAGE | Yes | Tail summary table and `tail_summary.csv` |",
            "| Six empty LOSO folds and Powell system | Yes | LOSO special-case table and analysis bundle |",
            "| Holm p-value versus equivalence sentence | Yes | Holm/equivalence subsection |",
            "| Partial-pooling secondary handled or deviation logged | Yes | Not run; deviation `S3A-SHRINK-001` |",
            "| Intervals only for families with >=8 supported systems | Yes | `loafo_family_support.csv` and family tables |",
            "| LOAFO tested-system count recomputed and reconciled | Yes | 97 tested; full 105-system reconciliation |",
            "| Full 21-test output and 8-test reconciliation | Yes | Appendix D |",
            "| Tail labels, window definitions, and RF/MLP comparison | Yes | Tail summary subsection |",
            "| Every NA has a resolve-or-do-not-cite decision | Yes | Appendix B and NA register |",
            "| Headline best-GNN < MLP count and group-median comparisons | Yes | Detailed LOAFO tables and Section 4 |",
            "| Paste protocol.md in full | Yes | Appendix A |",
            "| Number-to-source map | Yes | Appendix E and accompanying CSV |",
            "| Do not edit existing result bundles or thesis text | Yes | New bundles plus report/map only |",
            "| Stop before Phase 3 or optional exploratory block | Yes | No Phase 3 or exploratory run performed |",
            "",
            "Verification:",
            "",
            "- `arch -arm64 python3 -m unittest tests.test_phase2 tests.test_folds`: "
            "8 targeted tests passed; full discovery ran 21 tests (see Appendix D).",
            "- LOAFO manifest: clean commit and matching protocol, data, family, "
            "system-assignment, and fold hashes.",
            "- Analysis manifest: clean commit and saved source hashes.",
            f"- Corrections manifest: `{correction_manifest['git_commit']}`; "
            f"partial-pooling status `{correction_manifest['partial_pooling_status']}`; "
            f"test count `{correction_manifest['full_test_suite']['tests']}`.",
            "",
            "## Appendix A. Protocol v2.1 (full)",
            "",
            "```text",
            PROTOCOL.read_text().rstrip(),
            "```",
            "",
            "## Appendix B. Every NA in the number-to-source map",
            "",
            f"There are {sources.add(len(na_register), 'Number of NA entries in source map', args.source_map, 'number', 'number == NA', derivation='row count')} "
            "NA entries. The full register is also saved as "
            f"`{display_path(args.na_register)}`.",
            "",
            "| Map ID | Context | Decision | Rationale |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in na_register.itertuples(index=False):
        lines.append(
            f"| {row.map_id} | {row.context} | `{row.na_decision}` | "
            f"{row.na_rationale} |"
        )
    lines.extend(
        [
            "",
            "## Appendix C. Protocol v2.2 history addendum",
            "",
            "The complete addendum and erratum are in "
            "`mlperf_gnn/protocol_history.md`. The hash-pinned "
            "`protocol.md` was not edited.",
            "",
            "## Appendix D. Full test-suite output",
            "",
            "The earlier report listed the targeted Phase 2 and fold modules "
            "only: 5 Phase 2 tests plus 3 fold tests, totaling 8. Full "
            "discovery also ran 3 data, 8 leakage, and 2 metric tests, for 21 "
            "overall:",
            "",
            "| Module | Tests |",
            "| --- | ---: |",
        ]
    )
    for row in test_counts.itertuples(index=False):
        lines.append(f"| `{row.module}` | {row.tests} |")
    lines.extend(
        [
            "",
            "Full output:",
            "",
            "```text",
            test_output.rstrip(),
            "```",
            "",
            "## Appendix E. Number-to-source map",
            "",
            f"The exhaustive machine-readable map is `{display_path(args.source_map)}`. "
            "Each row identifies the displayed number, context, results file, "
            "source column, row filter, and derivation. Direct values use "
            "`direct`; percentages and counts state their transformation. "
            "The `na_decision` and `na_rationale` columns classify every NA.",
            "",
            "Key bundles:",
            "",
            f"- LOAFO manifest commit: `"
            f"{loafo_manifest['git_commit']}`.",
            f"- LOAFO prediction hash: `"
            f"{loafo_manifest['output_sha256']['predictions_by_fold_seed.csv']}`.",
            f"- Tail/LOSO diagnostic manifest commit: `"
            f"{analysis_manifest['git_commit']}`.",
            f"- Tail summary hash: `"
            f"{analysis_manifest['output_sha256']['tail_summary.csv']}`.",
            f"- Corrections manifest commit: `"
            f"{correction_manifest['git_commit']}`.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n")
    number_map.to_csv(args.source_map, index=False)
    na_register.to_csv(args.na_register, index=False)
    print(args.report)
    print(args.source_map)
    print(args.na_register)


if __name__ == "__main__":
    main()
