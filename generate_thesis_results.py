#!/usr/bin/env python3
"""Generate all thesis result tables and figures from saved bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
DATA = ROOT / "data" / "mlperf_long.csv"
FINAL = ROOT / "results" / "phase2-final-f51e57a"
TAIL = ROOT / "results" / "phase2-5-analysis-4faac06"
DIAG = ROOT / "results" / "phase2-cv5-diagnostics-77c4752"
CORRECTIONS = ROOT / "results" / "phase2-5-corrections-e97bb93"
LOAFO = ROOT / "results" / "phase2-loafo-final-1a65664"
PHASE1 = ROOT / "results" / "phase1-final-5c9dfde"
GATES = ROOT / "results" / "phase1_5-gates-f82e4d2"

DISPLAY_NAMES = {
    "best_gnn": "Best GNN",
    "gat": "GAT",
    "gcn": "GCN",
    "gradient_boosting": "Gradient boosting",
    "graphsage": "GraphSAGE",
    "graphsage_no_accelerator": "GraphSAGE without accelerator",
    "graphsage_no_processor": "GraphSAGE without processor",
    "graphsage_no_system": "GraphSAGE without system",
    "graphsage_no_workload": "GraphSAGE without workload",
    "graphsage_shuffled": "GraphSAGE shuffled edges",
    "group_median": "Group median",
    "hierarchical_mixed": "Hierarchical mixed",
    "mlp": "MLP",
    "random_forest": "Random forest",
    "ridge": "Ridge",
}
HEADLINE = (
    "amd_cdna4_mi350x",
    "amd_cdna4_mi355x",
    "intel_xe2_battlemage_arc_pro_b70",
    "nvidia_blackwell_b200",
    "nvidia_blackwell_b300",
    "nvidia_blackwell_rtx_pro",
)
OTHER = (
    "amd_cdna3_mi300x",
    "amd_rdna35_integrated_radeon_8060s",
    "nvidia_blackwell_integrated_gb10",
    "nvidia_blackwell_integrated_thor",
    "nvidia_hopper_h200",
    "nvidia_rubin_vr200",
    "cpu_only_intel_xeon6_granite_rapids",
)
TRAINED = ("ridge", "random_forest", "mlp", "graphsage", "gcn", "gat")
INPUT_FILES = {
    "data": DATA,
    "cv5_main": FINAL / "cv5_main_table.csv",
    "loso_main": FINAL / "loso_main_table.csv",
    "sensitivity_main": FINAL / "sensitivity_main_table.csv",
    "primary_decision": FINAL / "primary_decision.csv",
    "secondary_holm": FINAL / "secondary_holm.csv",
    "coverage": DIAG / "coverage_reconciliation.csv",
    "tail_summary": TAIL / "tail_summary.csv",
    "loafo_main": LOAFO / "main_table_by_family.csv",
    "loafo_pairs": LOAFO / "paired_differences_by_family.csv",
    "loafo_support": CORRECTIONS / "loafo_family_support.csv",
    "loafo_comparisons": CORRECTIONS / "loafo_group_median_comparisons.csv",
    "partition_errors": DIAG / "partition_primary_errors.csv",
    "gat_diagnostics": DIAG / "gat_training_summary.csv",
    "top_residuals": DIAG / "top_abs_z_residuals.csv",
    "leak_test": PHASE1 / "paired_bootstrap_differences.csv",
    "spearman_reconciliation": GATES / "spearman_reconciliation.csv",
}
SNAPSHOTS: dict[str, Path] = {}


def prepare_snapshots(output: Path, *, refresh: bool = True) -> None:
    snapshot_dir = output / "source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, source in INPUT_FILES.items():
        destination = snapshot_dir / source.name
        if refresh and source.exists():
            shutil.copy2(source, destination)
        if not destination.exists():
            raise FileNotFoundError(
                f"Missing result source and snapshot: {source} / {destination}"
            )
        SNAPSHOTS[name] = destination
    manifest = {
        name: {
            "snapshot": str(path.relative_to(THESIS)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": str(INPUT_FILES[name].relative_to(ROOT.parent)),
        }
        for name, path in sorted(SNAPSHOTS.items())
    }
    (snapshot_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def read_input(name: str) -> pd.DataFrame:
    return pd.read_csv(SNAPSHOTS[name])


def f(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}\\%"


def esc(value: str) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def table_wrapper(
    caption: str,
    label: str,
    columns: str,
    rows: list[str],
    *,
    size: str = r"\small",
    landscape: bool = False,
    note: str | None = None,
) -> str:
    environment = "table*" if landscape else "table"
    note_lines = (
        [r"\par\smallskip", rf"\footnotesize {note}"]
        if note is not None
        else []
    )
    return "\n".join(
        [
            f"\\begin{{{environment}}}[htbp]",
            r"\centering",
            size,
            r"\resizebox{\textwidth}{!}{%",
            f"\\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
            *note_lines,
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\end{{{environment}}}",
        ]
    )


def generate_tables(output: Path) -> None:
    cv = read_input("cv5_main").set_index("method")
    loso = read_input("loso_main").set_index("method")
    sens = read_input("sensitivity_main").set_index("method")
    methods = list(
        cv["primary_error_mean"].sort_values().index
    )
    data = read_input("data")
    coverage_rows = [
        r"\textbf{Suite} & \textbf{Systems} & \textbf{Measurements} & \textbf{Workload groups} & \textbf{Submitters} \\",
        r"\midrule",
    ]
    for suite_name, local in data.groupby("suite", sort=True):
        coverage_rows.append(
            f"{suite_name.title()} & {local['system_id'].nunique()} & "
            f"{len(local)} & {local['target_group'].nunique()} & "
            f"{local['organization'].nunique()} \\\\"
        )
    coverage_rows.append(r"\midrule")
    coverage_rows.append(
        f"Total & {data['system_id'].nunique()} & {len(data)} & "
        f"{data['target_group'].nunique()} & {data['organization'].nunique()} \\\\"
    )
    write(
        output / "tab_data_coverage.tex",
        table_wrapper(
            "Coverage of the MLPerf Inference v6.1 Closed-division "
            "Datacenter and Edge exports. Submitters are counted within each "
            "suite; the total is the number of unique organizations across "
            "both suites, so the suite counts need not sum to the total when "
            "submitters overlap.",
            "tab:data-coverage",
            "lrrrr",
            coverage_rows,
        ),
    )

    protocol_rows = [
        r"\textbf{Protocol} & \textbf{Outer folds} & \textbf{Neural seeds} & \textbf{Purpose} \\",
        r"\midrule",
        r"Repeated CV5 & 25 & 7--16 & Co-primary estimate \\",
        r"LOSO & 105 & 7, 11, 16 & Co-primary system extrapolation \\",
        r"Sensitivity & 5 & 7--16 & Near-duplicate-group guard \\",
        r"LOAFO & 13 & 7, 11, 16 & Descriptive family extrapolation \\",
    ]
    write(
        output / "tab_protocol.tex",
        table_wrapper(
            "Executed protocols. The repeated-CV design contains five "
            "partitions with five folds each.",
            "tab:protocol",
            "lrrp{0.45\\textwidth}",
            protocol_rows,
        ),
    )

    parameter_rows = [
        r"\textbf{Parameter} & \textbf{v6.1 rule or published-test setting} \\",
        r"\midrule",
        r"Scenario & Offline, Server, SingleStream, MultiStream, or Interactive as supported by the workload \\",
        r"Accuracy target & Workload-specific; Closed-division targets are normally 99\% or 99.9\% of reference accuracy \\",
        r"Minimum duration & 600,000 ms by default; some published runs used 1,200,000 ms \\",
        r"SingleStream / MultiStream & 90th / 99th percentile; base timing limits are workload-specific \\",
        r"Server / Offline & 99th / 90th percentile; target QPS is system-specific \\",
        r"LLM latency limits & TTFT and TPOT limits are specified per workload and scenario \\",
        r"Precision and parallelism & Record weight and KV-cache precision, batch size, sequence length, and parallel degree \\",
        r"Random seeds and artifacts & Preserve QSL, sample-index, schedule, summary, detail, accuracy, and compliance artifacts \\",
    ]
    write(
        output / "tab_mlperf_parameters.tex",
        table_wrapper(
            "Parameters required to reproduce an MLPerf Inference v6.1 "
            "Closed-division measurement.",
            "tab:mlperf-parameters",
            "p{0.24\\textwidth}p{0.66\\textwidth}",
            parameter_rows,
            size=r"\scriptsize",
        ),
    )
    rows = [
        r"\textbf{Method} & \textbf{CV5 primary error} & \textbf{LOSO primary error} & \textbf{Sensitivity primary error} \\",
        r"\midrule",
    ]
    for method in methods:
        cv_value = (
            f"{f(cv.loc[method, 'primary_error_mean'])} "
            f"[{f(cv.loc[method, 'primary_error_95ci_low'])}, "
            f"{f(cv.loc[method, 'primary_error_95ci_high'])}]"
        )
        loso_value = (
            f"{f(loso.loc[method, 'primary_error'])} "
            f"[{f(loso.loc[method, 'primary_error_95ci_low'])}, "
            f"{f(loso.loc[method, 'primary_error_95ci_high'])}]"
        )
        sens_value = (
            f"{f(sens.loc[method, 'primary_error'])} "
            f"[{f(sens.loc[method, 'primary_error_95ci_low'])}, "
            f"{f(sens.loc[method, 'primary_error_95ci_high'])}]"
        )
        rows.append(
            f"{DISPLAY_NAMES[method]} & {cv_value} & {loso_value} & {sens_value} \\\\"
        )
    write(
        output / "tab_model_ladder.tex",
        table_wrapper(
            "Primary error for the complete model ladder. Entries are median "
            "absolute log ratio and 95\\% system-clustered intervals. Lower is better.",
            "tab:model-ladder",
            "lccc",
            rows,
            size=r"\scriptsize",
            landscape=True,
        ),
    )

    primary = read_input("primary_decision")
    rows = [
        r"\textbf{Protocol} & \textbf{Best GNN $-$ MLP} & \textbf{95\% CI} & \textbf{Frozen label} \\",
        r"\midrule",
    ]
    protocol_names = {
        "cv5_partitioned": "Repeated CV5",
        "loso": "LOSO",
        "sensitivity": "Sensitivity",
    }
    for row in primary.itertuples(index=False):
        rows.append(
            f"{protocol_names[row.protocol]} & {f(row.mean_difference)} & "
            f"[{f(row.difference_95ci_low)}, {f(row.difference_95ci_high)}] & "
            f"{row.status} \\\\"
        )
    write(
        output / "tab_primary_decision.tex",
        table_wrapper(
            "Co-primary and sensitivity best-GNN-minus-MLP decisions. The "
            "intervals are raw 95\\% system-clustered bootstrap intervals. "
            "The sensitivity row applies the frozen label rule but is not "
            "co-primary and is nearly redundant with the system-level split.",
            "tab:primary-decision",
            "lccc",
            rows,
        ),
    )

    secondary = read_input("secondary_holm")
    rows = [
        r"\textbf{Protocol} & \textbf{Comparison} & \textbf{Difference} & \textbf{95\% CI} & \textbf{Holm $p$} & \textbf{Label} \\",
        r"\midrule",
    ]
    for row in secondary.itertuples(index=False):
        rows.append(
            f"{protocol_names[row.protocol]} & "
            f"{DISPLAY_NAMES[row.left]} $-$ {DISPLAY_NAMES[row.right]} & "
            f"{f(row.mean_difference)} & "
            f"[{f(row.difference_95ci_low)}, {f(row.difference_95ci_high)}] & "
            f"{f(row.holm_adjusted_p_value, 3)} & {row.status} \\\\"
        )
    write(
        output / "tab_secondary_holm.tex",
        table_wrapper(
            "Secondary comparisons with raw 95\\% system-clustered bootstrap "
            "intervals and Holm-adjusted p-values. Holm correction is applied "
            "to p-values only, not to interval endpoints. Equivalence is "
            "interval containment in the frozen $[-0.05,+0.05]$ margin and "
            "can coexist with an interval that excludes zero; for example, the "
            "repeated-CV edge ablation remains equivalent despite its positive "
            "interval and Holm p-value of 0.096.",
            "tab:secondary-holm",
            "llcccc",
            rows,
            size=r"\scriptsize",
            landscape=True,
        ),
    )

    tails = read_input("tail_summary")
    rows = [
        r"\textbf{Protocol} & \textbf{Method} & \textbf{Median} & \textbf{75th} & \textbf{90th} & \textbf{95th} & \textbf{Max} & \textbf{$\pm$25\%} & \textbf{$\pm$50\%} \\",
        r"\midrule",
    ]
    for protocol in ("cv5_partitioned", "loso"):
        for method in ("random_forest", "mlp", "graphsage"):
            row = tails.loc[
                tails["protocol"].eq(protocol) & tails["method"].eq(method)
            ].iloc[0]
            rows.append(
                f"{protocol_names[protocol]} & {DISPLAY_NAMES[method]} & "
                f"{f(row['median'])} & {f(row.p75)} & {f(row.p90)} & "
                f"{f(row.p95)} & {f(row['max'])} & "
                f"{pct(row.within_25pct)} & {pct(row.within_50pct)} \\\\"
            )
    write(
        output / "tab_tail_summary.tex",
        table_wrapper(
            r"Absolute-log-ratio tails for random forest, MLP, and GraphSAGE. "
            r"The CV5 maximum is the mean of five per-partition maxima; the "
            r"LOSO maximum is the pooled out-of-fold maximum. The $\pm$25\% "
            r"window is $|\log(\hat y/y)|\leq\log(1.25)$; the $\pm$50\% "
            r"window uses $\log(1.5)$.",
            "tab:tail-summary",
            "llccccccc",
            rows,
            size=r"\scriptsize",
            landscape=True,
        ),
    )

    support = read_input("loafo_support").set_index(
        "family_id"
    )
    loafo = read_input("loafo_main")
    rows = [
        r"\textbf{Family} & \textbf{Sys.} & \textbf{Sup. sys.} & \textbf{Sup. rows} & \textbf{Group median} & \textbf{Ridge} & \textbf{RF} & \textbf{MLP} & \textbf{GraphSAGE} & \textbf{GCN} & \textbf{GAT} & \textbf{Best GNN} \\",
        r"\midrule",
    ]
    for family in HEADLINE:
        local = loafo.loc[loafo["family_id"].eq(family)].set_index("method")
        support_row = support.loc[family]
        values = [
            esc(family),
            str(int(support_row.systems)),
            str(int(support_row.supported_systems)),
            str(int(support_row.supported_rows)),
        ]
        for method in (
            "group_median",
            "ridge",
            "random_forest",
            "mlp",
            "graphsage",
            "gcn",
            "gat",
            "best_gnn",
        ):
            values.append(f(local.loc[method, "primary_error"]))
        rows.append(" & ".join(values) + r" \\")
    write(
        output / "tab_loafo_headline.tex",
        table_wrapper(
            "Descriptive leave-one-family-out results for the six headline "
            "families. No pooled estimate is formed. MI350X has four systems, "
            "of which three are supported; Intel Arc Pro B70 also has four "
            "systems, of which three are supported.",
            "tab:loafo-headline",
            "lrrrcccccccc",
            rows,
            size=r"\scriptsize",
            landscape=True,
        ),
    )

    rows = [
        r"\textbf{Family} & \textbf{Sys.} & \textbf{Sup. sys.} & \textbf{Sup. rows} & \textbf{Group median} & \textbf{Ridge} & \textbf{RF} & \textbf{MLP} & \textbf{GraphSAGE} & \textbf{GCN} & \textbf{GAT} & \textbf{Best GNN} \\",
        r"\midrule",
    ]
    for family in OTHER:
        local = loafo.loc[loafo["family_id"].eq(family)].set_index("method")
        support_row = support.loc[family]
        values = [
            esc(family),
            str(int(support_row.systems)),
            str(int(support_row.supported_systems)),
            str(int(support_row.supported_rows)),
        ]
        for method in (
            "group_median",
            "ridge",
            "random_forest",
            "mlp",
            "graphsage",
            "gcn",
            "gat",
            "best_gnn",
        ):
            values.append(f(local.loc[method, "primary_error"]))
        rows.append(" & ".join(values) + r" \\")
    write(
        output / "tab_loafo_other.tex",
        table_wrapper(
            "Descriptive small/integrated families and the CPU-only stress "
            "case. The CPU-only family remains outside the headline set.",
            "tab:loafo-other",
            "lrrrcccccccc",
            rows,
            size=r"\scriptsize",
            landscape=True,
        ),
    )

    pairs = read_input("loafo_pairs")
    comparisons = read_input("loafo_comparisons")
    rows = [
        r"\textbf{Family} & \textbf{Sup. sys.} & \textbf{Models beating GM} & \textbf{Best GNN $-$ MLP} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for family in HEADLINE:
        local = comparisons.loc[comparisons["family_id"].eq(family)]
        beating = int(
            local.loc[
                local["method"].isin(TRAINED), "model_better_than_group_median"
            ].sum()
        )
        pair = pairs.loc[
            pairs["family_id"].eq(family) & pairs["left"].eq("best_gnn")
        ].iloc[0]
        support_row = support.loc[family]
        interval = (
            f"[{f(pair.difference_95ci_low)}, {f(pair.difference_95ci_high)}]"
            if bool(support_row.interval_eligible)
            else "NA (too few systems)"
        )
        rows.append(
            f"{esc(family)} & {int(support_row.supported_systems)} & "
            f"{beating}/6 & {f(pair.point_difference)} & {interval} \\\\"
        )
    rows.append(r"\midrule")
    family = "cpu_only_intel_xeon6_granite_rapids"
    local = comparisons.loc[comparisons["family_id"].eq(family)]
    beating = int(
        local.loc[
            local["method"].isin(TRAINED), "model_better_than_group_median"
        ].sum()
    )
    pair = pairs.loc[
        pairs["family_id"].eq(family) & pairs["left"].eq("best_gnn")
    ].iloc[0]
    support_row = support.loc[family]
    interval = (
        f"[{f(pair.difference_95ci_low)}, {f(pair.difference_95ci_high)}]"
        if bool(support_row.interval_eligible)
        else "NA (too few systems)"
    )
    rows.append(
        "CPU-only stress case (outside headline) & "
        f"{int(support_row.supported_systems)} & {beating}/6 & "
        f"{f(pair.point_difference)} & {interval} \\\\"
    )
    write(
        output / "tab_loafo_pairing.tex",
        table_wrapper(
            "Headline-family comparisons with the group-median baseline and "
            "best-GNN-minus-MLP intervals, followed by a separately labeled "
            "CPU-only stress case outside the six-family headline set. "
            "The eight-supported-system interval threshold excludes MI350X "
            "and Intel Arc Pro B70. No multiplicity adjustment is applied to "
            "these descriptive comparisons.",
            "tab:loafo-pairing",
            "lrccc",
            rows,
            size=r"\scriptsize",
            note=(
                "Best GNN has a lower point estimate than MLP in MI355X and "
                "Intel Arc Pro B70. MI350X and Intel Arc Pro B70 each have "
                "four systems but only three supported systems. Intervals are "
                "shown only for families with at least eight supported "
                "systems. Uncorrected intervals are descriptive."
            ),
        ),
    )

    partition = read_input("partition_errors")
    methods_to_show = ("random_forest", "mlp", "graphsage", "gcn", "gat")
    rows = [
        r"\textbf{Method} & \textbf{P0} & \textbf{P1} & \textbf{P2} & \textbf{P3} & \textbf{P4} & \textbf{SD} \\",
        r"\midrule",
    ]
    for method in methods_to_show:
        local = partition.loc[partition["method"].eq(method)].set_index(
            "protocol"
        )
        vals = [
            local.loc[f"cv5_partition_{index}", "primary_error"]
            for index in range(5)
        ]
        rows.append(
            f"{DISPLAY_NAMES[method]} & "
            + " & ".join(f(value) for value in vals)
            + f" & {f(np.std(vals, ddof=1))} \\\\"
        )
    write(
        output / "tab_partition_errors.tex",
        table_wrapper(
            "Partition-specific CV5 primary errors. The final column is the "
            "sample standard deviation across the five partitions.",
            "tab:partition-errors",
            "lcccccc",
            rows,
        ),
    )

    gat = read_input("gat_diagnostics")
    rows = [
        r"\textbf{Diagnostic} & \textbf{Value} \\",
        r"\midrule",
        f"Fold-seed fits & {len(gat) * int(gat['seeds'].iloc[0])} \\\\",
        f"Normalized prediction variance & {f(gat['normalized_prediction_variance_mean'].min())}--{f(gat['normalized_prediction_variance_mean'].max())} \\\\",
        f"Unique predictions & {int(gat['prediction_unique_mean'].min())}--{int(gat['prediction_unique_mean'].max())} per fold-seed \\\\",
        f"Best epoch & {int(gat['best_epoch_min'].min())}--{int(gat['best_epoch_max'].max())} \\\\",
        f"Final validation loss & {f(gat['final_validation_loss_mean'].min())}--{f(gat['final_validation_loss_mean'].max())} \\\\",
    ]
    write(
        output / "tab_gat_diagnostic.tex",
        table_wrapper(
            "GAT training diagnostics across the 25 repeated-CV fold cells "
            "(25 fold cells $\\times$ 10 neural seeds). GAT predictions vary; "
            "its failure is not numerical collapse.",
            "tab:gat-diagnostic",
            "lc",
            rows,
        ),
    )

    leak = read_input("leak_test").iloc[0]
    spearman = read_input("spearman_reconciliation")
    historical = spearman.loc[
        spearman["dataset"].eq("exact_phase0_graphsage")
        & spearman["definition"].eq("table_6_4_minimum_3_rows")
    ].iloc[0]
    reproducible = spearman.loc[
        spearman["dataset"].eq("attribution_baseline")
        & spearman["definition"].eq("table_6_4_minimum_3_rows")
    ].iloc[0]
    rows = [
        r"\textbf{Provenance quantity} & \textbf{Value} \\",
        r"\midrule",
        f"Leak effect on primary error & +{f(leak.mean_difference)} \\\\",
        f"Leak 95\\% interval & [{f(leak.difference_95ci_low)}, {f(leak.difference_95ci_high)}] \\\\",
        f"Historical single-run Spearman & {f(historical['mean'])} \\\\",
        f"Ten-seed pinned-fold Spearman & {f(reproducible['mean'])} (SD {f(reproducible['sd'])}) \\\\",
    ]
    write(
        output / "tab_provenance.tex",
        table_wrapper(
            "Provenance values used to explain the earlier leakage and the "
            "historical rank-correlation discrepancy. These are audit values, "
            "not replacement thesis results.",
            "tab:provenance",
            "lc",
            rows,
        ),
    )


def generate_figures(output: Path) -> None:
    cv = read_input("cv5_main").set_index("method")
    loso = read_input("loso_main").set_index("method")
    methods = [
        "random_forest",
        "mlp",
        "ridge",
        "gradient_boosting",
        "graphsage",
        "gcn",
        "gat",
        "best_gnn",
    ]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(cv.loc[methods, "primary_error_mean"], y, label="CV5", marker="o")
    ax.scatter(loso.loc[methods, "primary_error"], y, label="LOSO", marker="s")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES[value] for value in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Median absolute log ratio")
    ax.set_title("Primary error on held-out systems")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        output / "fig_model_ladder.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    primary = read_input("primary_decision")
    primary = primary.loc[
        primary["protocol"].isin(["cv5_partitioned", "loso", "sensitivity"])
    ]
    labels = ["CV5", "LOSO", "Sensitivity"]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.errorbar(
        primary["mean_difference"],
        y,
        xerr=np.vstack(
            [
                primary["mean_difference"] - primary["difference_95ci_low"],
                primary["difference_95ci_high"] - primary["mean_difference"],
            ]
        ),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.05, color="0.45", linestyle="--", linewidth=0.9)
    ax.axvline(-0.05, color="0.45", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best GNN minus MLP, positive means GNN error is higher")
    ax.set_title("Co-primary and sensitivity decisions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output / "fig_primary_decision.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    tails = read_input("tail_summary")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, protocol, title in zip(
        axes,
        ("cv5_partitioned", "loso"),
        ("CV5", "LOSO"),
    ):
        local = tails.loc[tails["protocol"].eq(protocol)].set_index("method")
        methods = ["random_forest", "mlp", "graphsage"]
        x = np.arange(2)
        width = 0.25
        for offset, method in enumerate(methods):
            ax.bar(
                x + (offset - 1) * width,
                [local.loc[method, "p90"], local.loc[method, "p95"]],
                width,
                label=DISPLAY_NAMES[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["90th percentile", "95th percentile"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Absolute log ratio")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Tail error depends on the statistic and protocol")
    fig.tight_layout()
    fig.savefig(
        output / "fig_tail_errors.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    comparisons = read_input("loafo_comparisons")
    heat = (
        comparisons.loc[
            comparisons["family_id"].isin(HEADLINE)
            & comparisons["method"].isin(TRAINED)
        ]
        .pivot(
            index="family_id",
            columns="method",
            values="model_minus_group_median",
        )
        .loc[list(HEADLINE), list(TRAINED)]
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    limit = min(float(np.nanmax(np.abs(heat.to_numpy()))), 0.6)
    plotted = np.clip(heat.to_numpy(), -limit, limit)
    image = ax.imshow(
        plotted, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
    )
    ax.set_xticks(np.arange(len(TRAINED)))
    ax.set_xticklabels([DISPLAY_NAMES[value] for value in TRAINED], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(HEADLINE)))
    ax.set_yticklabels([esc(value).replace(r"\_", "_") for value in HEADLINE], fontsize=8)
    ax.set_title("Difference from group-median error by held-out family")
    fig.colorbar(image, ax=ax, label="Model error minus group-median error")
    fig.tight_layout()
    fig.savefig(
        output / "fig_loafo_group_median.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=THESIS / "generated"
    )
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepare_snapshots(output, refresh=not args.snapshot_only)
    generate_tables(output)
    generate_figures(output)
    manifest = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "generated_asset_manifest.json"
    }
    (output / "generated_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
