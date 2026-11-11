#!/usr/bin/env python3
"""Post-lock exploratory ablation: frozen features versus memory features.

Compares the frozen protocol v2.1 result bundles with reruns that used
``AI_BENCHMARK_FEATURE_SET=memory`` (accelerator memory capacity and type from
``data/system_memory.csv``). Nothing here changes the frozen co-primary label;
the memory-feature labels are exploratory.

Both bundles use the same system-clustered bootstrap seed and the same
supported systems, so bootstrap replicate ``i`` resamples the same systems in
both. The replicate-wise difference ``memory - frozen`` is therefore a paired
system-clustered bootstrap of the feature-set effect. The script asserts that
assumption before using it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_phase2_5 import cv_tail_summary, loso_tail_summary
from combine_phase2_final import holm
from evaluation.data import sha256_file


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FROZEN = {
    "cv5": RESULTS / "phase2-cv5-final-aeeb4b4",
    "loso": RESULTS / "phase2-loso-final-a689ba5",
    "sensitivity": RESULTS / "phase2-sensitivity-final-a689ba5",
    "loafo": RESULTS / "phase2-loafo-final-1a65664",
}
MEMORY = {
    "cv5": RESULTS / "exploratory-memory-cv5-final",
    "loso": RESULTS / "exploratory-memory-loso-final",
    "sensitivity": RESULTS / "exploratory-memory-sensitivity-final",
    "loafo": RESULTS / "exploratory-memory-loafo-final",
}
PREDICTION_FILES = {
    "cv5": "predictions_by_partition_seed.csv",
    "loso": "predictions_by_fold_seed.csv",
    "sensitivity": "predictions_by_fold_seed.csv",
}
FOCUS_METHODS = [
    "random_forest",
    "mlp",
    "ridge",
    "gradient_boosting",
    "graphsage",
    "gcn",
    "gat",
    "best_gnn",
    "graphsage_shuffled",
    "hierarchical_mixed",
    "group_median",
]
MARGIN = 0.05


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def label(low: float, high: float) -> str:
    if high < -MARGIN:
        return "better"
    if low > MARGIN:
        return "worse"
    if low >= -MARGIN and high <= MARGIN:
        return "equivalent"
    return "inconclusive"


def two_sided_p(difference: pd.Series) -> float:
    return min(
        1.0,
        2.0 * min(float((difference >= 0).mean()), float((difference <= 0).mean())),
    )


def check_same_population(protocol: str) -> None:
    columns = ["row_uid", "system_id"]
    frozen = pd.read_csv(
        FROZEN[protocol] / PREDICTION_FILES[protocol], usecols=columns + ["method"]
    )
    memory = pd.read_csv(
        MEMORY[protocol] / PREDICTION_FILES[protocol], usecols=columns + ["method"]
    )
    for name, frame in [("frozen", frozen), ("memory", memory)]:
        frame.drop(frame.index[~frame["method"].eq("mlp")], inplace=True)
    if set(map(tuple, frozen[columns].values)) != set(
        map(tuple, memory[columns].values)
    ):
        raise AssertionError(f"{protocol}: supported rows differ between bundles")


def wide_draws(bundle: Path) -> pd.DataFrame:
    draws = pd.read_csv(bundle / "bootstrap_by_method.csv")
    return draws.pivot(
        index="bootstrap_replicate", columns="method", values="primary_error"
    )


def ladder_comparison(protocol: str) -> pd.DataFrame:
    # CV5 tables name the partition-averaged error primary_error_mean; LOSO and
    # sensitivity tables name the pooled error primary_error.
    rename = {"primary_error": "primary_error_mean"}
    frozen_main = pd.read_csv(FROZEN[protocol] / "main_table.csv").rename(columns=rename)
    memory_main = pd.read_csv(MEMORY[protocol] / "main_table.csv").rename(columns=rename)
    frozen_draws = wide_draws(FROZEN[protocol])
    memory_draws = wide_draws(MEMORY[protocol])
    if not frozen_draws.index.equals(memory_draws.index):
        raise AssertionError(f"{protocol}: bootstrap replicates differ")
    columns = ["method", "primary_error_mean", "primary_error_95ci_low", "primary_error_95ci_high"]
    merged = frozen_main[columns].merge(
        memory_main[columns], on="method", suffixes=("_frozen", "_memory")
    )
    rows = []
    for row in merged.itertuples(index=False):
        delta = memory_draws[row.method] - frozen_draws[row.method]
        low, high = float(delta.quantile(0.025)), float(delta.quantile(0.975))
        rows.append(
            {
                "protocol": protocol,
                "method": row.method,
                "primary_error_frozen": row.primary_error_mean_frozen,
                "primary_error_frozen_95ci_low": row.primary_error_95ci_low_frozen,
                "primary_error_frozen_95ci_high": row.primary_error_95ci_high_frozen,
                "primary_error_memory": row.primary_error_mean_memory,
                "primary_error_memory_95ci_low": row.primary_error_95ci_low_memory,
                "primary_error_memory_95ci_high": row.primary_error_95ci_high_memory,
                "memory_minus_frozen": row.primary_error_mean_memory
                - row.primary_error_mean_frozen,
                "memory_minus_frozen_bootstrap_mean": float(delta.mean()),
                "memory_minus_frozen_95ci_low": low,
                "memory_minus_frozen_95ci_high": high,
                "probability_memory_better": float((delta < 0).mean()),
                "bootstrap_p_value": two_sided_p(delta),
                "label": label(low, high),
            }
        )
    return pd.DataFrame(rows)


def memory_pair_decisions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pairs = {}
    for protocol in ["cv5", "loso", "sensitivity"]:
        frame = pd.read_csv(MEMORY[protocol] / "paired_differences.csv")
        if "bootstrap_p_value" not in frame.columns:
            wide = wide_draws(MEMORY[protocol])
            frame["bootstrap_p_value"] = [
                two_sided_p(wide[r.left] - wide[r.right])
                for r in frame.itertuples(index=False)
            ]
        frame["protocol"] = protocol
        pairs[protocol] = frame
    frozen_final = pd.read_csv(RESULTS / "phase2-final-f51e57a" / "primary_decision.csv")
    primary = pd.concat(
        [pairs[p].loc[pairs[p]["role"].eq("primary")] for p in pairs],
        ignore_index=True,
    )
    frozen_final["protocol"] = frozen_final["protocol"].replace(
        {"cv5_partitioned": "cv5"}
    )
    primary = primary.merge(
        frozen_final[["protocol", "mean_difference", "difference_95ci_low", "difference_95ci_high", "status"]],
        on="protocol",
        suffixes=("", "_frozen"),
    )
    secondary = holm(
        pd.concat(
            [pairs[p].loc[pairs[p]["role"].eq("secondary")] for p in ["cv5", "loso"]],
            ignore_index=True,
        )
    )
    labels = primary.loc[primary["protocol"].isin(["cv5", "loso"])].set_index(
        "protocol"
    )["status"].to_dict()
    decision = {
        "status": "post_lock_exploratory_not_co_primary",
        "hypothesis": "best_gnn - mlp primary error (memory feature set)",
        "cv5_label": labels["cv5"],
        "loso_label": labels["loso"],
        "combined_label": next(iter(labels.values()))
        if len(set(labels.values())) == 1
        else "inconclusive",
        "sensitivity_label": primary.loc[
            primary["protocol"].eq("sensitivity"), "status"
        ].iloc[0],
        "frozen_combined_label_unchanged": "inconclusive",
    }
    return primary, secondary, decision


def loafo_comparison() -> pd.DataFrame:
    keys = ["fold_id", "family_id", "holdout_disposition", "method"]
    frozen = pd.read_csv(FROZEN["loafo"] / "main_table_by_family.csv")
    memory = pd.read_csv(MEMORY["loafo"] / "main_table_by_family.csv")
    merged = frozen[keys + ["systems", "supported_rows", "primary_error"]].merge(
        memory[keys + ["primary_error"]], on=keys, suffixes=("_frozen", "_memory")
    )
    merged["memory_minus_frozen"] = (
        merged["primary_error_memory"] - merged["primary_error_frozen"]
    )
    baseline = merged.loc[merged["method"].eq("group_median"), ["fold_id", "primary_error_frozen"]]
    baseline = baseline.rename(columns={"primary_error_frozen": "group_median_error"})
    merged = merged.merge(baseline, on="fold_id", how="left")
    merged["beats_group_median_frozen"] = (
        merged["primary_error_frozen"] < merged["group_median_error"]
    )
    merged["beats_group_median_memory"] = (
        merged["primary_error_memory"] < merged["group_median_error"]
    )
    return merged


LOAFO_TRAINED = ["ridge", "random_forest", "mlp", "graphsage", "gcn", "gat"]


def loafo_counts(loafo: pd.DataFrame) -> pd.DataFrame:
    """Headline-family counts of trained models beating the group median."""
    local = loafo.loc[
        loafo["holdout_disposition"].eq("headline_holdout")
        & loafo["method"].isin(LOAFO_TRAINED)
    ]
    counts = local.groupby("family_id", as_index=False)[
        ["beats_group_median_frozen", "beats_group_median_memory"]
    ].sum()
    total = counts.sum(numeric_only=True).to_dict()
    counts.loc[len(counts)] = {"family_id": "TOTAL", **total}
    counts["models"] = [len(LOAFO_TRAINED)] * (len(counts) - 1) + [len(local)]
    return counts


def tail_comparison() -> pd.DataFrame:
    frames = []
    for name, bundles in [("frozen", FROZEN), ("memory", MEMORY)]:
        cv = cv_tail_summary(
            pd.read_csv(bundles["cv5"] / PREDICTION_FILES["cv5"])
        )
        loso = loso_tail_summary(
            pd.read_csv(bundles["loso"] / PREDICTION_FILES["loso"])
        )
        frames.append(pd.concat([cv, loso], ignore_index=True).assign(feature_set=name))
    return pd.concat(frames, ignore_index=True)


METHOD_LABELS = {
    "random_forest": "Random forest",
    "mlp": "MLP",
    "ridge": "Ridge",
    "gradient_boosting": "Gradient boosting",
    "graphsage": "GraphSAGE",
    "gcn": "GCN",
    "gat": "GAT",
    "best_gnn": "Best GNN",
    "graphsage_shuffled": "GraphSAGE shuffled edges",
    "hierarchical_mixed": "Hierarchical mixed$^{a}$",
    "group_median": "Group median",
}
PROTOCOL_LABELS = {"cv5": "Repeated CV5", "loso": "LOSO", "sensitivity": "Sensitivity"}
FAMILY_LABELS = {
    "amd_cdna4_mi350x": "AMD MI350X",
    "amd_cdna4_mi355x": "AMD MI355X",
    "intel_xe2_battlemage_arc_pro_b70": "Intel Arc Pro B70",
    "nvidia_blackwell_b200": "NVIDIA B200",
    "nvidia_blackwell_b300": "NVIDIA B300",
    "nvidia_blackwell_rtx_pro": "NVIDIA RTX PRO",
    "TOTAL": "Total",
}


def signed(value: float, digits: int = 3) -> str:
    text = f"{value:+.{digits}f}"
    return "+" + text[1:] if float(text) == 0 else text


def ci(low: float, high: float, digits: int = 3) -> str:
    return f"[{signed(low, digits)}, {signed(high, digits)}]"


def write_tex(output: Path, ladder: pd.DataFrame, primary: pd.DataFrame,
              secondary: pd.DataFrame, counts: pd.DataFrame) -> None:
    tex = output / "tex"
    tex.mkdir()
    memory = pd.read_csv(ROOT / "data" / "system_memory.csv", keep_default_na=False)
    gb = pd.to_numeric(memory["accelerator_memory_gb"].replace("", np.nan))
    types = memory["accelerator_memory_type"].replace("", "missing").value_counts()
    type_text = ", ".join(f"{k} {v}" for k, v in types.items())
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{p{0.32\textwidth}p{0.58\textwidth}}", r"\toprule",
        r"\textbf{Quantity} & \textbf{Value} \\", r"\midrule",
        f"Systems covered & {len(memory)} of 105 \\\\",
        f"Linked via \\texttt{{summary.csv}} / by hand & "
        f"{(memory['match_method'] == 'summary_row_match').sum()} / "
        f"{(memory['match_method'] == 'manual').sum()} \\\\",
        f"Distinct capacities (GB) & {gb.nunique()} (range {gb[gb > 0].min():.0f}--{gb.max():.0f}; "
        f"CPU-only 0) \\\\",
        f"Memory type counts & {type_text} \\\\",
        f"Normalization notes & {(memory['normalization_note'] != '').sum()} systems \\\\",
        f"Missing capacity & {gb.isna().sum()} system (reported N/A); imputed within each training fold \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Accelerator-memory features added in the post-lock exploratory "
        r"ablation, built from the MLPerf Inference v6.1 system-description JSONs "
        r"(\texttt{data/system\_memory.csv}).}",
        r"\label{tab:memory-features}", r"\end{table}",
    ]
    (tex / "tab_memory_features.tex").write_text("\n".join(lines) + "\n")

    rows = []
    for row in primary.itertuples(index=False):
        rows.append(
            f"{PROTOCOL_LABELS[row.protocol]} & {row.mean_difference_frozen:.6f} "
            f"{ci(row.difference_95ci_low_frozen, row.difference_95ci_high_frozen, 6)} & "
            f"{row.status_frozen} & {row.mean_difference:.6f} "
            f"{ci(row.difference_95ci_low, row.difference_95ci_high, 6)} & {row.status} \\\\"
        )
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Protocol} & \textbf{Frozen features} & \textbf{Label} & "
        r"\textbf{Memory features} & \textbf{Label} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}",
        r"\caption{Best-GNN-minus-MLP primary-error difference [95\% system-clustered "
        r"interval] with frozen and with memory features. The memory-feature labels "
        r"apply the frozen $\pm0.05$ rule but are post-lock exploratory results; they "
        r"do not replace the frozen co-primary label.}",
        r"\label{tab:memory-decision}", r"\end{table}",
    ]
    (tex / "tab_memory_decision.tex").write_text("\n".join(lines) + "\n")

    rows = []
    for method in FOCUS_METHODS:
        cells = [METHOD_LABELS[method]]
        for protocol in ["cv5", "loso"]:
            r = ladder.loc[ladder["protocol"].eq(protocol) & ladder["method"].eq(method)].iloc[0]
            cells += [
                f"{r.primary_error_frozen:.3f}",
                f"{r.primary_error_memory:.3f}",
                f"{signed(r.memory_minus_frozen)} {ci(r.memory_minus_frozen_95ci_low, r.memory_minus_frozen_95ci_high)}",
            ]
        rows.append(" & ".join(cells) + r" \\")
    lines = [
        r"\begin{table*}[htbp]", r"\centering", r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Repeated CV5}} & \multicolumn{3}{c}{\textbf{LOSO}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Method} & \textbf{Frozen} & \textbf{Memory} & \textbf{Memory $-$ frozen [95\% CI]} & "
        r"\textbf{Frozen} & \textbf{Memory} & \textbf{Memory $-$ frozen [95\% CI]} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}",
        r"\caption{Primary error with frozen and with memory features. Intervals are "
        r"paired system-clustered bootstrap intervals for the memory-minus-frozen "
        r"difference (same 2,000 system resamples in both runs). Negative differences "
        r"favor the memory features. $^{a}$The hierarchical mixed model already had an "
        r"accelerator-memory covariate parsed from accelerator names (available for 41 "
        r"of 105 systems); the memory feature set replaces it with the system-description "
        r"capacity.}",
        r"\label{tab:memory-ladder}", r"\end{table*}",
    ]
    (tex / "tab_memory_ladder.tex").write_text("\n".join(lines) + "\n")

    pair_labels = {
        ("graphsage", "random_forest"): "GraphSAGE $-$ random forest",
        ("graphsage", "graphsage_shuffled"): "GraphSAGE $-$ shuffled edges",
        ("gcn", "graphsage"): "GCN $-$ GraphSAGE",
        ("gat", "graphsage"): "GAT $-$ GraphSAGE",
    }
    rows = []
    for (left, right), text in pair_labels.items():
        for protocol in ["cv5", "loso"]:
            r = secondary.loc[
                secondary["left"].eq(left) & secondary["right"].eq(right)
                & secondary["protocol"].eq(protocol)
            ].iloc[0]
            p_text = "$<0.001$" if r.holm_adjusted_p_value < 0.001 else f"{r.holm_adjusted_p_value:.3f}"
            rows.append(
                f"{text} & {PROTOCOL_LABELS[protocol]} & {signed(r.mean_difference)} & "
                f"{ci(r.difference_95ci_low, r.difference_95ci_high)} & {r.status} & {p_text} \\\\"
            )
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
        r"\textbf{Comparison} & \textbf{Protocol} & \textbf{Difference} & "
        r"\textbf{95\% CI} & \textbf{Label} & \textbf{Holm $p$} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}",
        r"\caption{Secondary comparisons with memory features (exploratory). Holm "
        r"correction is applied across the eight tests, as in the frozen analysis.}",
        r"\label{tab:memory-secondary}", r"\end{table}",
    ]
    (tex / "tab_memory_secondary.tex").write_text("\n".join(lines) + "\n")

    rows = [
        f"{FAMILY_LABELS[r.family_id]} & {int(r.beats_group_median_frozen)} of {int(r.models)} & "
        f"{int(r.beats_group_median_memory)} of {int(r.models)} \\\\"
        for r in counts.itertuples(index=False)
    ]
    rows.insert(len(rows) - 1, r"\midrule")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"\textbf{Held-out family} & \textbf{Frozen features} & \textbf{Memory features} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{LOAFO headline families: number of the six trained non-baseline "
        r"models (Ridge, random forest, MLP, GraphSAGE, GCN, GAT) whose primary error "
        r"is below the group median. Descriptive and unpooled.}",
        r"\label{tab:memory-loafo}", r"\end{table}",
    ]
    (tex / "tab_memory_loafo.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    for protocol in PREDICTION_FILES:
        check_same_population(protocol)
    ladder = pd.concat(
        [ladder_comparison(p) for p in ["cv5", "loso", "sensitivity"]],
        ignore_index=True,
    )
    ladder.to_csv(output / "ladder_memory_vs_frozen.csv", index=False)
    primary, secondary, decision = memory_pair_decisions()
    primary.to_csv(output / "primary_decision_memory.csv", index=False)
    secondary.to_csv(output / "secondary_holm_memory.csv", index=False)
    (output / "decision_memory.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    loafo = loafo_comparison()
    loafo.to_csv(output / "loafo_memory_vs_frozen.csv", index=False)
    counts = loafo_counts(loafo)
    counts.to_csv(output / "loafo_beats_group_median_counts.csv", index=False)
    tail_comparison().to_csv(output / "tail_memory_vs_frozen.csv", index=False)

    write_tex(output, ladder, primary, secondary, counts)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-lock exploratory ablation; not a co-primary result",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "memory_table_sha256": sha256_file(ROOT / "data" / "system_memory.csv"),
        "frozen_bundles": {k: str(v.relative_to(ROOT)) for k, v in FROZEN.items()},
        "memory_bundles": {k: str(v.relative_to(ROOT)) for k, v in MEMORY.items()},
        "memory_manifests_sha256": {
            k: sha256_file(v / "results_manifest.json") for k, v in MEMORY.items()
        },
        "outputs": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted([*output.glob("*.csv"), *output.glob("tex/*.tex")])
        },
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
