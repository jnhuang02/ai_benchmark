#!/usr/bin/env python3
"""Exploratory data-analysis figures and table for the thesis data chapter.

Reads only the canonical dataset, the accelerator-family map, and the
accelerator-memory table, so the output does not depend on any model run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INK = "#17202a"
MUTED = "#5f6b76"
GRID = "#d8dde3"
NAVY = "#1f4e79"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def load() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "data" / "mlperf_long.csv", keep_default_na=False)
    data["accelerator"] = data["accelerator"].astype(str)
    families = pd.read_csv(ROOT / "config" / "accelerator_families.csv", keep_default_na=False)
    families["accelerator"] = families["accelerator"].replace("<blank>", "")
    data = data.merge(families[["accelerator", "accelerator_family"]], on="accelerator", how="left")
    memory = pd.read_csv(ROOT / "data" / "system_memory.csv", keep_default_na=False)
    memory["accelerator_memory_gb"] = pd.to_numeric(
        memory["accelerator_memory_gb"].replace("", np.nan)
    )
    return data.merge(
        memory[["system_id", "accelerator_memory_gb", "accelerator_memory_type"]],
        on="system_id",
        how="left",
    )


def hbar(ax: plt.Axes, labels: list[str], values: list[int], xlabel: str) -> None:
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=NAVY, height=0.6)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for position, value in zip(positions, values):
        ax.text(value, position, f" {value}", va="center", fontsize=8, color=INK)


def fig_workloads(data: pd.DataFrame, out: Path) -> None:
    counts = data.groupby("model").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    hbar(ax, counts.index.tolist(), counts.tolist(), "Measurements")
    fig.savefig(out / "fig_eda_workloads.pdf")
    plt.close(fig)


def fig_families(data: pd.DataFrame, out: Path) -> None:
    systems = data.drop_duplicates("system_id")
    counts = systems["accelerator_family"].fillna("unmapped").value_counts()
    labels = [label.replace("_", " ") for label in counts.index]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    hbar(ax, labels, counts.tolist(), "Systems")
    fig.savefig(out / "fig_eda_families.pdf")
    plt.close(fig)


def fig_results_per_system(data: pd.DataFrame, out: Path) -> None:
    per_system = data.groupby("system_id").size()
    bins = np.arange(0.5, per_system.max() + 1.5)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.hist(per_system, bins=bins, color=NAVY, rwidth=0.85)
    ax.set_xlabel("Measurements per system")
    ax.set_ylabel("Systems")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_eda_results_per_system.pdf")
    plt.close(fig)


def fig_memory_performance(data: pd.DataFrame, out: Path) -> str:
    group = "datacenter|v6.1|closed|Inference|llama2-70b-99.9||Offline|Tokens/s"
    local = data.loc[data["target_group"].eq(group) & data["total_accelerators"].gt(0)].copy()
    local["per_accelerator"] = local["result"] / local["total_accelerators"]
    local = local.dropna(subset=["accelerator_memory_gb"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(
        local["accelerator_memory_gb"],
        local["per_accelerator"],
        s=28,
        color=NAVY,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Accelerator memory capacity (GB)")
    ax.set_ylabel("Tokens/s per accelerator (log)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_eda_memory_performance.pdf")
    plt.close(fig)
    rho = local[["accelerator_memory_gb", "per_accelerator"]].corr(method="spearman").iloc[0, 1]
    return f"{len(local)} results, Spearman {rho:.3f}"


def table_summary(data: pd.DataFrame, out: Path) -> None:
    per_system = data.groupby("system_id").size()
    per_group = data.groupby("target_group").size()
    rows = [
        ("Measurements", f"{len(data)}"),
        ("Systems", f"{data['system_id'].nunique()}"),
        ("Workload groups", f"{data['target_group'].nunique()}"),
        ("Models (workloads)", f"{data['model'].nunique()}"),
        ("Accelerator models", f"{data.loc[data['accelerator'].ne(''), 'accelerator'].nunique()}"),
        ("Host processors", f"{data['host_processor'].nunique()}"),
        ("Submitters", f"{data['organization'].nunique()}"),
        ("Measurements per system (median, range)",
         f"{per_system.median():.0f} ({per_system.min()}--{per_system.max()})"),
        ("Measurements per workload group (median, range)",
         f"{per_group.median():.0f} ({per_group.min()}--{per_group.max()})"),
        ("Workload groups with fewer than three measurements", f"{int((per_group < 3).sum())}"),
        ("Throughput / latency measurements",
         f"{int(data['direction'].eq(1).sum())} / {int(data['direction'].eq(-1).sum())}"),
        ("CPU-only systems",
         f"{data.drop_duplicates('system_id')['total_accelerators'].eq(0).sum()}"),
    ]
    body = "\n".join(f"{name} & {value} \\\\" for name, value in rows)
    text = (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\begin{tabular}{lr}\n\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Summary of the analysis dataset.}\n"
        "\\label{tab:eda-summary}\n\\end{table}\n"
    )
    (out / "tab_eda_summary.tex").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    data = load()
    fig_workloads(data, args.output)
    fig_families(data, args.output)
    fig_results_per_system(data, args.output)
    print("memory figure:", fig_memory_performance(data, args.output))
    table_summary(data, args.output)
    print("scenarios:", data["scenario"].value_counts().to_dict())
    print("units:", data["units"].value_counts().to_dict())
    print("unmapped systems:", data.drop_duplicates("system_id")["accelerator_family"].isna().sum())


if __name__ == "__main__":
    main()
