#!/usr/bin/env python3
"""Generate publication-ready figures for the MLPerf GraphSAGE thesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "thesis_figures"

INK = "#17202a"
MUTED = "#5f6b76"
GRID = "#d8dde3"
NAVY = "#1f4e79"
BLUE = "#4f81bd"
ORANGE = "#c55a11"
TEAL = "#2e7d6e"
GOLD = "#d6a329"
LIGHT_BLUE = "#dce6f1"
LIGHT_ORANGE = "#f4dfd0"
LIGHT_GRAY = "#edf0f3"

METHOD_COLORS = {
    "GraphSAGE": NAVY,
    "Random forest": ORANGE,
    "Ridge": NAVY,
    "k-NN": ORANGE,
}


def read_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=7.3,
        linespacing=1.25,
    )


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def plot_scope_and_coverage(output_dir: Path, formats: tuple[str, ...]) -> None:
    scope = read_data("thesis_scope.csv").set_index("metric")["value"]
    workloads = read_data("workload_counts.csv").sort_values(
        "total_cells", ascending=True
    )

    fig, (ax_scope, ax_workloads) = plt.subplots(
        1,
        2,
        figsize=(9.2, 5.1),
        gridspec_kw={"width_ratios": [0.82, 1.18], "wspace": 0.42},
    )

    labels = ["Datacenter", "Edge"]
    cells = [
        int(scope["datacenter_result_cells"]),
        int(scope["edge_result_cells"]),
    ]
    submissions = [
        int(scope["datacenter_submissions"]),
        int(scope["edge_submissions"]),
    ]
    x = np.arange(len(labels))
    bars = ax_scope.bar(
        x,
        cells,
        color=[NAVY, TEAL],
        width=0.62,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, count, submission_count in zip(bars, cells, submissions):
        ax_scope.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{count}\n{submission_count} submissions",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    ax_scope.set_xticks(x, labels)
    ax_scope.set_ylim(0, 565)
    ax_scope.set_ylabel("Result cells")
    ax_scope.set_title("MLPerf Inference v6.1 / Closed", loc="left")
    clean_axis(ax_scope, "y")
    ax_scope.text(
        0.54,
        0.9,
        f"{int(scope['combined_systems'])} systems\n"
        f"{int(scope['combined_workload_groups'])} workload groups",
        transform=ax_scope.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": LIGHT_GRAY,
            "edgecolor": GRID,
        },
    )

    colors = [
        TEAL
        if dc > 0 and edge > 0
        else NAVY
        if dc > 0
        else ORANGE
        for dc, edge in zip(
            workloads["datacenter_cells"], workloads["edge_cells"]
        )
    ]
    y = np.arange(len(workloads))
    ax_workloads.barh(
        y,
        workloads["total_cells"],
        color=colors,
        height=0.68,
        alpha=0.95,
    )
    workload_labels = [
        str(label)
        .replace("End-to-End Vector DB", "End-to-End\nVector DB")
        .replace("End-to-End QA", "End-to-End\nQA")
        for label in workloads["workload"]
    ]
    ax_workloads.set_yticks(y, workload_labels)
    ax_workloads.tick_params(axis="y", labelsize=8.6)
    for position, value in enumerate(workloads["total_cells"]):
        ax_workloads.text(
            value + 2,
            position,
            f"{int(value)}",
            va="center",
            fontsize=8.2,
            color=INK,
        )
    ax_workloads.set_xlim(0, 124)
    ax_workloads.set_xlabel("Held-out result cells")
    ax_workloads.set_title("Measurements vary sharply by workload", loc="left")
    clean_axis(ax_workloads, "x")

    add_footer(
        fig,
        "Source: current thesis data snapshot, MLPerf Inference v6.1, Closed "
        "division. Navy = Datacenter; orange = Edge; teal = mixed.\n"
        "Model-development analysis, not an official MLPerf submission.",
    )
    fig.subplots_adjust(bottom=0.18, left=0.1, right=0.98)
    save_figure(fig, output_dir, "fig01_data_scope", formats)


def plot_associations(output_dir: Path, formats: tuple[str, ...]) -> None:
    associations = read_data("associations.csv").sort_values(
        "spearman_rho", ascending=True
    )
    y = np.arange(len(associations))
    colors = [
        ORANGE if value < 0 else NAVY for value in associations["spearman_rho"]
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    ax.hlines(y, 0, associations["spearman_rho"], color=colors, linewidth=2.2)
    ax.scatter(
        associations["spearman_rho"],
        y,
        s=65,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    labels = [
        f"{workload}  (n={int(n)})"
        for workload, n in zip(associations["workload"], associations["n"])
    ]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    for position, (value, metric) in enumerate(
        zip(associations["spearman_rho"], associations["metric"])
    ):
        label = f"{value:+.2f}"
        ax.text(
            value + (0.035 if value >= 0 else -0.035),
            position,
            label,
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8.2,
            color=colors[position],
            weight="bold",
        )
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
    ax.set_xlabel("Spearman rank association with accelerator count")
    clean_axis(ax, "x")
    add_footer(
        fig,
        "Throughput rows: higher is better. Text-to-video: raw latency, where "
        "lower is better.\nAssociations are exploratory and do not identify "
        "causal hardware effects. Systems with zero accelerators are excluded.",
    )
    fig.subplots_adjust(bottom=0.2, left=0.35, right=0.97)
    save_figure(fig, output_dir, "fig02_accelerator_associations", formats)


def plot_metadata_baseline(output_dir: Path, formats: tuple[str, ...]) -> None:
    data = read_data("metadata_baseline.csv").sort_values(
        ["ridge_mape_pct", "knn_mape_pct"], ascending=False
    )
    y = np.arange(len(data))
    height = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.barh(
        y + height / 2,
        data["ridge_mape_pct"],
        height=height,
        color=METHOD_COLORS["Ridge"],
        label="Ridge",
    )
    ax.barh(
        y - height / 2,
        data["knn_mape_pct"],
        height=height,
        color=METHOD_COLORS["k-NN"],
        label="k-NN",
    )
    for position, (ridge, knn) in enumerate(
        zip(data["ridge_mape_pct"], data["knn_mape_pct"])
    ):
        ax.text(
            ridge + 2,
            position + height / 2,
            f"{ridge:.1f}%",
            va="center",
            fontsize=8,
            color=NAVY,
        )
        ax.text(
            knn + 2,
            position - height / 2,
            f"{knn:.1f}%",
            va="center",
            fontsize=8,
            color=ORANGE,
        )
    ax.set_yticks(y, data["workload"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Held-out MAPE (%, lower is better)")
    ax.legend(loc="upper right", ncols=2)
    clean_axis(ax, "x")
    add_footer(
        fig,
        "Metadata-only reference analysis with leave-one-system-out validation.\n"
        "These are not directly comparable with the five-fold GraphSAGE "
        "development run.",
    )
    fig.subplots_adjust(bottom=0.18, left=0.29, right=0.96)
    save_figure(fig, output_dir, "fig03_metadata_baselines", formats)


def plot_headline_comparison(output_dir: Path, formats: tuple[str, ...]) -> None:
    data = read_data("headline_comparison.csv")
    metrics = [
        (
            "pooled_mape_pct",
            "Pooled MAPE",
            "Percent",
            (0, 660),
            "{:.1f}%",
        ),
        (
            "median_group_mape_pct",
            "Median group MAPE",
            "Percent",
            (0, 82),
            "{:.1f}%",
        ),
        (
            "median_row_mape_pct",
            "Median row MAPE",
            "Percent",
            (0, 34),
            "{:.1f}%",
        ),
        (
            "normalized_rmse",
            "Normalized RMSE",
            "Log-target units",
            (0, 1.72),
            "{:.3f}",
        ),
        (
            "median_group_spearman",
            "Median group Spearman rho",
            "Rank correlation",
            (0, 0.86),
            "{:.3f}",
        ),
    ]
    methods = ["GraphSAGE", "Random forest"]
    fig, axes = plt.subplots(1, 5, figsize=(12.4, 3.7))
    for index, (ax, metric_info) in enumerate(zip(axes, metrics)):
        metric, title, ylabel, limits, number_format = metric_info
        subset = (
            data[data["metric"].eq(metric)]
            .set_index("method")
            .loc[methods]
            .reset_index()
        )
        values = subset["value"].to_numpy()
        bars = ax.bar(
            np.arange(len(methods)),
            values,
            color=[METHOD_COLORS[item] for item in methods],
            width=0.68,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + limits[1] * 0.025,
                number_format.format(value),
                ha="center",
                va="bottom",
                fontsize=8.2,
                weight="bold",
            )
        ax.set_ylim(*limits)
        ax.set_xticks(np.arange(len(methods)), ["GraphSAGE", "Random\nforest"])
        ax.set_title(title, fontsize=9.8)
        if index == 0:
            ax.set_ylabel(ylabel)
        clean_axis(ax, "y")
    fig.text(
        0.5,
        0.91,
        "Neither model wins every metric: GraphSAGE has lower pooled and group "
        "error; random forest has lower typical-row error and higher rank "
        "correlation.",
        ha="center",
        va="top",
        fontsize=9.4,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": LIGHT_GRAY,
            "edgecolor": GRID,
        },
    )
    add_footer(
        fig,
        "Five system-held-out development folds. MAPE is on the original "
        "measurement scale; RMSE is on the normalized logarithmic target.\n"
        "Source: current Overleaf thesis, section 5.4 (commit 536f2de).",
    )
    fig.subplots_adjust(top=0.76, bottom=0.24, left=0.07, right=0.98, wspace=0.36)
    save_figure(fig, output_dir, "fig04_model_comparison", formats)


def plot_workload_performance(output_dir: Path, formats: tuple[str, ...]) -> None:
    data = read_data("workload_performance.csv").sort_values(
        "median_ape_pct", ascending=False
    )
    y = np.arange(len(data))
    labels = [
        f"{workload}  (n={int(n)})"
        for workload, n in zip(data["workload"], data["n"])
    ]

    fig, (ax_error, ax_rank) = plt.subplots(
        1,
        2,
        figsize=(9.4, 5.1),
        sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.12},
    )
    ax_error.barh(
        y,
        data["median_ape_pct"],
        color=NAVY,
        height=0.62,
        alpha=0.94,
    )
    for position, value in enumerate(data["median_ape_pct"]):
        ax_error.text(
            value + 0.8,
            position,
            f"{value:.1f}%",
            va="center",
            fontsize=8.2,
        )
    ax_error.set_xlim(0, 41)
    ax_error.set_xlabel("Median APE (%, lower is better)")
    ax_error.set_title("Typical absolute error", loc="left")
    clean_axis(ax_error, "x")

    rank_colors = [
        "#b9cce3" if rho < 0.75 else BLUE if rho < 0.9 else NAVY
        for rho in data["spearman_rho"]
    ]
    ax_rank.hlines(y, 0, data["spearman_rho"], color=GRID, linewidth=2)
    ax_rank.scatter(
        data["spearman_rho"],
        y,
        s=62,
        color=rank_colors,
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )
    for position, value in enumerate(data["spearman_rho"]):
        ax_rank.text(
            value + 0.025,
            position,
            f"{value:.2f}",
            va="center",
            fontsize=8.2,
            weight="bold",
            color=INK,
        )
    ax_rank.axvline(0, color=MUTED, linewidth=0.9)
    ax_rank.set_xlim(-0.05, 1.05)
    ax_rank.set_xlabel("Spearman rho (higher is better)")
    ax_rank.set_title("Within-workload ordering", loc="left")
    clean_axis(ax_rank, "x")

    ax_error.set_yticks(y, labels)
    ax_error.tick_params(axis="y", length=0)
    add_footer(
        fig,
        "Selected Datacenter Closed workloads with at least five held-out "
        "measurements. Low error does not guarantee good rank ordering (and "
        "vice versa).\nSource: current Overleaf thesis, table 5.5 "
        "(commit 536f2de).",
    )
    fig.subplots_adjust(bottom=0.2, left=0.30, right=0.97)
    save_figure(fig, output_dir, "fig05_workload_diagnostics", formats)


def plot_sparse_group_diagnostics(
    output_dir: Path, formats: tuple[str, ...]
) -> None:
    concentration = read_data("error_concentration.csv")
    diagnostics = read_data("filtered_diagnostics.csv").set_index("method")
    balance = read_data("selection_balance.csv").set_index("model")
    methods = ["GraphSAGE", "Random forest"]

    fig = plt.figure(figsize=(10.2, 6.0))
    ax_concentration = fig.add_axes([0.07, 0.62, 0.9, 0.23])
    axes = [
        fig.add_axes([0.07, 0.13, 0.24, 0.3]),
        fig.add_axes([0.38, 0.13, 0.24, 0.3]),
        fig.add_axes([0.69, 0.13, 0.24, 0.3]),
    ]

    left = 0.0
    for _, row in concentration.iterrows():
        color = ORANGE if row["group_class"] == "Singleton groups" else BLUE
        ax_concentration.barh(
            [0],
            [row["error_share_pct"]],
            left=left,
            height=0.52,
            color=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax_concentration.text(
            left + row["error_share_pct"] / 2,
            0,
            f"{row['error_share_pct']:.1f}%\n"
            f"{int(row['groups'])} groups",
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=10,
        )
        left += row["error_share_pct"]
    ax_concentration.set_xlim(0, 100)
    ax_concentration.set_ylim(-0.55, 0.55)
    ax_concentration.set_yticks([])
    ax_concentration.set_xlabel("Share of GraphSAGE's summed absolute percentage error")
    ax_concentration.set_title(
        "Five singleton groups dominate pooled error", loc="left"
    )
    clean_axis(ax_concentration, "x")

    panel_specs = [
        ("mean_mape_pct", "Mean MAPE", "Percent", (0, 184), "{:.1f}%"),
        ("median_mape_pct", "Median MAPE", "Percent", (0, 31), "{:.1f}%"),
        ("within_50_pct", "Within +/-50%", "Percent of rows", (0, 85), "{:.1f}%"),
    ]
    for ax, (field, title, ylabel, limits, number_format) in zip(
        axes, panel_specs
    ):
        values = diagnostics.loc[methods, field].to_numpy()
        bars = ax.bar(
            np.arange(2),
            values,
            color=[METHOD_COLORS[item] for item in methods],
            width=0.68,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + limits[1] * 0.025,
                number_format.format(value),
                ha="center",
                va="bottom",
                fontsize=8.4,
                weight="bold",
            )
        ax.set_ylim(*limits)
        ax.set_xticks(np.arange(2), ["GraphSAGE", "Random\nforest"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        clean_axis(ax, "y")

    note = (
        "Groups with >=5 rows: GraphSAGE has the lower mean error, while random "
        "forest has the lower median error. "
        f"Lower median APE in {int(balance.loc['GraphSAGE', 'groups_with_lower_median_ape'])}"
        " GraphSAGE groups versus "
        f"{int(balance.loc['Random forest', 'groups_with_lower_median_ape'])}"
        " random-forest groups."
    )
    fig.text(
        0.5,
        0.505,
        note,
        ha="center",
        va="center",
        fontsize=8.8,
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": LIGHT_GRAY,
            "edgecolor": GRID,
        },
    )
    add_footer(
        fig,
        "The lower three panels exclude workload groups with fewer than five "
        "measurements. Pooled error is unstable for singleton groups.\n"
        "Source: current Overleaf thesis, section 5.4 (commit 536f2de).",
    )
    fig.subplots_adjust(bottom=0.15, left=0.07, right=0.98, top=0.95)
    save_figure(fig, output_dir, "fig06_sparse_group_diagnostics", formats)


def plot_edge_evidence(output_dir: Path, formats: tuple[str, ...]) -> None:
    data = read_data("edge_groups.csv").sort_values("rows", ascending=True)
    y = np.arange(len(data))
    colors = [
        ORANGE if rows < 3 else TEAL for rows in data["rows"]
    ]

    fig = plt.figure(figsize=(9.2, 4.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.85], wspace=0.25)
    ax = fig.add_subplot(grid[0, 0])
    cards = fig.add_subplot(grid[0, 1])
    ax.barh(y, data["rows"], color=colors, height=0.62)
    ax.axvline(3, color=MUTED, linestyle=(0, (3, 2)), linewidth=1)
    for position, value in enumerate(data["rows"]):
        ax.text(
            value + 0.12,
            position,
            str(int(value)),
            va="center",
            fontsize=8.5,
            weight="bold",
        )
    ax.set_yticks(y, data["group"])
    ax.set_xlim(0, 6)
    ax.set_xlabel("Measurements in workload group")
    ax.set_title("Edge groups are sparse", loc="left")
    clean_axis(ax, "x")

    cards.axis("off")
    callouts = [
        ("19", "Edge measurements"),
        ("8", "Edge workload groups"),
        ("4", "Groups with fewer than 3 rows"),
    ]
    for index, (value, label) in enumerate(callouts):
        y_position = 0.86 - index * 0.24
        cards.text(
            0.08,
            y_position + 0.055,
            value,
            fontsize=20,
            weight="bold",
            color=NAVY if index != 2 else ORANGE,
            transform=cards.transAxes,
        )
        cards.text(
            0.08,
            y_position - 0.035,
            label,
            fontsize=8.8,
            color=MUTED,
            transform=cards.transAxes,
        )
    cards.text(
        0.08,
        0.04,
        "14 / 19 rows in groups with >=3\n"
        "GraphSAGE: mean error 62.9%\n"
        "median 99.9% | 21.4% within +/-50%",
        fontsize=8.4,
        color=INK,
        transform=cards.transAxes,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": LIGHT_ORANGE,
            "edgecolor": "#e5b994",
        },
    )
    add_footer(
        fig,
        "The Edge sample is too small for a stable standalone generalization "
        "claim. Dashed threshold marks three rows per group.\n"
        "Source: current Overleaf thesis, section 5.4 (commit 536f2de).",
    )
    fig.subplots_adjust(bottom=0.2, left=0.31, right=0.98, top=0.94)
    save_figure(fig, output_dir, "fig07_edge_evidence", formats)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.37,
        body,
        ha="center",
        va="center",
        fontsize=7.9,
        color=MUTED,
        linespacing=1.35,
    )


def draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=MUTED,
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_pipeline_diagram(output_dir: Path, formats: tuple[str, ...]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax,
        0.015,
        0.58,
        0.18,
        0.22,
        "Datacenter export",
        "457 results\n87 systems",
        LIGHT_BLUE,
        BLUE,
    )
    draw_box(
        ax,
        0.015,
        0.25,
        0.18,
        0.22,
        "Edge export",
        "19 results\n9 systems",
        LIGHT_ORANGE,
        ORANGE,
    )
    draw_box(
        ax,
        0.255,
        0.58,
        0.18,
        0.22,
        "Canonical table",
        "476 result cells\n96 systems / 31 groups\nintegrity checks + units",
        LIGHT_GRAY,
        MUTED,
    )
    draw_box(
        ax,
        0.255,
        0.23,
        0.18,
        0.24,
        "Heterogeneous graph",
        "System / accelerator\nprocessor / workload\nresult",
        "#e5f1ee",
        TEAL,
    )
    draw_box(
        ax,
        0.515,
        0.58,
        0.19,
        0.22,
        "GraphSAGE",
        "2 x heterogeneous layers\n64 hidden units\n0.2 dropout",
        LIGHT_BLUE,
        NAVY,
    )
    draw_box(
        ax,
        0.515,
        0.25,
        0.19,
        0.22,
        "Tabular baselines",
        "group median / Ridge\nrandom forest",
        LIGHT_ORANGE,
        ORANGE,
    )
    draw_box(
        ax,
        0.79,
        0.34,
        0.19,
        0.31,
        "Validation",
        "Five system-held-out folds\nlog target + group normalization\n"
        "MAPE / RMSE / Spearman\nsame-fold diagnostics",
        LIGHT_GRAY,
        MUTED,
    )

    draw_arrow(ax, 0.195, 0.69, 0.255, 0.69)
    draw_arrow(ax, 0.195, 0.36, 0.255, 0.65)
    draw_arrow(ax, 0.345, 0.58, 0.345, 0.47)
    draw_arrow(ax, 0.435, 0.47, 0.515, 0.69)
    draw_arrow(ax, 0.435, 0.38, 0.515, 0.36)
    draw_arrow(ax, 0.705, 0.69, 0.79, 0.55)
    draw_arrow(ax, 0.705, 0.36, 0.79, 0.44)

    ax.text(
        0.345,
        0.515,
        "Clean and version inputs",
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        0.885,
        0.29,
        "Evaluate predictions in native units",
        ha="center",
        va="center",
        fontsize=8.2,
        color=NAVY,
        weight="bold",
    )
    ax.text(
        0.015,
        0.93,
        "Published MLPerf measurements -> reproducible GraphSAGE evaluation",
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.015,
        0.89,
        "Metadata prediction is not an official MLPerf submission or profiler measurement.",
        ha="left",
        va="top",
        fontsize=8.8,
        color=MUTED,
    )
    add_footer(
        fig,
        "The graph is a system/workload relationship graph. It is not an "
        "operator-level execution DAG.",
    )
    fig.subplots_adjust(bottom=0.12, left=0.01, right=0.99, top=0.99)
    save_figure(fig, output_dir, "fig08_evaluation_pipeline", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    formats = tuple(args.formats)
    args.out.mkdir(parents=True, exist_ok=True)
    plotters = [
        plot_scope_and_coverage,
        plot_associations,
        plot_metadata_baseline,
        plot_headline_comparison,
        plot_workload_performance,
        plot_sparse_group_diagnostics,
        plot_edge_evidence,
        plot_pipeline_diagram,
    ]
    for plotter in plotters:
        plotter(args.out, formats)
        print(f"Generated {plotter.__name__}")


if __name__ == "__main__":
    main()
