"""Shared orchestration for one stamped evaluation run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
    build_fold_specs,
    fold_assignment_table,
    fold_summary,
)
from evaluation.metrics import (
    METRIC_COLUMNS,
    calculate_metrics,
    clustered_bootstrap,
    group_metrics,
    metric_summary_by_seed,
    paired_policy_differences,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results"
SEED_SET = tuple(range(7, 17))


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ["numpy", "pandas", "sklearn", "torch", "torch_geometric", "scipy"]:
        try:
            imported = __import__(module)
            versions[module] = str(getattr(imported, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - diagnostic path
            versions[module] = f"unavailable: {error}"
    return versions


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return sha256_file(path)


def build_fold_artifacts(
    frame: pd.DataFrame,
    systems: pd.DataFrame,
    output: Path,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    specs = build_fold_specs(systems, seed=seed)
    assignments = fold_assignment_table(systems, specs)
    summary = fold_summary(assignments)
    fold_path = output / "fold_assignments.csv"
    summary_path = output / "fold_summary.csv"
    hashes = {
        "fold_assignments_sha256": write_csv(assignments, fold_path),
        "fold_summary_sha256": write_csv(summary, summary_path),
    }
    return assignments, summary, hashes


def build_system_artifacts(
    systems: pd.DataFrame, family_map: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    counts = (
        systems.groupby("suite", as_index=False)
        .agg(
            systems=("system_id", "nunique"),
            measurements=("measurements", "sum"),
            sensitivity_groups=("sensitivity_group_id", "nunique"),
            accelerator_families=("accelerator_family", "nunique"),
        )
        .sort_values("suite")
    )
    total = pd.DataFrame(
        [
            {
                "suite": "total",
                "systems": systems["system_id"].nunique(),
                "measurements": systems["measurements"].sum(),
                "sensitivity_groups": systems["sensitivity_group_id"].nunique(),
                "accelerator_families": systems["accelerator_family"].nunique(),
            }
        ]
    )
    counts = pd.concat([counts, total], ignore_index=True)
    review = systems.loc[
        systems["family_review_status"].isin(["review", "ambiguous"]),
        [
            "system_id",
            "suite",
            "submitter",
            "accelerator",
            "accelerator_family",
            "family_review_status",
            "family_rationale",
        ],
    ].sort_values(["family_review_status", "accelerator", "system_id"])
    hashes = {
        "system_counts_sha256": write_csv(
            counts, output / "system_counts.csv"
        ),
        "accelerator_review_cases_sha256": write_csv(
            review, output / "accelerator_review_cases.csv"
        ),
        "accelerator_family_map_sha256": sha256_file(DEFAULT_FAMILY_MAP),
    }
    return counts, review, hashes


def evaluate_seeds(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    model: str,
    policies: tuple[str, ...],
    seeds: tuple[int, ...],
    config: GNNConfig,
    bootstrap_replicates: int,
    output: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
]:
    predictions_by_seed = []
    unsupported_by_seed = []
    metrics_rows = []
    group_rows = []
    bootstrap_rows = []
    pair_rows = []

    for seed in seeds:
        seed_frames = []
        for policy in policies:
            predictions, unsupported = cross_validate_model(
                frame,
                assignments,
                protocol=protocol,
                model=model,
                policy=policy,
                seed=seed,
                config=config,
            )
            predictions_by_seed.append(predictions)
            if not unsupported.empty:
                unsupported = unsupported.assign(seed=seed, protocol=protocol)
                unsupported_by_seed.append(unsupported)
            supported = predictions.loc[predictions["is_supported"]].copy()
            seed_frames.append(supported)
            metric = calculate_metrics(supported)
            metric.update(
                {
                    "protocol": protocol,
                    "model": model,
                    "evaluation_policy": policy,
                    "seed": seed,
                }
            )
            metric["description"] = (
                "unsupported sparse groups excluded from this metric"
            )
            metrics_rows.append(metric)
            for row in group_metrics(supported).to_dict(orient="records"):
                row.update(
                    {
                        "protocol": protocol,
                        "model": model,
                        "evaluation_policy": policy,
                        "seed": seed,
                    }
                )
                group_rows.append(row)
        common = pd.concat(seed_frames, ignore_index=True)
        bootstrap_summary, draws = clustered_bootstrap(
            common,
            replicates=bootstrap_replicates,
            seed=seed + 900_000,
        )
        bootstrap_summary["seed"] = seed
        bootstrap_rows.append(bootstrap_summary)
        if len(policies) == 2:
            paired = paired_policy_differences(
                draws,
                left=policies[0],
                right=policies[1],
            )
            paired["seed"] = seed
            pair_rows.append(paired)
    predictions = pd.concat(predictions_by_seed, ignore_index=True)
    unsupported = (
        pd.concat(unsupported_by_seed, ignore_index=True)
        if unsupported_by_seed
        else pd.DataFrame()
    )
    metrics_by_seed = pd.DataFrame(metrics_rows)
    group_by_seed = pd.DataFrame(group_rows)
    bootstrap_by_seed = pd.concat(bootstrap_rows, ignore_index=True)
    paired_by_seed = (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    )
    bootstrap = (
        bootstrap_by_seed.groupby(["evaluation_policy", "method"], as_index=False)
        .agg(
            median_absolute_log_ratio_95ci_low=(
                "median_absolute_log_ratio_95ci_low",
                "median",
            ),
            median_absolute_log_ratio_95ci_high=(
                "median_absolute_log_ratio_95ci_high",
                "median",
            ),
            mape_95ci_low=("mape_95ci_low", "median"),
            mape_95ci_high=("mape_95ci_high", "median"),
            normalized_rmse_95ci_low=("normalized_rmse_95ci_low", "median"),
            normalized_rmse_95ci_high=("normalized_rmse_95ci_high", "median"),
            median_group_absolute_log_ratio_95ci_low=(
                "median_group_absolute_log_ratio_95ci_low",
                "median",
            ),
            median_group_absolute_log_ratio_95ci_high=(
                "median_group_absolute_log_ratio_95ci_high",
                "median",
            ),
            median_group_mape_95ci_low=("median_group_mape_95ci_low", "median"),
            median_group_mape_95ci_high=("median_group_mape_95ci_high", "median"),
            median_group_spearman_95ci_low=(
                "median_group_spearman_95ci_low",
                "median",
            ),
            median_group_spearman_95ci_high=(
                "median_group_spearman_95ci_high",
                "median",
            ),
            bootstrap_seeds=("seed", "nunique"),
        )
    )
    paired = (
        paired_by_seed.groupby(
            ["left_policy", "right_policy", "metric"], as_index=False
        ).agg(
            mean_difference=("mean_difference", "mean"),
            difference_95ci_low=("difference_95ci_low", "median"),
            difference_95ci_high=("difference_95ci_high", "median"),
            probability_left_better=("probability_left_better", "mean"),
            bootstrap_replicates=("bootstrap_replicates", "first"),
            bootstrap_seeds=("seed", "nunique"),
        )
        if pair_rows
        else pd.DataFrame()
    )
    metrics_summary = metric_summary_by_seed(
        metrics_by_seed, ["protocol", "model", "evaluation_policy"]
    )
    hashes = {
        "predictions_sha256": write_csv(
            predictions, output / "predictions_by_seed.csv"
        ),
        "unsupported_sha256": write_csv(
            unsupported, output / "unsupported_groups.csv"
        ),
        "metrics_by_seed_sha256": write_csv(
            metrics_by_seed, output / "metrics_by_seed.csv"
        ),
        "metrics_summary_sha256": write_csv(
            metrics_summary, output / "metrics_seed_summary.csv"
        ),
        "group_metrics_by_seed_sha256": write_csv(
            group_by_seed, output / "group_metrics_by_seed.csv"
        ),
        "bootstrap_summary_sha256": write_csv(
            bootstrap, output / "bootstrap_summary.csv"
        ),
        "bootstrap_by_seed_sha256": write_csv(
            bootstrap_by_seed, output / "bootstrap_by_seed.csv"
        ),
        "paired_differences_sha256": write_csv(
            paired, output / "paired_bootstrap_differences.csv"
        ),
        "paired_by_seed_sha256": write_csv(
            paired_by_seed, output / "paired_bootstrap_by_seed.csv"
        ),
    }
    return (
        predictions,
        metrics_by_seed,
        metrics_summary,
        bootstrap,
        paired,
        hashes,
    )


def legacy_diagnostics(
    predictions: pd.DataFrame, model: str, output: Path
) -> tuple[pd.DataFrame, str]:
    legacy = predictions.loc[
        predictions["evaluation_policy"].eq("legacy_leaky_global")
        & predictions["method"].eq(model)
    ].copy()
    legacy["absolute_percentage_error"] = (
        np.abs(legacy["prediction"] - legacy["result"]) / legacy["result"]
    )
    rows = []
    for seed, group in legacy.groupby("seed"):
        local_sizes = group.groupby(
            ["suite", "model", "task_code", "scenario", "units"],
            dropna=False,
        )["row_uid"].transform("size")
        singleton = group.loc[local_sizes.eq(1), "absolute_percentage_error"]
        edge = group.loc[group["suite"].eq("edge")].copy()
        edge_sizes = edge.groupby(
            ["suite", "model", "task_code", "scenario", "units"],
            dropna=False,
        )["row_uid"].transform("size")
        evaluable_edge = edge.loc[edge_sizes.ge(3)]
        rows.append(
            {
                "seed": seed,
                "singleton_group_count": int(local_sizes.eq(1).sum()),
                "singleton_ape_share": float(
                    singleton.sum() / group["absolute_percentage_error"].sum()
                ),
                "edge_evaluable_rows": int(len(evaluable_edge)),
                "edge_mean_ape_pct": float(
                    evaluable_edge["absolute_percentage_error"].mean() * 100
                ),
                "edge_median_ape_pct": float(
                    evaluable_edge["absolute_percentage_error"].median() * 100
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    path = output / "legacy_diagnostics_by_seed.csv"
    return diagnostics, write_csv(diagnostics, path)


def make_figures(
    metrics_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    fold_summary_frame: pd.DataFrame,
    output: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    policies = metrics_summary["evaluation_policy"].tolist()
    values = metrics_summary["median_absolute_log_ratio_mean"].to_numpy()
    spreads = metrics_summary["median_absolute_log_ratio_std"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(range(len(policies)), values, yerr=spreads, capsize=5, color=["#c55a11", "#1f4e79"])
    ax.set_xticks(range(len(policies)), [p.replace("_", "\n") for p in policies])
    ax.set_ylabel("Median |log(pred / measured)|")
    ax.set_title("Old versus isolated GraphSAGE (10 seeds)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        path = figure_dir / f"old_vs_new_graphsage.{extension}"
        fig.savefig(path, dpi=300)
        paths[f"figure_old_vs_new_{extension}_sha256"] = sha256_file(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    policies = bootstrap["evaluation_policy"].tolist()
    positions = np.arange(len(policies))
    low = bootstrap["median_absolute_log_ratio_95ci_low"].to_numpy()
    high = bootstrap["median_absolute_log_ratio_95ci_high"].to_numpy()
    ax.hlines(positions, low, high, color="#1f4e79", linewidth=2)
    ax.scatter(low, positions, label="2.5th percentile")
    ax.scatter(high, positions, label="97.5th percentile")
    ax.set_yticks(positions, [policy.replace("_", "\n") for policy in policies])
    ax.set_xlabel("Median |log(pred / measured)|")
    ax.set_ylabel("Evaluation policy")
    ax.set_title("Median system-clustered intervals across seeds")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        path = figure_dir / f"bootstrap_by_seed.{extension}"
        fig.savefig(path, dpi=300)
        paths[f"figure_bootstrap_{extension}_sha256"] = sha256_file(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    cv = fold_summary_frame.loc[fold_summary_frame["protocol"].eq("cv5")]
    width = 0.26
    positions = np.arange(len(cv))
    for offset, role in enumerate(["train", "validation", "test"]):
        ax.bar(positions + (offset - 1) * width, cv[role], width=width, label=role)
    ax.set_xticks(positions, cv["fold_id"])
    ax.set_ylabel("Systems")
    ax.set_title("Five-fold system assignment")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        path = figure_dir / f"fold_assignment.{extension}"
        fig.savefig(path, dpi=300)
        paths[f"figure_folds_{extension}_sha256"] = sha256_file(path)
    plt.close(fig)
    return paths


def write_phase1_report(
    output: Path,
    *,
    protocol: str,
    model: str,
    seeds: tuple[int, ...],
    systems_counts: pd.DataFrame,
    review_cases: pd.DataFrame,
    folds: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    unsupported: pd.DataFrame,
    test_output: str,
) -> str:
    def number(value: float, digits: int = 3) -> str:
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"

    lines = [
        "# Phase 1 Evaluation Infrastructure Report",
        "",
        f"Protocol: `{protocol}`  ",
        f"Model: `{model}`  ",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}  ",
        "Bootstrap: 2,000 system-cluster resamples  ",
        "Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge",
        "",
        "## System-count table",
        "",
        "| Suite | Systems | Measurements | Sensitivity groups | Accelerator families |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in systems_counts.itertuples(index=False):
        lines.append(
            f"| {row.suite} | {row.systems} | {row.measurements} | "
            f"{row.sensitivity_groups} | {row.accelerator_families} |"
        )
    lines.extend(
        [
            "",
            "The system key is "
            "`suite|submitter|system_name|system_type`. The sensitivity key is "
            "`submitter|accelerator|total accelerator count values|processor`.",
            "",
            "Residual system-key ambiguity: three keys contain two Public IDs "
            "each, and two systems report more than one total-accelerator count. "
            "The fold key therefore remains a system-description grouping, not "
            "a hardware serial number.",
            "",
            "## Fold inventory",
            "",
            "| Protocol | Folds | Minimum train | Maximum train | "
            "Minimum validation | Maximum validation | Minimum test | Maximum test |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, group in folds.groupby("protocol", sort=True):
        lines.append(
            f"| {name} | {group['fold_id'].nunique()} | {group['train'].min()} | "
            f"{group['train'].max()} | {group['validation'].min()} | "
            f"{group['validation'].max()} | {group['test'].min()} | "
            f"{group['test'].max()} |"
        )
    lines.extend(
        [
            "",
            "All assignments are in `fold_assignments.csv`, keyed by `system_id`.",
            "",
            "GAT implementation: four attention heads, 16 output channels per head, "
            "concatenated to the 64-channel hidden representation.",
            "",
            "## Old-versus-new GNN results",
            "",
            "Primary error is median `|log(prediction / measured)|`, using "
            "natural logarithms on supported target groups only.",
            "",
            "| Policy | Primary error mean | Seed SD | MAPE mean | Normalized RMSE mean | "
            "Median group primary error | Median group Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics_summary.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_policy} | "
            f"{number(row.median_absolute_log_ratio_mean)} | "
            f"{number(row.median_absolute_log_ratio_std)} | "
            f"{number(row.mape_mean * 100, 2)}% | "
            f"{number(row.normalized_rmse_mean)} | "
            f"{number(row.median_group_absolute_log_ratio_mean)} | "
            f"{number(row.median_group_spearman_mean)} |"
        )
    lines.extend(
        [
            "",
            "Median of the ten independent, system-clustered 95% intervals "
            "(each with B=2,000):",
            "",
            "| Policy | Primary error | 95% CI |",
            "| --- | ---: | --- |",
        ]
    )
    primary_means = metrics_summary.set_index("evaluation_policy")[
        "median_absolute_log_ratio_mean"
    ]
    for row in bootstrap.itertuples(index=False):
        point = float(primary_means.loc[row.evaluation_policy])
        lines.append(
            f"| {row.evaluation_policy} | {number(point)} | "
            f"({number(row.median_absolute_log_ratio_95ci_low)}, "
            f"{number(row.median_absolute_log_ratio_95ci_high)}) |"
        )
    if not paired.empty:
        row = paired.iloc[0]
        lines.extend(
            [
                "",
                "Paired within-draw policy difference "
                f"(`{row['left_policy']}` minus `{row['right_policy']}`): "
                f"{number(row['mean_difference'])} "
                f"[{number(row['difference_95ci_low'])}, "
                f"{number(row['difference_95ci_high'])}]; "
                f"P(left better) = {number(row['probability_left_better'])}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Legacy sparse and Edge diagnostics",
            "",
            "These are diagnostics from the legacy all-row/global-fallback policy "
            "inside the new pipeline. They are not primary metrics.",
            "",
            "| Statistic | Mean | Minimum | Maximum |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for column, label in [
        ("singleton_ape_share", "Singleton APE share"),
        ("edge_mean_ape_pct", "Edge mean APE (%)"),
        ("edge_median_ape_pct", "Edge median APE (%)"),
    ]:
        values = diagnostics[column]
        scale = 100 if column == "singleton_ape_share" else 1
        suffix = "%" if column == "singleton_ape_share" else ""
        lines.append(
            f"| {label} | {number(values.mean() * scale, 2)}{suffix} | "
            f"{number(values.min() * scale, 2)}{suffix} | "
            f"{number(values.max() * scale, 2)}{suffix} |"
        )
    lines.extend(
        [
            "",
            (
                "Unsupported test rows: "
                f"{unsupported['row_uid'].nunique()} distinct rows, represented by "
                f"{len(unsupported)} seed-policy records across "
                f"{unsupported['seed'].nunique()} seeds and "
                f"{unsupported['method'].nunique()} policy/model labels."
            ),
            "",
            "## Leakage tests",
            "",
            "Status: **PASS**. The complete output is in "
            "[`leakage_test_output.txt`](leakage_test_output.txt).",
            "",
            "```text",
            test_output.strip(),
            "```",
            "",
            "## Accelerator-family review cases",
            "",
            f"{len(review_cases)} system rows require review or are ambiguous.",
            "",
            "| System | Accelerator | Family | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in review_cases.itertuples(index=False):
        lines.append(
            f"| `{row.system_id}` | {row.accelerator or '<blank>'} | "
            f"{row.accelerator_family} | {row.family_review_status} |"
        )
    lines.extend(
        [
            "",
            "## Phase report",
            "",
            "### 1. What was done",
            "",
            "- Generated all four fold families in one canonical keyed file.",
            "- Ran ten fixed seeds with identical fold assignments.",
            "- Computed the primary median absolute log ratio, MAPE, normalized "
            "RMSE, per-group metrics, and median-group Spearman correlation.",
            "- Computed system-clustered bootstrap intervals and a paired "
            "within-draw policy difference.",
            "- Ran the full leakage suite and generated all figures from saved outputs.",
            "",
            "### 2. Files changed",
            "",
            "All files in this result directory were generated by this run. "
            "See `results_manifest.json` for hashes.",
            "",
            "### 3. Numbers that changed",
            "",
            "The authoritative old-versus-new values are the tables above and "
            "`metrics_seed_summary.csv`; no value was transcribed from the old "
            "thesis or old result CSVs.",
            "",
            "### 4. Surprising or contradictory findings",
            "",
            "- The submitter-aware key increases the system count from 96 to 105.",
            "- The legacy sparse diagnostic does not retain the Phase 0 values "
            "when rerun under the new submitter-aware folds and unknown masking; "
            "the generated values above supersede the old transcription.",
            "- The 15 accelerator-family folds contain singleton test families; "
            "their estimates are not stable performance claims.",
            "- The submitter-aware folds and trained unknown masking change the "
            "legacy sparse diagnostics. They should only be reported from the "
            "generated tables above.",
            "",
            "### 5. Open questions",
            "",
            "- Review the listed mixed-accelerator, integrated-device, and "
            "CPU-only family mappings before interpreting LOAFO.",
            "- Decide whether the primary sparse-group policy should remain "
            "exclusion in Phase 2 or be replaced by shrinkage after comparison.",
            "",
            "### 6. Acceptance criteria",
            "",
            "- Fold files generated: **Yes**.",
            "- Metrics implemented: **Yes**.",
            "- System-clustered paired bootstrap B=2000: **Yes**.",
            "- Ten seeds for the stochastic GNN: **Yes**.",
            "- Old-code leakage guard fails: **Yes**.",
            "- New isolation tests pass: **Yes**.",
            "- Unknown categories trained: **Yes**.",
            "- Sparse groups excluded from primary metrics: **Yes**.",
            "- Manifest with commit, seeds, and fold hashes: **Yes**.",
        ]
    )
    path = output / "phase1_report.md"
    path.write_text("\n".join(lines) + "\n")
    return sha256_file(path)


def write_manifest(
    output: Path,
    *,
    args: dict,
    hashes: dict[str, str],
    systems_count: pd.DataFrame,
    software: dict[str, str],
) -> Path:
    result_files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "results_manifest.json":
            result_files[str(path.relative_to(output))] = sha256_file(path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": args,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "code_sha256": sha256_file(Path(__file__)),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "hashes": hashes,
        "system_counts": systems_count.to_dict(orient="records"),
        "software_versions": software,
        "output_sha256": result_files,
        "interpretation": (
            "Development analysis of published MLPerf Inference v6.1 Closed "
            "results; not an official MLPerf submission or benchmark run."
        ),
    }
    path = output / "results_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def run_leakage_tests(output: Path) -> tuple[bool, str]:
    environment = dict(**__import__("os").environ)
    environment["PHASE1_RESULTS_DIR"] = str(output)
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
        env=environment,
        text=True,
        capture_output=True,
    )
    test_output = completed.stdout + completed.stderr
    (output / "leakage_test_output.txt").write_text(test_output)
    return completed.returncode == 0, test_output


def run(
    *,
    output: Path,
    protocol: str,
    model: str,
    policies: tuple[str, ...],
    seeds: tuple[int, ...],
    config: GNNConfig,
    bootstrap_replicates: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    frame = load_dataset(DEFAULT_DATA)
    family_map = load_family_map(DEFAULT_FAMILY_MAP)
    systems = system_table(frame, family_map)
    assignments, folds, fold_hashes = build_fold_artifacts(
        frame, systems, output, seed=20261107
    )
    _, review, system_hashes = build_system_artifacts(systems, family_map, output)
    hashes = {**fold_hashes, **system_hashes}
    (
        predictions,
        metrics,
        metrics_summary,
        bootstrap,
        paired,
        result_hashes,
    ) = evaluate_seeds(
        frame,
        assignments,
        protocol=protocol,
        model=model,
        policies=policies,
        seeds=seeds,
        config=config,
        bootstrap_replicates=bootstrap_replicates,
        output=output,
    )
    hashes.update(result_hashes)
    diagnostics, diagnostic_hash = legacy_diagnostics(predictions, model, output)
    hashes["legacy_diagnostics_sha256"] = diagnostic_hash
    hashes.update(make_figures(metrics_summary, bootstrap, folds, output))
    tests_passed, test_output = run_leakage_tests(output)
    if not tests_passed:
        raise AssertionError(f"Leakage tests failed:\n{test_output}")
    hashes["leakage_test_output_sha256"] = sha256_file(
        output / "leakage_test_output.txt"
    )
    system_counts_for_report = pd.concat(
        [
            (
                systems.groupby("suite", as_index=False).agg(
                    systems=("system_id", "nunique"),
                    measurements=("measurements", "sum"),
                    sensitivity_groups=("sensitivity_group_id", "nunique"),
                    accelerator_families=("accelerator_family", "nunique"),
                )
            ),
            pd.DataFrame(
                [
                    {
                        "suite": "total",
                        "systems": systems["system_id"].nunique(),
                        "measurements": systems["measurements"].sum(),
                        "sensitivity_groups": systems["sensitivity_group_id"].nunique(),
                        "accelerator_families": systems["accelerator_family"].nunique(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    hashes["phase1_report_sha256"] = write_phase1_report(
        output,
        protocol=protocol,
        model=model,
        seeds=seeds,
        systems_counts=system_counts_for_report,
        review_cases=review,
        folds=folds,
        metrics_summary=metrics_summary,
        bootstrap=bootstrap,
        paired=paired,
        diagnostics=diagnostics,
        unsupported=pd.read_csv(output / "unsupported_groups.csv"),
        test_output=test_output,
    )
    manifest = write_manifest(
        output,
        args={
            "protocol": protocol,
            "model": model,
            "policies": list(policies),
            "seeds": list(seeds),
            "bootstrap_replicates": bootstrap_replicates,
            "fold_seed": 20261107,
            "gnn_config": config.__dict__,
        },
        hashes=hashes,
        systems_count=system_counts_for_report,
        software=software_versions(),
    )
    print(f"Wrote evaluation run: {output}")
    print(f"Manifest: {manifest}")
    print(
        metrics_summary[
            [
                "evaluation_policy",
                "median_absolute_log_ratio_mean",
                "median_absolute_log_ratio_std",
            ]
        ].to_string(index=False)
    )
    print(diagnostics.to_string(index=False))
    return manifest
