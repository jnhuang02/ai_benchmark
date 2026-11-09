#!/usr/bin/env python3
"""B4: GCN isolation replay and degree-normalization coefficient audit."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import GNNConfig, evaluate_fold_model
from evaluation.phase2 import load_phase2_frame
from exploratory_manifest import ALLOWED_LABEL, write_manifest


ROOT = Path(__file__).resolve().parent
FOLDS = ROOT / "folds" / "phase2_fold_assignments.csv"
PROTOCOL = "cv5_partition_0"
SEEDS = (7, 11, 16)
BOOTSTRAP_REPLICATES = 2000
POLICIES = (
    "legacy_leaky_global",
    "audit_supported_leaky",
    "isolated_supported",
)
NODE_IDENTIFIER = {
    "system": "system_id",
    "accelerator": "accelerator_id",
    "processor": "processor_id",
    "workload": "target_group",
    "result": "row_uid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_edges(
    graph, edge_type: tuple[str, str, str]
) -> dict[tuple[str, str], float]:
    source_type, _, destination_type = edge_type
    source_index, destination_index = (
        value.cpu().numpy() for value in graph[edge_type].edge_index
    )
    source_ids = getattr(graph[source_type], NODE_IDENTIFIER[source_type])
    destination_ids = getattr(
        graph[destination_type], NODE_IDENTIFIER[destination_type]
    )
    source_degree = torch_bincount(source_index, len(source_ids))
    destination_degree = torch_bincount(
        destination_index, len(destination_ids)
    )
    normalization = np.power(source_degree[source_index], -0.5) * np.power(
        destination_degree[destination_index], -0.5
    )
    return {
        (str(source_ids[source]), str(destination_ids[destination])): float(value)
        for source, destination, value in zip(
            source_index, destination_index, normalization
        )
    }


def torch_bincount(index, minimum_length: int) -> np.ndarray:
    return np.bincount(index, minlength=minimum_length).clip(min=1).astype(float)


def normalization_diagnostics(
    isolated_graph,
    legacy_graph,
    *,
    fold_id: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_rows = []
    relation_rows = []
    for edge_type in sorted(isolated_graph.edge_types):
        isolated = normalized_edges(isolated_graph, edge_type)
        legacy = normalized_edges(legacy_graph, edge_type)
        missing = sorted(set(isolated) - set(legacy))
        if missing:
            raise AssertionError(
                f"{fold_id}/{seed}/{edge_type} is missing training edges from legacy graph"
            )
        differences = []
        for edge, isolated_value in isolated.items():
            legacy_value = legacy[edge]
            difference = legacy_value - isolated_value
            differences.append(difference)
            edge_rows.append(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "source_type": edge_type[0],
                    "relation": edge_type[1],
                    "destination_type": edge_type[2],
                    "source_id": edge[0],
                    "destination_id": edge[1],
                    "isolated_normalization": isolated_value,
                    "legacy_normalization": legacy_value,
                    "legacy_minus_isolated": difference,
                }
            )
        differences_array = np.asarray(differences, dtype=float)
        changed = np.abs(differences_array) > 1e-12
        relation_rows.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "source_type": edge_type[0],
                "relation": edge_type[1],
                "destination_type": edge_type[2],
                "training_edges": len(differences),
                "changed_edges": int(changed.sum()),
                "changed_edge_share": float(changed.mean()),
                "mean_absolute_change": float(np.mean(np.abs(differences_array))),
                "p95_absolute_change": float(
                    np.quantile(np.abs(differences_array), 0.95)
                ),
                "max_absolute_change": float(
                    np.max(np.abs(differences_array))
                ),
            }
        )
    return pd.DataFrame(edge_rows), pd.DataFrame(relation_rows)


def aggregate_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "evaluation_policy",
        "protocol",
        "fold_id",
        "row_uid",
        "system_id",
        "target_group",
        "result",
        "direction",
    ]
    aggregated = predictions.groupby(identifiers, as_index=False).agg(
        signed_log_prediction=("signed_log_prediction", "mean"),
        y=("y", "mean"),
        seed_count=("seed", "nunique"),
    )
    aggregated["absolute_log_error"] = np.abs(
        aggregated["signed_log_prediction"]
        - aggregated["direction"] * np.log(aggregated["result"])
    )
    return aggregated


def system_bootstrap(aggregated: pd.DataFrame) -> pd.DataFrame:
    systems = np.array(sorted(aggregated["system_id"].unique()), dtype=object)
    by_policy: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for policy, local in aggregated.groupby("evaluation_policy", sort=True):
        local = local.reset_index(drop=True)
        positions = defaultdict(list)
        for index, system_id in enumerate(local["system_id"]):
            positions[str(system_id)].append(index)
        by_policy[policy] = {
            key: np.asarray(value, dtype=int)
            for key, value in positions.items()
        }
    rng = np.random.default_rng(20261107)
    rows = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(systems, size=len(systems), replace=True)
        row = {"bootstrap_replicate": replicate}
        for policy in POLICIES:
            local = aggregated.loc[aggregated["evaluation_policy"].eq(policy)]
            positions = by_policy[policy]
            selected = np.concatenate(
                [
                    positions[str(system)]
                    for system in sampled
                    if str(system) in positions
                ]
            )
            row[policy] = float(
                np.median(local["absolute_log_error"].to_numpy()[selected])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frame = load_phase2_frame()
    assignments = pd.read_csv(FOLDS)
    fold_ids = sorted(
        assignments.loc[
            assignments["protocol"].eq(PROTOCOL), "fold_id"
        ].unique()
    )
    prediction_frames = []
    edge_frames = []
    relation_frames = []
    pair_rows = []
    runtimes = []
    for fold_id in fold_ids:
        for seed in SEEDS:
            fold_started = time.perf_counter()
            results = {}
            for policy in POLICIES:
                result = evaluate_fold_model(
                    frame,
                    assignments,
                    protocol=PROTOCOL,
                    fold_id=fold_id,
                    model="gcn",
                    policy=policy,
                    seed=seed,
                    config=GNNConfig(),
                )
                results[policy] = result
                prediction_frames.append(result.predictions)
                pair_rows.append(
                    {
                        "fold_id": fold_id,
                        "seed": seed,
                        "policy": policy,
                        "prediction_rows": len(result.predictions),
                        "test_systems": len(result.test_systems),
                        "runtime_seconds": time.perf_counter() - fold_started,
                    }
                )
            controlled = results["audit_supported_leaky"]
            isolated = results["isolated_supported"]
            edges, relations = normalization_diagnostics(
                isolated.training_graph,
                controlled.training_graph,
                fold_id=fold_id,
                seed=seed,
            )
            edge_frames.append(edges)
            relation_frames.append(relations)
            runtimes.append(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "runtime_seconds": time.perf_counter() - fold_started,
                }
            )
            print(
                f"fold={fold_id} seed={seed} "
                f"elapsed_s={time.perf_counter() - fold_started:.3f}",
                flush=True,
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    duplicate = predictions.duplicated(
        ["evaluation_policy", "method", "protocol", "fold_id", "seed", "row_uid"]
    )
    if duplicate.any():
        raise AssertionError("GCN replay produced duplicate predictions")
    aggregated = aggregate_seed_predictions(predictions)
    draws = system_bootstrap(aggregated)
    primary_errors = (
        aggregated.groupby("evaluation_policy")["absolute_log_error"]
        .median()
        .rename("primary_error")
        .reset_index()
    )
    observed = primary_errors.set_index("evaluation_policy")["primary_error"]
    comparisons = [
        ("legacy_leaky_global", "isolated_supported"),
        ("audit_supported_leaky", "isolated_supported"),
    ]
    summary_rows = []
    draw_rows = []
    for left, right in comparisons:
        difference = draws[left] - draws[right]
        summary_rows.append(
            {
                "left_policy": left,
                "right_policy": right,
                "observed_left_error": float(observed[left]),
                "observed_right_error": float(observed[right]),
                "observed_difference": float(observed[left] - observed[right]),
                "bootstrap_mean_difference": float(difference.mean()),
                "difference_95ci_low": float(difference.quantile(0.025)),
                "difference_95ci_high": float(difference.quantile(0.975)),
                "probability_left_better": float((difference < 0).mean()),
                "bootstrap_replicates": len(difference),
            }
        )
        draw_rows.append(
            pd.DataFrame(
                {
                    "left_policy": left,
                    "right_policy": right,
                    "bootstrap_replicate": draws["bootstrap_replicate"],
                    "difference": difference,
                }
            )
        )

    edges = pd.concat(edge_frames, ignore_index=True)
    relations = pd.concat(relation_frames, ignore_index=True)
    normalization_summary = (
        relations.groupby(
            ["source_type", "relation", "destination_type"], as_index=False
        )
        .agg(
            fit_count=("fold_id", "size"),
            training_edges=("training_edges", "sum"),
            changed_edges=("changed_edges", "sum"),
            mean_changed_edge_share=("changed_edge_share", "mean"),
            mean_absolute_change=("mean_absolute_change", "mean"),
            p95_absolute_change=("p95_absolute_change", "mean"),
            max_absolute_change=("max_absolute_change", "max"),
        )
    )
    overall_normalization = pd.DataFrame(
        [
            {
                "fit_count": int(len(relations)),
                "training_edges": int(relations["training_edges"].sum()),
                "changed_edges": int(relations["changed_edges"].sum()),
                "changed_edge_share": float(
                    relations["changed_edges"].sum()
                    / relations["training_edges"].sum()
                ),
                "mean_relation_absolute_change": float(
                    relations["mean_absolute_change"].mean()
                ),
                "max_absolute_change": float(
                    relations["max_absolute_change"].max()
                ),
            }
        ]
    )
    predictions.to_csv(output / "gcn_predictions.csv", index=False)
    aggregated.to_csv(output / "gcn_seed_aggregated_predictions.csv", index=False)
    primary_errors.to_csv(output / "policy_primary_errors.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "performance_summary.csv", index=False)
    pd.concat(draw_rows, ignore_index=True).to_csv(
        output / "performance_bootstrap_draws.csv", index=False
    )
    edges.to_csv(output / "normalization_by_edge.csv", index=False)
    relations.to_csv(output / "normalization_by_fit_relation.csv", index=False)
    normalization_summary.to_csv(
        output / "normalization_relation_summary.csv", index=False
    )
    overall_normalization.to_csv(
        output / "normalization_overall_summary.csv", index=False
    )
    pd.DataFrame(pair_rows).to_csv(output / "policy_runtime.csv", index=False)
    pd.DataFrame(runtimes).to_csv(output / "paired_runtime.csv", index=False)
    (output / "README.md").write_text(
        "\n".join(
            [
                "# B4: GCN isolation replay and degree normalization",
                "",
                f"Label: **{ALLOWED_LABEL}**.",
                "",
                "This analysis executes the existing GCN implementation and the "
                "existing `legacy_leaky_global` policy without reconstructing an "
                "older model. Five folds from canonical CV5 partition 0 and neural "
                "seeds 7, 11, and 16 are held fixed across policies.",
                "",
                "`performance_summary.csv` compares the old all-row/global-fallback "
                "pipeline as-is with isolated training. The controlled comparison "
                "`audit_supported_leaky` versus `isolated_supported` keeps the "
                "supported evaluation set fixed and isolates the graph boundary.",
                "",
                "`normalization_overall_summary.csv` and related files isolate the "
                "degree-normalization pathway structurally: they compare each "
                "training edge's GCN normalization coefficient with and without "
                "held-out nodes in the graph. These coefficient changes are not "
                "model-performance estimates.",
                "",
                "The analysis is descriptive and is not promoted to a co-primary "
                "or secondary result.",
            ]
        )
        + "\n"
    )
    write_manifest(
        output,
        root=ROOT,
        command=[sys.executable, str(Path(__file__)), *sys.argv[1:]],
        input_files={
            "mlperf_long": ROOT / "data" / "mlperf_long.csv",
            "fold_assignments": FOLDS,
            "accelerator_families": ROOT / "config" / "accelerator_families.csv",
            "protocol": ROOT / "protocol.md",
            "evaluate_source": ROOT / "evaluation" / "evaluate.py",
            "graph_source": ROOT / "evaluation" / "graph.py",
            "model_source": ROOT / "train_graphsage.py",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
