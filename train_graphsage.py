#!/usr/bin/env python3
"""Train a heterogeneous GraphSAGE performance model and grouped baselines."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


NUMERIC_COLUMNS = [
    "nodes",
    "accelerators_per_node",
    "processors_per_node",
    "host_processor_cores",
    "total_accelerators",
]
RESULT_CATEGORICAL_COLUMNS = ["precision"]
BASELINE_CATEGORICAL_COLUMNS = [
    "organization",
    "system_type",
    "accelerator",
    "host_processor",
    "target_group",
    "precision",
]
BASELINE_NUMERIC_COLUMNS = NUMERIC_COLUMNS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(
    values: pd.Series, categories: list[str], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    lookup = {value: index for index, value in enumerate(categories)}
    indices = values.astype(str).map(lookup).fillna(0).to_numpy(dtype=np.int64)
    matrix = np.zeros((len(values), len(categories)), dtype=np.float32)
    matrix[np.arange(len(values)), indices] = 1.0
    return torch.as_tensor(matrix, dtype=dtype)


def fit_numeric(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    values = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    transformed = np.log1p(values.clip(lower=0))
    return transformed.median(), transformed.std().replace(0, 1).fillna(1)


def numeric_features(
    frame: pd.DataFrame, center: pd.Series, scale: pd.Series
) -> torch.Tensor:
    values = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    transformed = np.log1p(values.clip(lower=0))
    scaled = ((transformed - center) / scale).fillna(0.0)
    return torch.as_tensor(scaled.to_numpy(dtype=np.float32, copy=True))


@dataclass
class TargetScaler:
    center: pd.Series
    scale: pd.Series
    global_center: float
    global_scale: float

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "TargetScaler":
        grouped = frame.groupby("target_group")["log_performance"]
        center = grouped.median()
        scale = grouped.std().replace(0, np.nan)
        valid = grouped.size().ge(3) & scale.notna()
        center = center.where(valid)
        scale = scale.where(valid)
        global_center = float(frame["log_performance"].median())
        global_scale = float(frame["log_performance"].std())
        if not np.isfinite(global_scale) or global_scale <= 1e-8:
            global_scale = 1.0
        center = center.fillna(global_center)
        scale = scale.fillna(global_scale)
        return cls(center, scale, global_center, global_scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        center = frame["target_group"].map(self.center).fillna(self.global_center)
        scale = frame["target_group"].map(self.scale).fillna(self.global_scale)
        return (
            (frame["log_performance"].to_numpy() - center.to_numpy())
            / scale.to_numpy()
        )

    def normalize(
        self, frame: pd.DataFrame, signed_log_performance: np.ndarray
    ) -> np.ndarray:
        center = frame["target_group"].map(self.center).fillna(self.global_center)
        scale = frame["target_group"].map(self.scale).fillna(self.global_scale)
        return (
            signed_log_performance - center.to_numpy()
        ) / scale.to_numpy()

    def inverse(self, frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
        center = frame["target_group"].map(self.center).fillna(self.global_center)
        scale = frame["target_group"].map(self.scale).fillna(self.global_scale)
        signed_log = prediction * scale.to_numpy() + center.to_numpy()
        signed_log = np.clip(signed_log, -50.0, 50.0)
        return np.exp(frame["direction"].to_numpy(dtype=float) * signed_log)


def baseline_matrix(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="unknown"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    numeric = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    numeric_columns = [
        column
        for column in BASELINE_NUMERIC_COLUMNS
        if train[column].notna().any()
    ]
    transformers = [("categorical", categorical, BASELINE_CATEGORICAL_COLUMNS)]
    if numeric_columns:
        transformers.append(("numeric", numeric, numeric_columns))
    transform = ColumnTransformer(transformers)
    train_x = transform.fit_transform(train)
    eval_x = transform.transform(evaluation)
    return train_x, eval_x, transform


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        hidden: int,
        layers: int,
        model_name: str,
        dropout: float = 0.2,
        edge_types: list[tuple[str, str, str]] | None = None,
    ) -> None:
        super().__init__()
        selected_edge_types = (
            graph_edge_types() if edge_types is None else list(edge_types)
        )
        self.convs = nn.ModuleList()
        for _ in range(layers):
            convolutions = {
                edge_type: bipartite_conv(edge_type, hidden, model_name)
                for edge_type in selected_edge_types
            }
            self.convs.append(HeteroConv(convolutions, aggr="sum"))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data: HeteroData) -> torch.Tensor:
        values = data.x_dict
        for conv in self.convs:
            updated = conv(values, data.edge_index_dict)
            values = {
                node_type: self.dropout(torch.relu(embedding))
                for node_type, embedding in updated.items()
            }
        return self.head(values["result"]).squeeze(-1)


class BipartiteGCNConv(nn.Module):
    """Relation-specific GCN message passing for a heterogeneous edge type."""

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.source_linear = nn.LazyLinear(output_size, bias=False)
        self.destination_linear = nn.LazyLinear(output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(
        self, source_and_destination: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source_x, destination_x = source_and_destination
        source_x = self.source_linear(source_x)
        destination_x = self.destination_linear(destination_x)
        row, column = edge_index
        source_degree = torch.bincount(
            row, minlength=source_x.size(0)
        ).clamp_min(1).to(source_x.dtype)
        destination_degree = torch.bincount(
            column, minlength=destination_x.size(0)
        ).clamp_min(1).to(destination_x.dtype)
        normalization = source_degree[row].pow(-0.5) * destination_degree[
            column
        ].pow(-0.5)
        messages = source_x[row] * normalization.unsqueeze(-1)
        output = torch.zeros_like(destination_x)
        output.index_add_(0, column, messages)
        return output + destination_x + self.bias


def bipartite_conv(
    edge_type: tuple[str, str, str], hidden: int, model_name: str
) -> nn.Module:
    if model_name == "graphsage":
        return SAGEConv((-1, -1), hidden)
    if model_name == "gcn":
        return BipartiteGCNConv(hidden)
    if model_name == "gat":
        return GATConv(
            (-1, -1),
            hidden // 4,
            heads=4,
            concat=True,
            add_self_loops=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


def graph_edge_types() -> list[tuple[str, str, str]]:
    return [
        ("system", "uses_accelerator", "accelerator"),
        ("accelerator", "used_by_system", "system"),
        ("system", "uses_processor", "processor"),
        ("processor", "used_by_system", "system"),
        ("system", "runs_workload", "workload"),
        ("workload", "runs_on_system", "system"),
        ("result", "of_system", "system"),
        ("system", "has_result", "result"),
        ("result", "uses_accelerator", "accelerator"),
        ("accelerator", "has_result", "result"),
        ("result", "uses_processor", "processor"),
        ("processor", "has_result", "result"),
        ("result", "measures", "workload"),
        ("workload", "has_result", "result"),
    ]


def unique_values(frame: pd.DataFrame, column: str) -> list[str]:
    return sorted(frame[column].fillna("unknown").astype(str).unique())


def map_indices(nodes: list[str], values: pd.Series) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(nodes)}
    return values.astype(str).map(lookup).to_numpy(dtype=np.int64)


def undirected_edges(
    source_nodes: list[str],
    target_nodes: list[str],
    source_values: pd.Series,
    target_values: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = pd.DataFrame(
        {
            "source": map_indices(source_nodes, source_values),
            "target": map_indices(target_nodes, target_values),
        }
    ).drop_duplicates()
    source = torch.as_tensor(
        pairs["source"].to_numpy(copy=True), dtype=torch.long
    )
    target = torch.as_tensor(
        pairs["target"].to_numpy(copy=True), dtype=torch.long
    )
    edge = torch.stack([source, target])
    return edge, edge.flip(0)


def make_graph(
    all_rows: pd.DataFrame, train_rows: pd.DataFrame
) -> HeteroData:
    categories = {
        column: unique_values(train_rows, column)
        for column in set(BASELINE_CATEGORICAL_COLUMNS + RESULT_CATEGORICAL_COLUMNS)
    }
    for column in categories:
        categories[column] = ["<unknown>"] + [
            item for item in categories[column] if item != "<unknown>"
        ]

    systems = unique_values(all_rows, "system_id")
    accelerators = unique_values(all_rows, "accelerator_id")
    processors = unique_values(all_rows, "processor_id")
    workloads = unique_values(all_rows, "target_group")

    data = HeteroData()
    system_rows = all_rows.drop_duplicates("system_id")
    accelerator_rows = all_rows.drop_duplicates("accelerator_id")
    processor_rows = all_rows.drop_duplicates("processor_id")
    workload_rows = all_rows.drop_duplicates("target_group")

    data["system"].x = torch.cat(
        [
            one_hot(system_rows["organization"], categories["organization"]),
            one_hot(system_rows["system_type"], categories["system_type"]),
        ],
        dim=1,
    )
    data["accelerator"].x = one_hot(
        accelerator_rows["accelerator"], categories["accelerator"]
    )
    data["processor"].x = one_hot(
        processor_rows["host_processor"], categories["host_processor"]
    )
    data["workload"].x = one_hot(
        workload_rows["target_group"], categories["target_group"]
    )

    numeric_center, numeric_scale = fit_numeric(train_rows)
    data["result"].x = torch.cat(
        [
            numeric_features(all_rows, numeric_center, numeric_scale),
            one_hot(all_rows["precision"], categories["precision"]),
        ],
        dim=1,
    )

    system_to_accel, accel_to_system = undirected_edges(
        systems, accelerators, all_rows["system_id"], all_rows["accelerator_id"]
    )
    system_to_processor, processor_to_system = undirected_edges(
        systems, processors, all_rows["system_id"], all_rows["processor_id"]
    )
    system_to_workload, workload_to_system = undirected_edges(
        systems, workloads, all_rows["system_id"], all_rows["target_group"]
    )
    result_to_system, system_to_result = undirected_edges(
        [str(index) for index in range(len(all_rows))],
        systems,
        pd.Series(range(len(all_rows))).astype(str),
        all_rows["system_id"],
    )
    result_to_accel, accel_to_result = undirected_edges(
        [str(index) for index in range(len(all_rows))],
        accelerators,
        pd.Series(range(len(all_rows))).astype(str),
        all_rows["accelerator_id"],
    )
    result_to_processor, processor_to_result = undirected_edges(
        [str(index) for index in range(len(all_rows))],
        processors,
        pd.Series(range(len(all_rows))).astype(str),
        all_rows["processor_id"],
    )
    result_to_workload, workload_to_result = undirected_edges(
        [str(index) for index in range(len(all_rows))],
        workloads,
        pd.Series(range(len(all_rows))).astype(str),
        all_rows["target_group"],
    )

    data["system", "uses_accelerator", "accelerator"].edge_index = system_to_accel
    data["accelerator", "used_by_system", "system"].edge_index = accel_to_system
    data["system", "uses_processor", "processor"].edge_index = system_to_processor
    data["processor", "used_by_system", "system"].edge_index = processor_to_system
    data["system", "runs_workload", "workload"].edge_index = system_to_workload
    data["workload", "runs_on_system", "system"].edge_index = workload_to_system
    data["result", "of_system", "system"].edge_index = result_to_system
    data["system", "has_result", "result"].edge_index = system_to_result
    data["result", "uses_accelerator", "accelerator"].edge_index = result_to_accel
    data["accelerator", "has_result", "result"].edge_index = accel_to_result
    data["result", "uses_processor", "processor"].edge_index = result_to_processor
    data["processor", "has_result", "result"].edge_index = processor_to_result
    data["result", "measures", "workload"].edge_index = result_to_workload
    data["workload", "has_result", "result"].edge_index = workload_to_result
    return data


def group_spearman(evaluation: pd.DataFrame) -> float:
    scores = []
    for _, group in evaluation.groupby("target_group"):
        if (
            len(group) < 3
            or group["result"].nunique() < 2
            or group["prediction"].nunique() < 2
        ):
            continue
        score = group["result"].rank().corr(group["prediction"].rank())
        if np.isfinite(score):
            scores.append(float(score))
    return float(np.median(scores)) if scores else float("nan")


def metrics(evaluation: pd.DataFrame, log_predictions: np.ndarray) -> dict[str, float]:
    truth = evaluation["result"].to_numpy(dtype=float)
    prediction = evaluation["prediction"].to_numpy(dtype=float)
    mape = np.abs(prediction - truth) / truth
    group_mape = []
    for _, group in evaluation.groupby("target_group"):
        group_mape.append(
            float(
                np.median(
                    np.abs(group["prediction"] - group["result"]) / group["result"]
                )
            )
        )
    return {
        "pooled_mape": float(np.mean(mape)),
        "median_group_mape": float(np.median(group_mape)),
        "rmse_log_score": float(np.sqrt(np.mean((log_predictions - evaluation["y"]) ** 2))),
        "median_group_spearman": group_spearman(evaluation),
    }


def bootstrap_mape_ci(evaluation: pd.DataFrame, seed: int) -> tuple[float, float]:
    truth = evaluation["result"].to_numpy(dtype=float)
    prediction = evaluation["prediction"].to_numpy(dtype=float)
    errors = np.abs(prediction - truth) / truth
    rng = np.random.default_rng(seed)
    sampled = rng.choice(errors, size=(2000, len(errors)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def train_gnn(
    graph: HeteroData,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    model = HeteroGraphSAGE(args.hidden, args.layers, args.model, args.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_index = torch.as_tensor(np.flatnonzero(train_mask), dtype=torch.long)
    validation_index = torch.as_tensor(
        np.flatnonzero(validation_mask), dtype=torch.long
    )
    test_index = torch.as_tensor(np.flatnonzero(test_mask), dtype=torch.long)
    target = graph["result"].y
    best_state = None
    best_loss = float("inf")
    patience = 0

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(graph)
        loss = nn.functional.huber_loss(prediction[train_index], target[train_index])
        loss.backward()
        optimizer.step()

        if len(validation_index):
            model.eval()
            with torch.no_grad():
                validation_prediction = model(graph)
                validation_loss = nn.functional.huber_loss(
                    validation_prediction[validation_index],
                    target[validation_index],
                ).item()
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(graph)[test_index].numpy()


def cross_validate(
    frame: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = np.array(sorted(frame["system_id"].unique()))
    if args.loso:
        outer_folds = [([system], [s for s in systems if s != system]) for system in systems]
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = rng.permutation(systems)
        chunks = np.array_split(shuffled, min(args.folds, len(systems)))
        outer_folds = [(list(chunk), [s for s in systems if s not in set(chunk)]) for chunk in chunks]

    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []
    for fold, (test_systems, train_systems) in enumerate(outer_folds):
        test_mask = frame["system_id"].isin(test_systems).to_numpy()
        outer_train_mask = frame["system_id"].isin(train_systems).to_numpy()
        if not test_mask.any() or not outer_train_mask.any():
            continue

        rng = np.random.default_rng(args.seed + fold)
        train_system_values = np.array(
            sorted(frame.loc[outer_train_mask, "system_id"].unique())
        )
        validation_count = max(1, round(0.12 * len(train_system_values)))
        validation_systems = set(rng.choice(train_system_values, validation_count, replace=False))
        validation_mask = frame["system_id"].isin(validation_systems).to_numpy()
        train_mask = outer_train_mask & ~validation_mask
        if not train_mask.any():
            train_mask = outer_train_mask
            validation_mask[:] = False

        train = frame.loc[train_mask].copy()
        scaler = TargetScaler.fit(train)
        y_norm = scaler.transform(frame)
        output_frames: dict[str, pd.DataFrame] = {}

        grouped_center = train.groupby("target_group")["log_performance"].median()
        grouped_prediction = frame["target_group"].map(grouped_center).fillna(
            scaler.global_center
        )
        evaluation = frame.loc[test_mask].copy()
        grouped_log_prediction = grouped_prediction.loc[test_mask].to_numpy()
        output_frames["group_median"] = evaluation.assign(
            prediction=scaler.inverse(
                evaluation,
                scaler.normalize(evaluation, grouped_log_prediction),
            ),
            y=y_norm[test_mask],
        )

        train_x, test_x, _ = baseline_matrix(train, evaluation)
        for name, baseline in [
            ("ridge", Ridge(alpha=10.0)),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features=0.7,
                    random_state=args.seed + fold,
                    n_jobs=-1,
                ),
            ),
        ]:
            baseline.fit(train_x, y_norm[train_mask])
            log_prediction = baseline.predict(test_x)
            output_frames[name] = evaluation.assign(
                prediction=scaler.inverse(evaluation, log_prediction),
                y=y_norm[test_mask],
                prediction_log=log_prediction,
            )

        graph = make_graph(frame, train)
        graph["result"].y = torch.as_tensor(y_norm, dtype=torch.float32)
        set_seed(args.seed + fold)
        log_prediction = train_gnn(
            graph, train_mask, validation_mask, test_mask, args
        )
        output_frames[args.model] = evaluation.assign(
            prediction=scaler.inverse(evaluation, log_prediction),
            y=y_norm[test_mask],
            prediction_log=log_prediction,
        )

        for name, prediction_frame in output_frames.items():
            if "prediction_log" not in prediction_frame:
                signed_log = (
                    prediction_frame["direction"].to_numpy(dtype=float)
                    * np.log(prediction_frame["prediction"].to_numpy(dtype=float))
                )
                prediction_frame["prediction_log"] = scaler.normalize(
                    prediction_frame, signed_log
                )
            prediction_frame = prediction_frame.assign(
                fold=fold, method=name, heldout_systems=";".join(test_systems)
            )
            all_predictions.append(prediction_frame)
            fold_metrics.append(
                {
                    "fold": fold,
                    "method": name,
                    **metrics(
                        prediction_frame,
                        prediction_frame["prediction_log"].to_numpy(),
                    ),
                }
            )
        print(
            f"fold {fold + 1:>3}/{len(outer_folds)} | "
            f"held out {len(evaluation):>3} rows / {len(test_systems):>2} systems"
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_table = pd.DataFrame(fold_metrics)
    summary_rows = []
    for name, frame_for_model in predictions.groupby("method"):
        low, high = bootstrap_mape_ci(frame_for_model, args.seed)
        row = {
            "method": name,
            **metrics(
                frame_for_model,
                frame_for_model["prediction_log"].to_numpy(dtype=float),
            ),
            "pooled_mape_95ci_low": low,
            "pooled_mape_95ci_high": high,
            "fold_median_mape": float(
                fold_table.loc[
                    fold_table["method"].eq(name), "pooled_mape"
                ].median()
            ),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("pooled_mape")
    return summary, predictions


def grouped_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, target_group), group in predictions.groupby(
        ["method", "target_group"]
    ):
        errors = np.abs(group["prediction"] - group["result"]) / group["result"]
        rank_score = (
            group["result"].rank().corr(group["prediction"].rank())
            if group["result"].nunique() >= 2 and group["prediction"].nunique() >= 2
            else np.nan
        )
        rows.append(
            {
                "method": method,
                "target_group": target_group,
                "rows": len(group),
                "mape": float(np.mean(errors)),
                "median_mape": float(np.median(errors)),
                "spearman": rank_score,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "mlperf_long.csv",
    )
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "datacenter", "edge"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Leave one whole system out per fold. Use for final evaluation.",
    )
    parser.add_argument(
        "--model",
        default="graphsage",
        choices=["graphsage", "gcn", "gat"],
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} does not exist; run prepare_data.py first"
        )
    frame = pd.read_csv(args.data)
    if args.suite != "all":
        frame = frame.loc[frame["suite"].eq(args.suite)].copy()
    frame = frame.dropna(subset=["result", "log_performance", "system_id"]).reset_index(
        drop=True
    )
    if frame["system_id"].nunique() < 3:
        raise ValueError("At least three independent systems are required")

    summary, predictions = cross_validate(frame, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.suite}_{args.model}_summary.csv"
    prediction_path = args.out_dir / f"{args.suite}_{args.model}_predictions.csv"
    group_path = args.out_dir / f"{args.suite}_{args.model}_groups.csv"
    summary.to_csv(summary_path, index=False)
    prediction_columns = [
        "fold",
        "method",
        "suite",
        "version",
        "division",
        "public_id",
        "organization",
        "system_id",
        "benchmark",
        "model",
        "task_code",
        "scenario",
        "units",
        "result",
        "prediction",
        "y",
        "prediction_log",
    ]
    predictions[prediction_columns].to_csv(prediction_path, index=False)
    grouped_report(predictions).to_csv(group_path, index=False)

    print("\nGrouped system-level validation (development evidence, not an official MLPerf result):")
    printable = summary.copy()
    for column in printable.select_dtypes(include="number").columns:
        printable[column] = printable[column].map(
            lambda value: f"{value:.3f}" if math.isfinite(value) else "nan"
        )
    print(printable.to_string(index=False))
    print(f"\nSummary:      {summary_path}")
    print(f"Predictions:  {prediction_path}")
    print(f"Groups:       {group_path}")


if __name__ == "__main__":
    main()
