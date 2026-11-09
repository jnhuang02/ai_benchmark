"""Heterogeneous graph construction with an explicit train/inference boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData

from evaluation.data import UNKNOWN
from evaluation.features import (
    CategoryVocabulary,
    NumericScaler,
    StrictTargetScaler,
    normalized_category,
)
from train_graphsage import HeteroGraphSAGE, graph_edge_types


GRAPH_CATEGORICAL_COLUMNS = [
    "organization",
    "system_type",
    "accelerator",
    "host_processor",
    "target_group",
    "precision",
]

RELATION_EDGE_TYPES = {
    "system": [
        ("system", "uses_accelerator", "accelerator"),
        ("accelerator", "used_by_system", "system"),
        ("system", "uses_processor", "processor"),
        ("processor", "used_by_system", "system"),
        ("system", "runs_workload", "workload"),
        ("workload", "runs_on_system", "system"),
        ("result", "of_system", "system"),
        ("system", "has_result", "result"),
    ],
    "accelerator": [
        ("system", "uses_accelerator", "accelerator"),
        ("accelerator", "used_by_system", "system"),
        ("result", "uses_accelerator", "accelerator"),
        ("accelerator", "has_result", "result"),
    ],
    "processor": [
        ("system", "uses_processor", "processor"),
        ("processor", "used_by_system", "system"),
        ("result", "uses_processor", "processor"),
        ("processor", "has_result", "result"),
    ],
    "workload": [
        ("system", "runs_workload", "workload"),
        ("workload", "runs_on_system", "system"),
        ("result", "measures", "workload"),
        ("workload", "has_result", "result"),
    ],
}


@dataclass(frozen=True)
class GraphFeatures:
    vocabulary: CategoryVocabulary
    numeric: NumericScaler

    @classmethod
    def fit(
        cls,
        train_rows: pd.DataFrame,
        numeric_columns: list[str] | tuple[str, ...] | None = None,
    ) -> "GraphFeatures":
        return cls(
            vocabulary=CategoryVocabulary.fit(train_rows),
            numeric=(
                NumericScaler.fit(train_rows)
                if numeric_columns is None
                else NumericScaler.fit(train_rows, numeric_columns)
            ),
        )


def _undirected(
    source_values: pd.Series, target_values: pd.Series
) -> tuple[torch.Tensor, torch.Tensor]:
    source_nodes = np.array(sorted(set(source_values.tolist())))
    target_nodes = np.array(sorted(set(target_values.tolist())))
    source_lookup = {value: index for index, value in enumerate(source_nodes)}
    target_lookup = {value: index for index, value in enumerate(target_nodes)}
    pairs = pd.DataFrame(
        {
            "source": source_values.map(source_lookup),
            "target": target_values.map(target_lookup),
        }
    ).drop_duplicates()
    edge = torch.as_tensor(
        np.stack([pairs["source"].to_numpy(), pairs["target"].to_numpy()]),
        dtype=torch.long,
    )
    return edge, edge.flip(0)


def make_graph(
    rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    features: GraphFeatures,
    target_scaler: StrictTargetScaler,
    seed: int,
    unknown_mask_probability: float = 0.1,
) -> HeteroData:
    """Build a graph; random unknown masking applies only to train systems."""
    rows = rows.reset_index(drop=True).copy()
    train_systems = set(train_rows["system_id"])
    train_accelerators = set(train_rows["accelerator_id"])
    train_processors = set(train_rows["processor_id"])
    train_workloads = set(train_rows["target_group"])

    system_rows = rows.drop_duplicates("system_id").sort_values("system_id").reset_index(
        drop=True
    )
    accelerator_rows = (
        rows.drop_duplicates("accelerator_id")
        .sort_values("accelerator_id")
        .reset_index(drop=True)
    )
    processor_rows = (
        rows.drop_duplicates("processor_id")
        .sort_values("processor_id")
        .reset_index(drop=True)
    )
    workload_rows = (
        rows.drop_duplicates("target_group")
        .sort_values("target_group")
        .reset_index(drop=True)
    )

    system_org = normalized_category(system_rows["organization"]).copy()
    system_type = normalized_category(system_rows["system_type"]).copy()
    rng = np.random.default_rng(seed + 1)
    system_draw = rng.random((len(system_rows), 2)) < unknown_mask_probability
    system_training = system_rows["system_id"].isin(train_systems).to_numpy()
    system_org.loc[system_draw[:, 0] & system_training] = UNKNOWN
    system_type.loc[system_draw[:, 1] & system_training] = UNKNOWN

    accelerator = normalized_category(accelerator_rows["accelerator"]).copy()
    rng = np.random.default_rng(seed + 2)
    draw = rng.random(len(accelerator_rows)) < unknown_mask_probability
    training = accelerator_rows["accelerator_id"].isin(train_accelerators).to_numpy()
    accelerator.loc[draw & training] = UNKNOWN

    processor = normalized_category(processor_rows["host_processor"]).copy()
    rng = np.random.default_rng(seed + 3)
    draw = rng.random(len(processor_rows)) < unknown_mask_probability
    training = processor_rows["processor_id"].isin(train_processors).to_numpy()
    processor.loc[draw & training] = UNKNOWN

    workload = normalized_category(workload_rows["target_group"]).copy()
    rng = np.random.default_rng(seed + 4)
    draw = rng.random(len(workload_rows)) < unknown_mask_probability
    training = workload_rows["target_group"].isin(train_workloads).to_numpy()
    workload.loc[draw & training] = UNKNOWN

    precision = normalized_category(rows["precision"]).copy()
    rng = np.random.default_rng(seed + 5)
    draw = rng.random(len(rows)) < unknown_mask_probability
    training = rows["system_id"].isin(train_systems).to_numpy()
    precision.loc[draw & training] = UNKNOWN

    system_features = np.hstack(
        [
            features.vocabulary.one_hot_array(
                pd.DataFrame({"organization": system_org}), "organization"
            ),
            features.vocabulary.one_hot_array(
                pd.DataFrame({"system_type": system_type}), "system_type"
            ),
        ]
    )
    result_features = np.hstack(
        [
            features.numeric.transform(rows),
            features.vocabulary.one_hot_array(
                pd.DataFrame({"precision": precision}), "precision"
            ),
        ]
    )

    data = HeteroData()
    data["system"].x = torch.as_tensor(system_features, dtype=torch.float32)
    data["accelerator"].x = torch.as_tensor(
        features.vocabulary.one_hot_array(
            pd.DataFrame({"accelerator": accelerator}), "accelerator"
        ),
        dtype=torch.float32,
    )
    data["processor"].x = torch.as_tensor(
        features.vocabulary.one_hot_array(
            pd.DataFrame({"host_processor": processor}), "host_processor"
        ),
        dtype=torch.float32,
    )
    data["workload"].x = torch.as_tensor(
        features.vocabulary.one_hot_array(
            pd.DataFrame({"target_group": workload}), "target_group"
        ),
        dtype=torch.float32,
    )
    data["result"].x = torch.as_tensor(result_features, dtype=torch.float32)

    system_accel, accel_system = _undirected(rows["system_id"], rows["accelerator_id"])
    system_processor, processor_system = _undirected(
        rows["system_id"], rows["processor_id"]
    )
    system_workload, workload_system = _undirected(
        rows["system_id"], rows["target_group"]
    )
    result_ids = pd.Series([f"result:{index}" for index in range(len(rows))])
    result_system, system_result = _undirected(result_ids, rows["system_id"])
    result_accel, accel_result = _undirected(result_ids, rows["accelerator_id"])
    result_processor, processor_result = _undirected(
        result_ids, rows["processor_id"]
    )
    result_workload, workload_result = _undirected(
        result_ids, rows["target_group"]
    )

    data["system", "uses_accelerator", "accelerator"].edge_index = system_accel
    data["accelerator", "used_by_system", "system"].edge_index = accel_system
    data["system", "uses_processor", "processor"].edge_index = system_processor
    data["processor", "used_by_system", "system"].edge_index = processor_system
    data["system", "runs_workload", "workload"].edge_index = system_workload
    data["workload", "runs_on_system", "system"].edge_index = workload_system
    data["result", "of_system", "system"].edge_index = result_system
    data["system", "has_result", "result"].edge_index = system_result
    data["result", "uses_accelerator", "accelerator"].edge_index = result_accel
    data["accelerator", "has_result", "result"].edge_index = accel_result
    data["result", "uses_processor", "processor"].edge_index = result_processor
    data["processor", "has_result", "result"].edge_index = processor_result
    data["result", "measures", "workload"].edge_index = result_workload
    data["workload", "has_result", "result"].edge_index = workload_result

    normalized_targets = target_scaler.transform(rows)
    data["result"].y = torch.as_tensor(normalized_targets, dtype=torch.float32)
    data["result"].row_uid = rows["row_uid"].tolist()
    data["result"].system_id = rows["system_id"].tolist()
    data["system"].system_id = system_rows["system_id"].tolist()
    data["accelerator"].accelerator_id = accelerator_rows["accelerator_id"].tolist()
    data["processor"].processor_id = processor_rows["processor_id"].tolist()
    data["workload"].target_group = workload_rows["target_group"].tolist()
    return data


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class TrainedGNN:
    model: HeteroGraphSAGE
    best_validation_loss: float
    epochs_run: int
    weight_sha256: str
    best_epoch: int
    final_train_loss: float
    final_validation_loss: float


def fit_gnn(
    graph: HeteroData,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    model_name: str,
    hidden: int,
    layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
    edge_types: list[tuple[str, str, str]] | None = None,
) -> TrainedGNN:
    set_seed(seed)
    model = HeteroGraphSAGE(
        hidden,
        layers,
        model_name,
        dropout,
        edge_types=edge_types,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_index = torch.as_tensor(np.flatnonzero(train_mask), dtype=torch.long)
    validation_index = torch.as_tensor(
        np.flatnonzero(validation_mask), dtype=torch.long
    )
    target = graph["result"].y
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    epochs_run = 0
    final_train_loss = float("nan")
    final_validation_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(graph)
        loss = nn.functional.huber_loss(
            prediction[train_index], target[train_index], delta=1.0
        )
        final_train_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        epochs_run = epoch
        if len(validation_index):
            model.eval()
            with torch.no_grad():
                validation_loss = nn.functional.huber_loss(
                    model(graph)[validation_index],
                    target[validation_index],
                    delta=1.0,
                ).item()
                final_validation_loss = float(validation_loss)
            if validation_loss < best_loss - 1e-12:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainedGNN(
        model,
        best_loss,
        epochs_run,
        state_dict_sha256(model),
        best_epoch,
        final_train_loss,
        final_validation_loss,
    )


def predict_gnn(model: HeteroGraphSAGE, graph: HeteroData) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(graph).cpu().numpy()


def subset_graph_edges(
    graph: HeteroData, edge_types: list[tuple[str, str, str]]
) -> HeteroData:
    selected = set(edge_types)
    adjusted = graph.clone()
    for edge_type in list(adjusted.edge_types):
        if edge_type not in selected:
            del adjusted[edge_type]
    return adjusted


def degree_preserving_shuffled_graph(
    graph: HeteroData, *, seed: int
) -> HeteroData:
    """Permute destinations within each relation, retaining both degree sets."""
    adjusted = graph.clone()
    rng = np.random.default_rng(seed)
    for edge_type in adjusted.edge_types:
        edge_index = adjusted[edge_type].edge_index
        destinations = edge_index[1].numpy().copy()
        rng.shuffle(destinations)
        adjusted[edge_type].edge_index = torch.stack(
            [
                edge_index[0],
                torch.as_tensor(destinations, dtype=torch.long),
            ]
        )
    return adjusted


def graph_isolation_audit(
    graph: HeteroData, forbidden_systems: set[str]
) -> None:
    present_result_systems = set(graph["result"].system_id)
    present_system_nodes = set(graph["system"].system_id)
    leaked = forbidden_systems & (present_result_systems | present_system_nodes)
    if leaked:
        raise AssertionError(f"Held-out systems found in training graph: {leaked}")
    for edge_type in graph.edge_types:
        if edge_type not in graph.edge_index_dict:
            raise AssertionError(f"Missing relation {edge_type}")
