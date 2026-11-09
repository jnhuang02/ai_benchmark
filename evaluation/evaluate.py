"""Fold-level model fitting with auditable leakage boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch_geometric.data import HeteroData

from evaluation.features import (
    CategoryVocabulary,
    LegacyGlobalTargetScaler,
    StrictTargetScaler,
    TabularFeatureEncoder,
    mask_identity_values,
)
from evaluation.folds import systems_for_role
from evaluation.graph import (
    GraphFeatures,
    TrainedGNN,
    fit_gnn,
    graph_isolation_audit,
    make_graph,
    predict_gnn,
)


POLICIES = {
    "legacy_leaky_global": {
        "description": "Old all-row training graph with global sparse fallback",
        "isolated": False,
        "global_fallback": True,
    },
    "audit_supported_leaky": {
        "description": "All-row training graph with unsupported groups excluded",
        "isolated": False,
        "global_fallback": False,
    },
    "isolated_supported": {
        "description": "Train-only graph and unsupported sparse groups excluded",
        "isolated": True,
        "global_fallback": False,
    },
}


@dataclass(frozen=True)
class GNNConfig:
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 150
    patience: int = 20
    unknown_mask_probability: float = 0.1


@dataclass
class FoldEvaluation:
    predictions: pd.DataFrame
    unsupported: pd.DataFrame
    train_systems: tuple[str, ...]
    validation_systems: tuple[str, ...]
    test_systems: tuple[str, ...]
    scaler: StrictTargetScaler | LegacyGlobalTargetScaler
    vocabulary: CategoryVocabulary
    baseline_train_row_uids: tuple[str, ...] = ()
    baseline_test_row_uids: tuple[str, ...] = ()
    training_graph: HeteroData | None = field(default=None, repr=False)
    inference_graph: HeteroData | None = field(default=None, repr=False)
    weight_sha256: str | None = None
    trained_weights: dict = field(default_factory=dict, repr=False)


def fold_role_sets(
    assignments: pd.DataFrame, protocol: str, fold_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return tuple(
        systems_for_role(assignments, protocol, fold_id, role)
        for role in ("train", "validation", "test")
    )


def _frame_for_systems(frame: pd.DataFrame, systems: set[str]) -> pd.DataFrame:
    return frame.loc[frame["system_id"].isin(systems)].copy().reset_index(drop=True)


def _prediction_frame(
    original: pd.DataFrame,
    normalized_prediction: np.ndarray,
    scaler: StrictTargetScaler | LegacyGlobalTargetScaler,
    strict_supported: set[str],
    *,
    method: str,
    policy: str,
    seed: int,
    protocol: str,
    fold_id: str,
) -> pd.DataFrame:
    prediction = scaler.inverse(original, normalized_prediction)
    return original.assign(
        method=method,
        evaluation_policy=policy,
        seed=seed,
        protocol=protocol,
        fold_id=fold_id,
        prediction=prediction,
        prediction_log=normalized_prediction,
        y=scaler.transform(original),
        is_supported=original["target_group"].isin(strict_supported),
    )


def _group_median_prediction(
    test: pd.DataFrame,
    train_labeled: pd.DataFrame,
    scaler: StrictTargetScaler | LegacyGlobalTargetScaler,
) -> np.ndarray:
    centers = train_labeled.groupby("target_group")["log_performance"].median()
    test_center = test["target_group"].map(centers)
    scale = test["target_group"].map(scaler.scale)
    global_center = getattr(
        scaler, "global_center", float(train_labeled["log_performance"].median())
    )
    global_scale = getattr(
        scaler, "global_scale", float(train_labeled["log_performance"].std())
    )
    center = test_center.fillna(global_center).to_numpy(dtype=float)
    scale = scale.fillna(global_scale).to_numpy(dtype=float)
    return (center - test["target_group"].map(scaler.center).fillna(global_center).to_numpy(dtype=float)) / scale


def _prepare_tabular(
    train_labeled: pd.DataFrame,
    test: pd.DataFrame,
    vocabulary: CategoryVocabulary,
    seed: int,
    unknown_mask_probability: float,
) -> tuple[TabularFeatureEncoder, np.ndarray, np.ndarray]:
    masked_train = mask_identity_values(
        train_labeled,
        seed=seed + 70_000,
        probability=unknown_mask_probability,
    )
    encoder = TabularFeatureEncoder.fit(masked_train, vocabulary)
    return encoder, encoder.transform(masked_train), encoder.transform(test)


def evaluate_fold_model(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    fold_id: str,
    model: str,
    policy: str,
    seed: int,
    config: GNNConfig,
    include_diagnostics: bool = False,
) -> FoldEvaluation:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    policy_config = POLICIES[policy]
    train_systems, validation_systems, test_systems = fold_role_sets(
        assignments, protocol, fold_id
    )
    train = _frame_for_systems(frame, set(train_systems))
    validation = _frame_for_systems(frame, set(validation_systems))
    test_all = _frame_for_systems(frame, set(test_systems))
    if train.empty or validation.empty or test_all.empty:
        raise ValueError(f"{protocol}/{fold_id} has an empty role")

    if policy_config["global_fallback"]:
        scaler: StrictTargetScaler | LegacyGlobalTargetScaler = (
            LegacyGlobalTargetScaler.fit(train, minimum_rows=3)
        )
    else:
        scaler = StrictTargetScaler.fit(train, minimum_rows=3)
    strict_supported = set(scaler.supported_groups)

    train_labeled = train.loc[train["target_group"].isin(strict_supported)].copy()
    validation_labeled = validation.loc[
        validation["target_group"].isin(strict_supported)
    ].copy()
    if policy_config["global_fallback"]:
        train_labeled = train.copy()
        validation_labeled = validation.copy()

    test_eligible = (
        test_all.copy()
        if policy_config["global_fallback"]
        else test_all.loc[test_all["target_group"].isin(strict_supported)].copy()
    )
    unsupported = test_all.loc[
        ~test_all["target_group"].isin(strict_supported)
    ].copy()
    unsupported = unsupported.assign(
        method=f"{model}:{policy}",
        reason="fewer than 3 training rows or undefined training scale",
    )
    if test_eligible.empty:
        return FoldEvaluation(
            pd.DataFrame(),
            unsupported,
            train_systems,
            validation_systems,
            test_systems,
            scaler,
            CategoryVocabulary.fit(train),
            tuple(train_labeled["row_uid"]),
            tuple(test_eligible["row_uid"]),
        )

    vocabulary = CategoryVocabulary.fit(train)
    outputs = []
    if include_diagnostics or model == "group_median":
        normalized = _group_median_prediction(test_eligible, train_labeled, scaler)
        outputs.append(
            _prediction_frame(
                test_eligible,
                normalized,
                scaler,
                strict_supported,
                method="group_median",
                policy=policy,
                seed=seed,
                protocol=protocol,
                fold_id=fold_id,
            )
        )

    if include_diagnostics or model in {"ridge", "random_forest"}:
        encoder, train_x, test_x = _prepare_tabular(
            train_labeled,
            test_eligible,
            vocabulary,
            seed,
            config.unknown_mask_probability,
        )
        del encoder
        train_y = scaler.transform(train_labeled)
        if model in {"ridge", "random_forest"}:
            model_map = {
                "ridge": Ridge(alpha=10.0),
                "random_forest": RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features=0.7,
                    random_state=seed,
                    n_jobs=-1,
                ),
            }
            estimator = model_map[model]
            estimator.fit(train_x, train_y)
            normalized = estimator.predict(test_x)
            outputs.append(
                _prediction_frame(
                    test_eligible,
                    normalized,
                    scaler,
                    strict_supported,
                    method=model,
                    policy=policy,
                    seed=seed,
                    protocol=protocol,
                    fold_id=fold_id,
                )
            )

    training_graph = None
    inference_graph = None
    weight_hash = None
    if model in {"graphsage", "gcn", "gat"}:
        graph_features = GraphFeatures.fit(train)
        if policy_config["isolated"]:
            training_rows = pd.concat(
                [train_labeled, validation_labeled], ignore_index=True
            )
            training_graph = make_graph(
                training_rows,
                train_labeled,
                graph_features,
                scaler,
                seed=seed + 80_000,
                unknown_mask_probability=config.unknown_mask_probability,
            )
            graph_isolation_audit(training_graph, set(test_systems))
            inference_rows = pd.concat(
                [training_rows, test_eligible], ignore_index=True
            )
        else:
            training_rows = frame.reset_index(drop=True)
            training_graph = make_graph(
                training_rows,
                train_labeled,
                graph_features,
                scaler,
                seed=seed + 80_000,
                unknown_mask_probability=config.unknown_mask_probability,
            )
            inference_rows = training_rows

        inference_graph = make_graph(
            inference_rows,
            train_labeled,
            graph_features,
            scaler,
            seed=seed + 80_000,
            unknown_mask_probability=config.unknown_mask_probability,
        )
        train_mask = np.asarray(
            [
                system in set(train_systems)
                for system in training_graph["result"].system_id
            ]
        )
        validation_masks = np.asarray(
            [
                system in set(validation_systems)
                for system in training_graph["result"].system_id
            ]
        )
        # Unsupported groups carry a NaN normalized target. Restrict the loss
        # masks even when an audit policy retains the leaky all-row graph.
        graph_row_uids = pd.Series(training_graph["result"].row_uid)
        train_mask = train_mask & graph_row_uids.isin(
            set(train_labeled["row_uid"])
        ).to_numpy()
        validation_masks = validation_masks & graph_row_uids.isin(
            set(validation_labeled["row_uid"])
        ).to_numpy()
        trained = fit_gnn(
            training_graph,
            train_mask,
            validation_masks,
            model_name=model,
            hidden=config.hidden,
            layers=config.layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            patience=config.patience,
            seed=seed,
        )
        weight_hash = trained.weight_sha256
        inference_predictions = predict_gnn(trained.model, inference_graph)
        inference_uids = pd.Series(inference_graph["result"].row_uid)
        test_uids = set(test_eligible["row_uid"])
        test_positions = np.flatnonzero(inference_uids.isin(test_uids).to_numpy())
        if len(test_positions) != len(test_eligible):
            raise AssertionError("Inference graph does not contain every test row")
        outputs.append(
            _prediction_frame(
                test_eligible,
                inference_predictions[test_positions],
                scaler,
                strict_supported,
                method=model,
                policy=policy,
                seed=seed,
                protocol=protocol,
                fold_id=fold_id,
            )
        )

    predictions = (
        pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    )
    return FoldEvaluation(
        predictions=predictions,
        unsupported=unsupported,
        train_systems=train_systems,
        validation_systems=validation_systems,
        test_systems=test_systems,
        scaler=scaler,
        vocabulary=vocabulary,
        baseline_train_row_uids=tuple(train_labeled["row_uid"]),
        baseline_test_row_uids=tuple(test_eligible["row_uid"]),
        training_graph=training_graph,
        inference_graph=inference_graph,
        weight_sha256=weight_hash,
        trained_weights=(
            {
                key: value.detach().cpu().clone()
                for key, value in trained.model.state_dict().items()
            }
            if weight_hash is not None
            else {}
        ),
    )


def cross_validate_model(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    model: str,
    policy: str,
    seed: int,
    config: GNNConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_ids = sorted(
        assignments.loc[assignments["protocol"].eq(protocol), "fold_id"].unique()
    )
    prediction_frames = []
    unsupported_frames = []
    for fold_id in fold_ids:
        result = evaluate_fold_model(
            frame,
            assignments,
            protocol=protocol,
            fold_id=fold_id,
            model=model,
            policy=policy,
            seed=seed,
            config=config,
        )
        if not result.predictions.empty:
            prediction_frames.append(result.predictions)
        if not result.unsupported.empty:
            unsupported_frames.append(result.unsupported)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    unsupported = (
        pd.concat(unsupported_frames, ignore_index=True)
        if unsupported_frames
        else pd.DataFrame()
    )
    if predictions.duplicated(
        ["method", "evaluation_policy", "seed", "row_uid"]
    ).any():
        raise AssertionError("Cross-validation produced duplicate OOF predictions")
    return predictions, unsupported
