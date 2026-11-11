"""Pre-registered Phase 2 fold fitting and model ladder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch_geometric.data import HeteroData

from evaluation.data import (
    CATEGORICAL_COLUMNS,
    UNKNOWN,
    load_dataset,
    load_family_map,
    system_table,
)
from evaluation.evaluate import (
    GNNConfig,
    _frame_for_systems,
    fold_role_sets,
)
from evaluation.features import (
    CategoryVocabulary,
    PHASE2_NUMERIC_COLUMNS,
    StrictTargetScaler,
    mask_identity_values,
    normalized_category,
)
from evaluation.graph import (
    GraphFeatures,
    RELATION_EDGE_TYPES,
    degree_preserving_shuffled_graph,
    fit_gnn,
    graph_edge_types,
    graph_isolation_audit,
    make_graph,
    predict_gnn,
    subset_graph_edges,
)


DETERMINISTIC_METHODS = (
    "group_median",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "hierarchical_mixed",
)
NEURAL_METHODS = (
    "mlp",
    "graphsage",
    "gcn",
    "gat",
    "graphsage_shuffled",
    "graphsage_no_system",
    "graphsage_no_accelerator",
    "graphsage_no_processor",
    "graphsage_no_workload",
)
MODEL_ORDER = DETERMINISTIC_METHODS + NEURAL_METHODS


def load_phase2_frame() -> pd.DataFrame:
    frame = load_dataset()
    systems = system_table(frame, load_family_map())[
        ["system_id", "accelerator_family"]
    ]
    frame = frame.merge(systems, on="system_id", how="left", validate="many_to_one")
    if frame["accelerator_family"].isna().any():
        raise AssertionError("Phase 2 frame has unmapped accelerator families")
    return frame


def phase2_numeric_values(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(PHASE2_NUMERIC_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Phase 2 numeric columns: {sorted(missing)}")
    values = frame[PHASE2_NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    transformed = values.clip(lower=0).copy()
    for column in PHASE2_NUMERIC_COLUMNS:
        if column != "cpu_only":
            transformed[column] = np.log1p(transformed[column])
    return transformed


@dataclass
class TabularFeatures:
    vocabulary: CategoryVocabulary
    categorical_encoder: OneHotEncoder
    numeric_imputer: SimpleImputer
    numeric_scaler: StandardScaler

    @classmethod
    def fit(
        cls, train_masked: pd.DataFrame, vocabulary: CategoryVocabulary
    ) -> "TabularFeatures":
        categories = [
            list(vocabulary.categories[column])
            for column in CATEGORICAL_COLUMNS
        ]
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        encoder.fit(train_masked[CATEGORICAL_COLUMNS])
        numeric = phase2_numeric_values(train_masked)
        imputer = SimpleImputer(strategy="median")
        numeric_filled = imputer.fit_transform(numeric)
        scaler = StandardScaler().fit(numeric_filled)
        return cls(vocabulary, encoder, imputer, scaler)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        categorical = self.vocabulary.encode(frame)[CATEGORICAL_COLUMNS]
        categorical_features = self.categorical_encoder.transform(categorical)
        numeric = self.numeric_scaler.transform(
            self.numeric_imputer.transform(phase2_numeric_values(frame))
        )
        return np.hstack([categorical_features, numeric]).astype(np.float32)


def masked_tabular_features(
    train_labeled: pd.DataFrame,
    evaluation_frames: list[pd.DataFrame],
    vocabulary: CategoryVocabulary,
    *,
    seed: int,
    probability: float = 0.1,
) -> tuple[TabularFeatures, list[np.ndarray]]:
    masked_train = mask_identity_values(
        train_labeled, seed=seed + 70_000, probability=probability
    )
    features = TabularFeatures.fit(masked_train, vocabulary)
    return features, [
        features.transform(frame) for frame in evaluation_frames
    ]


def category_index_matrix(
    frame: pd.DataFrame, vocabulary: CategoryVocabulary
) -> np.ndarray:
    encoded = vocabulary.encode(frame)
    columns = []
    for column in CATEGORICAL_COLUMNS:
        lookup = {
            value: index
            for index, value in enumerate(vocabulary.categories[column])
        }
        columns.append(encoded[column].map(lookup).to_numpy(dtype=np.int64))
    return np.stack(columns, axis=1)


class MLPWithoutEdges(nn.Module):
    def __init__(
        self,
        vocabulary: CategoryVocabulary,
        numeric_features: int,
        hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dimensions = []
        self.embeddings = nn.ModuleList()
        for column in CATEGORICAL_COLUMNS:
            count = len(vocabulary.categories[column])
            dimension = min(16, max(2, round(math.sqrt(count))))
            dimensions.append(dimension)
            self.embeddings.append(nn.Embedding(count, dimension))
        self.layers = nn.Sequential(
            nn.Linear(sum(dimensions) + numeric_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, categorical: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        values = torch.cat([*embedded, numeric], dim=1)
        return self.layers(values).squeeze(-1)


def fit_mlp(
    train_labeled: pd.DataFrame,
    validation_labeled: pd.DataFrame,
    test: pd.DataFrame,
    vocabulary: CategoryVocabulary,
    scaler: StrictTargetScaler,
    *,
    seed: int,
    config: GNNConfig,
) -> np.ndarray:
    masked_train = mask_identity_values(
        train_labeled,
        seed=seed + 70_000,
        probability=config.unknown_mask_probability,
    )
    features = TabularFeatures.fit(masked_train, vocabulary)
    train_x = features.transform(masked_train)
    validation_x = features.transform(validation_labeled)
    test_x = features.transform(test)

    train_cat = torch.as_tensor(
        category_index_matrix(masked_train, vocabulary), dtype=torch.long
    )
    validation_cat = torch.as_tensor(
        category_index_matrix(validation_labeled, vocabulary), dtype=torch.long
    )
    test_cat = torch.as_tensor(
        category_index_matrix(test, vocabulary), dtype=torch.long
    )
    train_num = torch.as_tensor(train_x, dtype=torch.float32)
    validation_num = torch.as_tensor(validation_x, dtype=torch.float32)
    test_num = torch.as_tensor(test_x, dtype=torch.float32)
    train_y = torch.as_tensor(
        scaler.transform(train_labeled), dtype=torch.float32
    )
    validation_y = torch.as_tensor(
        scaler.transform(validation_labeled), dtype=torch.float32
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPWithoutEdges(
        vocabulary,
        numeric_features=train_num.shape[1],
        hidden=config.hidden,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(train_cat, train_num)
        loss = nn.functional.huber_loss(
            prediction, train_y, delta=1.0
        )
        loss.backward()
        optimizer.step()
        if len(validation_y):
            model.eval()
            with torch.no_grad():
                validation_loss = nn.functional.huber_loss(
                    model(validation_cat, validation_num),
                    validation_y,
                    delta=1.0,
                ).item()
            if validation_loss < best_loss - 1e-12:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(test_cat, test_num).cpu().numpy()


def _group_median_prediction(
    test: pd.DataFrame,
    train_labeled: pd.DataFrame,
    scaler: StrictTargetScaler,
) -> np.ndarray:
    centers = train_labeled.groupby("target_group")["log_performance"].median()
    predicted_signed_log = test["target_group"].map(centers)
    if predicted_signed_log.isna().any():
        raise AssertionError("Unsupported target group reached group median")
    true_signed_log = test["target_group"].map(scaler.center)
    scale = test["target_group"].map(scaler.scale)
    return (
        (predicted_signed_log - true_signed_log) / scale
    ).to_numpy(dtype=float)


def fit_deterministic_models(
    train_labeled: pd.DataFrame,
    validation_labeled: pd.DataFrame,
    test: pd.DataFrame,
    vocabulary: CategoryVocabulary,
    scaler: StrictTargetScaler,
    methods: tuple[str, ...] = DETERMINISTIC_METHODS,
) -> dict[str, np.ndarray]:
    unknown = set(methods) - set(DETERMINISTIC_METHODS)
    if unknown:
        raise ValueError(f"Unsupported deterministic methods: {sorted(unknown)}")
    predictions = {}
    if "group_median" in methods:
        predictions["group_median"] = _group_median_prediction(
            test, train_labeled, scaler
        )
    estimators = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=7,
            n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=7,
        ),
    }
    fitted_methods = [name for name in estimators if name in methods]
    if fitted_methods:
        _, (train_x, test_x) = masked_tabular_features(
            train_labeled,
            [train_labeled, test],
            vocabulary,
            seed=7,
            probability=0.1,
        )
        train_y = scaler.transform(train_labeled)
        for name in fitted_methods:
            estimator = estimators[name]
            estimator.fit(train_x, train_y)
            predictions[name] = estimator.predict(test_x)
    if "hierarchical_mixed" in methods:
        predictions["hierarchical_mixed"] = fit_hierarchical_mixed(
            train_labeled, test, scaler
        )
    return predictions


def _mixed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["log_total"] = np.log1p(output["total_accelerators"])
    output["is_cpu"] = output["cpu_only"]
    memory = output["accelerator"].map(
        lambda value: max(
            (float(item) for item in __import__("re").findall(r"(\d+)\s*GB", str(value))),
            default=np.nan,
        )
    )
    if "accelerator_memory_gb" in output.columns:
        # Memory feature set: use the system-description capacity instead of
        # parsing the accelerator name.
        memory = pd.to_numeric(output["accelerator_memory_gb"], errors="coerce")
    output["accelerator_memory"] = memory
    output["family"] = output["accelerator_family"].fillna("unknown")
    output["submitter"] = output["organization"]
    return output


def fit_hierarchical_mixed(
    train_labeled: pd.DataFrame,
    test: pd.DataFrame,
    scaler: StrictTargetScaler,
) -> np.ndarray:
    prediction, _ = fit_hierarchical_mixed_detailed(
        train_labeled, test, scaler
    )
    return prediction


def fit_hierarchical_mixed_detailed(
    train_labeled: pd.DataFrame,
    test: pd.DataFrame,
    scaler: StrictTargetScaler,
) -> tuple[np.ndarray, dict[str, object]]:
    train = _mixed_frame(train_labeled)
    evaluation = _mixed_frame(test)
    train["y"] = scaler.transform(train_labeled)
    fill_value = train["accelerator_memory"].median()
    if not np.isfinite(fill_value):
        fill_value = 0.0
    train["accelerator_memory"] = train["accelerator_memory"].fillna(fill_value)
    evaluation["accelerator_memory"] = evaluation[
        "accelerator_memory"
    ].fillna(fill_value)
    formula = (
        "y ~ C(target_group) + log_total + is_cpu + accelerator_memory"
    )
    model = smf.mixedlm(
        formula,
        train,
        groups=train["family"],
        vc_formula={
            "submitter": "0 + C(submitter)",
            "system": "0 + C(system_id)",
        },
    )
    fitted = None
    optimizer = None
    for method, maxiter in [
        ("lbfgs", 200),
        ("powell", 500),
        ("nm", 500),
        ("bfgs", 500),
    ]:
        try:
            candidate = model.fit(
                method=method, maxiter=maxiter, disp=False
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        if candidate.converged:
            fitted = candidate
            optimizer = method
            break
    if fitted is None:
        raise RuntimeError("Hierarchical mixed model did not converge")
    prediction = fitted.predict(exog=evaluation).to_numpy(dtype=float)
    variance_values = np.asarray(fitted.vcomp, dtype=float).reshape(-1)
    variance_names = list(fitted.model.exog_vc.names)
    if len(variance_names) != len(variance_values):
        variance_names = [
            f"variance_component_{index}"
            for index in range(len(variance_values))
        ]
    diagnostics = {
        "converged": bool(fitted.converged),
        "optimizer": optimizer,
        "cov_re": fitted.cov_re.to_numpy().tolist(),
        "variance_components": dict(
            zip(variance_names, variance_values.tolist())
        ),
        "formula": formula,
        "groups": "family",
        "random_intercepts": ["family", "submitter", "system_id"],
        "heldout_random_effect": "zero random effect for unseen test system",
    }
    return prediction, diagnostics


def _validation_primary(
    normalized_prediction: np.ndarray,
    validation: pd.DataFrame,
    scaler: StrictTargetScaler,
) -> float:
    prediction = scaler.inverse(validation, normalized_prediction)
    return float(
        np.median(
            np.abs(np.log(prediction) - np.log(validation["result"].to_numpy()))
        )
    )


def _prediction_table(
    test: pd.DataFrame,
    normalized_prediction: np.ndarray,
    scaler: StrictTargetScaler,
    *,
    method: str,
    protocol: str,
    fold_id: str,
    seed: int,
) -> pd.DataFrame:
    prediction = scaler.inverse(test, normalized_prediction)
    return pd.DataFrame(
        {
            "method": method,
            "protocol": protocol,
            "fold_id": fold_id,
            "seed": seed,
            "row_uid": test["row_uid"].to_numpy(),
            "system_id": test["system_id"].to_numpy(),
            "target_group": test["target_group"].to_numpy(),
            "suite": test["suite"].to_numpy(),
            "result": test["result"].to_numpy(dtype=float),
            "direction": test["direction"].to_numpy(dtype=float),
            "prediction": prediction,
            "prediction_log": normalized_prediction,
            "signed_log_prediction": (
                test["direction"].to_numpy(dtype=float) * np.log(prediction)
            ),
            "y": scaler.transform(test),
        }
    )


def fit_neural_models(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    fold_id: str,
    seed: int,
    config: GNNConfig,
    include_secondary_graph_variants: bool = True,
) -> pd.DataFrame:
    train_systems, validation_systems, test_systems = fold_role_sets(
        assignments, protocol, fold_id
    )
    train = _frame_for_systems(frame, set(train_systems))
    validation = _frame_for_systems(frame, set(validation_systems))
    test_all = _frame_for_systems(frame, set(test_systems))
    scaler = StrictTargetScaler.fit(train)
    supported = set(scaler.supported_groups)
    train_labeled = train.loc[train["target_group"].isin(supported)].copy()
    validation_labeled = validation.loc[
        validation["target_group"].isin(supported)
    ].copy()
    test = test_all.loc[test_all["target_group"].isin(supported)].copy()
    if test.empty:
        return pd.DataFrame()
    vocabulary = CategoryVocabulary.fit(train)
    outputs = []

    mlp_prediction = fit_mlp(
        train_labeled,
        validation_labeled,
        test,
        vocabulary,
        scaler,
        seed=seed,
        config=config,
    )
    outputs.append(
        _prediction_table(
            test,
            mlp_prediction,
            scaler,
            method="mlp",
            protocol=protocol,
            fold_id=fold_id,
            seed=seed,
        )
    )

    graph_features = GraphFeatures.fit(
        train, numeric_columns=PHASE2_NUMERIC_COLUMNS
    )
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
    inference_graph = make_graph(
        pd.concat([training_rows, test], ignore_index=True),
        train_labeled,
        graph_features,
        scaler,
        seed=seed + 80_000,
        unknown_mask_probability=config.unknown_mask_probability,
    )
    training_uids = pd.Series(training_graph["result"].row_uid)
    train_mask = training_uids.isin(set(train_labeled["row_uid"])).to_numpy()
    validation_mask = training_uids.isin(
        set(validation_labeled["row_uid"])
    ).to_numpy()
    inference_uids = pd.Series(inference_graph["result"].row_uid)
    test_positions = np.flatnonzero(
        inference_uids.isin(set(test["row_uid"])).to_numpy()
    )

    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_validation: dict[str, float] = {}
    for model_name in ("graphsage", "gcn", "gat"):
        trained = fit_gnn(
            training_graph,
            train_mask,
            validation_mask,
            model_name=model_name,
            hidden=config.hidden,
            layers=config.layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            patience=config.patience,
            seed=seed,
        )
        validation_prediction = predict_gnn(
            trained.model, training_graph
        )[validation_mask]
        candidate_validation[model_name] = _validation_primary(
            validation_prediction, validation_labeled, scaler
        )
        inference_prediction = predict_gnn(
            trained.model, inference_graph
        )
        candidate_predictions[model_name] = inference_prediction[test_positions]
        outputs.append(
            _prediction_table(
                test,
                candidate_predictions[model_name],
                scaler,
                method=model_name,
                protocol=protocol,
                fold_id=fold_id,
                seed=seed,
            )
        )
    tie_order = {"graphsage": 0, "gcn": 1, "gat": 2}
    selected = min(
        ("graphsage", "gcn", "gat"),
        key=lambda name: (candidate_validation[name], tie_order[name]),
    )
    outputs.append(
        _prediction_table(
            test,
            candidate_predictions[selected],
            scaler,
            method="best_gnn",
            protocol=protocol,
            fold_id=fold_id,
            seed=seed,
        ).assign(selected_gnn=selected)
    )

    if not include_secondary_graph_variants:
        return pd.concat(outputs, ignore_index=True)

    graph_variants = {
        "graphsage_shuffled": degree_preserving_shuffled_graph(
            training_graph, seed=seed + 90_000
        ),
        **{
            f"graphsage_no_{relation}": subset_graph_edges(
                training_graph,
                [
                    edge_type
                    for edge_type in graph_edge_types()
                    if edge_type not in set(RELATION_EDGE_TYPES[relation])
                ],
            )
            for relation in RELATION_EDGE_TYPES
        },
    }
    inference_variants = {
        "graphsage_shuffled": degree_preserving_shuffled_graph(
            inference_graph, seed=seed + 90_000
        ),
        **{
            f"graphsage_no_{relation}": subset_graph_edges(
                inference_graph,
                [
                    edge_type
                    for edge_type in graph_edge_types()
                    if edge_type not in set(RELATION_EDGE_TYPES[relation])
                ],
            )
            for relation in RELATION_EDGE_TYPES
        },
    }
    for method, variant_graph in graph_variants.items():
        edge_types = [edge_type for edge_type in variant_graph.edge_types]
        trained = fit_gnn(
            variant_graph,
            train_mask,
            validation_mask,
            model_name="graphsage",
            hidden=config.hidden,
            layers=config.layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            patience=config.patience,
            seed=seed,
            edge_types=edge_types,
        )
        inference_prediction = predict_gnn(
            trained.model, inference_variants[method]
        )
        outputs.append(
            _prediction_table(
                test,
                inference_prediction[test_positions],
                scaler,
                method=method,
                protocol=protocol,
                fold_id=fold_id,
                seed=seed,
            )
        )
    return pd.concat(outputs, ignore_index=True)


def fit_phase2_graph_model(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    fold_id: str,
    model_name: str,
    seed: int,
    config: GNNConfig,
    numeric_columns: tuple[str, ...] = PHASE2_NUMERIC_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Fit one Phase 2 graph model for diagnostics and targeted reruns."""
    if model_name not in {"graphsage", "gcn", "gat"}:
        raise ValueError(f"Unsupported graph diagnostic model: {model_name}")
    train_systems, validation_systems, test_systems = fold_role_sets(
        assignments, protocol, fold_id
    )
    train = _frame_for_systems(frame, set(train_systems))
    validation = _frame_for_systems(frame, set(validation_systems))
    test_all = _frame_for_systems(frame, set(test_systems))
    scaler = StrictTargetScaler.fit(train)
    supported = set(scaler.supported_groups)
    train_labeled = train.loc[train["target_group"].isin(supported)].copy()
    validation_labeled = validation.loc[
        validation["target_group"].isin(supported)
    ].copy()
    test = test_all.loc[test_all["target_group"].isin(supported)].copy()
    if test.empty:
        return pd.DataFrame(), {}
    graph_features = GraphFeatures.fit(
        train, numeric_columns=numeric_columns
    )
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
    inference_graph = make_graph(
        pd.concat([training_rows, test], ignore_index=True),
        train_labeled,
        graph_features,
        scaler,
        seed=seed + 80_000,
        unknown_mask_probability=config.unknown_mask_probability,
    )
    training_uids = pd.Series(training_graph["result"].row_uid)
    train_mask = training_uids.isin(set(train_labeled["row_uid"])).to_numpy()
    validation_mask = training_uids.isin(
        set(validation_labeled["row_uid"])
    ).to_numpy()
    trained = fit_gnn(
        training_graph,
        train_mask,
        validation_mask,
        model_name=model_name,
        hidden=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        patience=config.patience,
        seed=seed,
    )
    inference_uids = pd.Series(inference_graph["result"].row_uid)
    positions = np.flatnonzero(
        inference_uids.isin(set(test["row_uid"])).to_numpy()
    )
    prediction_log = predict_gnn(trained.model, inference_graph)[positions]
    predictions = _prediction_table(
        test,
        prediction_log,
        scaler,
        method=model_name,
        protocol=protocol,
        fold_id=fold_id,
        seed=seed,
    )
    diagnostics = {
        "best_epoch": trained.best_epoch,
        "epochs_run": trained.epochs_run,
        "best_validation_loss": trained.best_validation_loss,
        "final_train_loss": trained.final_train_loss,
        "final_validation_loss": trained.final_validation_loss,
        "prediction_variance": float(
            np.var(predictions["prediction"].to_numpy(dtype=float), ddof=1)
        ),
        "normalized_prediction_variance_mean": float(
            np.mean(
                predictions.groupby("fold_id")["prediction_log"].var(ddof=1)
            )
        ),
        "prediction_unique": int(predictions["prediction"].nunique()),
    }
    return predictions, diagnostics


def fit_deterministic_fold(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    fold_id: str,
    methods: tuple[str, ...] = DETERMINISTIC_METHODS,
) -> pd.DataFrame:
    train_systems, validation_systems, test_systems = fold_role_sets(
        assignments, protocol, fold_id
    )
    train = _frame_for_systems(frame, set(train_systems))
    validation = _frame_for_systems(frame, set(validation_systems))
    test_all = _frame_for_systems(frame, set(test_systems))
    scaler = StrictTargetScaler.fit(train)
    supported = set(scaler.supported_groups)
    train_labeled = train.loc[train["target_group"].isin(supported)].copy()
    validation_labeled = validation.loc[
        validation["target_group"].isin(supported)
    ].copy()
    test = test_all.loc[test_all["target_group"].isin(supported)].copy()
    if test.empty:
        return pd.DataFrame()
    vocabulary = CategoryVocabulary.fit(train)
    predictions = fit_deterministic_models(
        train_labeled,
        validation_labeled,
        test,
        vocabulary,
        scaler,
        methods=methods,
    )
    return pd.concat(
        [
            _prediction_table(
                test,
                values,
                scaler,
                method=method,
                protocol=protocol,
                fold_id=fold_id,
                seed=0,
            )
            for method, values in predictions.items()
        ],
        ignore_index=True,
    )
