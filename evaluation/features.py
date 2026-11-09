"""Leakage-safe target, identity, and numeric feature transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, UNKNOWN


PHASE2_NUMERIC_COLUMNS = NUMERIC_COLUMNS + ["cpu_only"]


def normalized_category(values: pd.Series) -> pd.Series:
    cleaned = values.fillna("").astype(str)
    return cleaned.mask(cleaned.eq(""), UNKNOWN)


@dataclass(frozen=True)
class CategoryVocabulary:
    categories: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "CategoryVocabulary":
        categories = {}
        for column in CATEGORICAL_COLUMNS:
            values = normalized_category(frame[column])
            categories[column] = tuple(sorted(set(values.tolist()) | {UNKNOWN}))
        return cls(categories)

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=frame.index)
        for column in CATEGORICAL_COLUMNS:
            values = normalized_category(frame[column])
            encoded[column] = values.where(
                values.isin(self.categories[column]), UNKNOWN
            )
        return encoded

    def one_hot_array(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        categories = self.categories[column]
        lookup = {value: index for index, value in enumerate(categories)}
        values = normalized_category(frame[column])
        indices = values.map(lookup).fillna(lookup[UNKNOWN]).to_numpy(dtype=int)
        result = np.zeros((len(frame), len(categories)), dtype=np.float32)
        result[np.arange(len(frame)), indices] = 1.0
        return result


@dataclass(frozen=True)
class NumericScaler:
    center: pd.Series
    scale: pd.Series
    columns: tuple[str, ...] = tuple(NUMERIC_COLUMNS)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        numeric_columns: list[str] | tuple[str, ...] = NUMERIC_COLUMNS,
    ) -> "NumericScaler":
        columns = tuple(numeric_columns)
        values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
        transformed = cls._transform_values(values, columns)
        center = transformed.median()
        scale = transformed.std().replace(0, 1).fillna(1)
        return cls(center, scale, columns)

    @staticmethod
    def _transform_values(
        values: pd.DataFrame, columns: tuple[str, ...]
    ) -> pd.DataFrame:
        transformed = values.clip(lower=0).copy()
        for column in columns:
            if column != "cpu_only":
                transformed[column] = np.log1p(transformed[column])
        return transformed

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].apply(pd.to_numeric, errors="coerce")
        transformed = self._transform_values(values, self.columns)
        scaled = (transformed - self.center) / self.scale
        return scaled.fillna(0.0).to_numpy(dtype=np.float32, copy=True)


@dataclass(frozen=True, kw_only=True)
class StrictTargetScaler:
    center: pd.Series
    scale: pd.Series
    supported_groups: tuple[str, ...]
    minimum_rows: int = 3

    @classmethod
    def fit(
        cls, train: pd.DataFrame, minimum_rows: int = 3
    ) -> "StrictTargetScaler":
        grouped = train.groupby("target_group")["log_performance"]
        center = grouped.median()
        scale = grouped.std()
        support = grouped.size().ge(minimum_rows) & scale.gt(1e-8)
        supported = sorted(support[support].index.tolist())
        if not supported:
            raise ValueError("No target group has enough training rows")
        return cls(
            center=center,
            scale=scale,
            supported_groups=tuple(supported),
            minimum_rows=minimum_rows,
        )

    def is_supported(self, value: str) -> bool:
        return value in self.supported_groups

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        center = frame["target_group"].map(self.center)
        scale = frame["target_group"].map(self.scale)
        transformed = (
            frame["log_performance"].to_numpy(dtype=float)
            - center.to_numpy(dtype=float)
        ) / scale.to_numpy(dtype=float)
        transformed[~frame["target_group"].isin(self.supported_groups)] = np.nan
        return transformed

    def inverse(
        self, frame: pd.DataFrame, normalized_prediction: np.ndarray
    ) -> np.ndarray:
        if not frame["target_group"].isin(self.supported_groups).all():
            raise ValueError("Cannot inverse-transform an unsupported target group")
        center = frame["target_group"].map(self.center).to_numpy(dtype=float)
        scale = frame["target_group"].map(self.scale).to_numpy(dtype=float)
        signed_log = normalized_prediction * scale + center
        signed_log = np.clip(signed_log, -50.0, 50.0)
        return np.exp(frame["direction"].to_numpy(dtype=float) * signed_log)


@dataclass(frozen=True, kw_only=True)
class LegacyGlobalTargetScaler(StrictTargetScaler):
    """Audit-only scaler preserving the old global fallback."""

    global_center: float
    global_scale: float

    @classmethod
    def fit(
        cls, train: pd.DataFrame, minimum_rows: int = 3
    ) -> "LegacyGlobalTargetScaler":
        strict = StrictTargetScaler.fit(train, minimum_rows)
        global_scale = float(train["log_performance"].std())
        if not np.isfinite(global_scale) or global_scale <= 1e-8:
            global_scale = 1.0
        return cls(
            center=strict.center,
            scale=strict.scale,
            supported_groups=strict.supported_groups,
            minimum_rows=minimum_rows,
            global_center=float(train["log_performance"].median()),
            global_scale=global_scale,
        )

    def is_supported(self, value: str) -> bool:
        return True

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        center = (
            frame["target_group"]
            .map(self.center)
            .fillna(self.global_center)
            .to_numpy(dtype=float)
        )
        scale = (
            frame["target_group"]
            .map(self.scale)
            .fillna(self.global_scale)
            .to_numpy(dtype=float)
        )
        return (
            frame["log_performance"].to_numpy(dtype=float) - center
        ) / scale

    def inverse(
        self, frame: pd.DataFrame, normalized_prediction: np.ndarray
    ) -> np.ndarray:
        center = (
            frame["target_group"]
            .map(self.center)
            .fillna(self.global_center)
            .to_numpy(dtype=float)
        )
        scale = (
            frame["target_group"]
            .map(self.scale)
            .fillna(self.global_scale)
            .to_numpy(dtype=float)
        )
        signed_log = np.clip(
            normalized_prediction * scale + center, -50.0, 50.0
        )
        return np.exp(frame["direction"].to_numpy(dtype=float) * signed_log)


def mask_identity_values(
    frame: pd.DataFrame,
    seed: int,
    probability: float = 0.1,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Randomly train explicit unknown categories without mutating inputs."""
    if not 0 <= probability < 1:
        raise ValueError("Mask probability must be in [0, 1)")
    masked = frame.copy()
    rng = np.random.default_rng(seed)
    for column in columns or CATEGORICAL_COLUMNS:
        draw = rng.random(len(masked)) < probability
        masked.loc[draw, column] = UNKNOWN
    return masked


@dataclass
class TabularFeatureEncoder:
    vocabulary: CategoryVocabulary
    encoder: OneHotEncoder
    numeric_imputer: SimpleImputer
    numeric_scaler: StandardScaler

    @classmethod
    def fit(
        cls, train_masked: pd.DataFrame, vocabulary: CategoryVocabulary
    ) -> "TabularFeatureEncoder":
        categories = [list(vocabulary.categories[column]) for column in CATEGORICAL_COLUMNS]
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        encoder.fit(train_masked[CATEGORICAL_COLUMNS])
        imputer = SimpleImputer(strategy="median")
        numeric = imputer.fit_transform(train_masked[NUMERIC_COLUMNS])
        scaler = StandardScaler().fit(numeric)
        return cls(vocabulary, encoder, imputer, scaler)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        categorical = self.vocabulary.encode(frame)[CATEGORICAL_COLUMNS]
        categorical_features = self.encoder.transform(categorical)
        numeric = self.numeric_scaler.transform(
            self.numeric_imputer.transform(frame[NUMERIC_COLUMNS])
        )
        return np.hstack([categorical_features, numeric]).astype(np.float32)
