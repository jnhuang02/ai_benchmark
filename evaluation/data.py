"""Data loading, system identity, and accelerator-family mapping."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "mlperf_long.csv"
DEFAULT_FAMILY_MAP = ROOT / "config" / "accelerator_families.csv"
UNKNOWN = "<unknown>"

CATEGORICAL_COLUMNS = [
    "organization",
    "system_type",
    "accelerator",
    "host_processor",
    "target_group",
    "precision",
]
NUMERIC_COLUMNS = [
    "nodes",
    "accelerators_per_node",
    "processors_per_node",
    "host_processor_cores",
    "total_accelerators",
]

# Feature-set switch. "frozen" reproduces protocol v2.1 exactly. "memory" is a
# post-lock exploratory extension that adds per-accelerator memory capacity and
# memory type from data/system_memory.csv (built by prepare_memory_features.py).
# It is read once at import time so every module sees the same column lists.
FEATURE_SETS = ("frozen", "memory")
FEATURE_SET = os.environ.get("AI_BENCHMARK_FEATURE_SET", "frozen")
if FEATURE_SET not in FEATURE_SETS:
    raise ValueError(
        f"AI_BENCHMARK_FEATURE_SET must be one of {FEATURE_SETS}, got {FEATURE_SET!r}"
    )
DEFAULT_MEMORY_TABLE = ROOT / "data" / "system_memory.csv"
MEMORY_CATEGORICAL_COLUMNS = ["accelerator_memory_type"]
MEMORY_NUMERIC_COLUMNS = ["accelerator_memory_gb"]
if FEATURE_SET == "memory":
    CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS + MEMORY_CATEGORICAL_COLUMNS
    NUMERIC_COLUMNS = NUMERIC_COLUMNS + MEMORY_NUMERIC_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"public_id": str})
    frame.insert(0, "row_uid", [f"row:{index:04d}" for index in range(len(frame))])
    required = {
        "suite",
        "version",
        "division",
        "organization",
        "system_name",
        "system_type",
        "system_id",
        "accelerator",
        "host_processor",
        "target_group",
        "precision",
        "result",
        "log_performance",
        "direction",
        *(column for column in NUMERIC_COLUMNS if column not in MEMORY_NUMERIC_COLUMNS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Canonical data is missing columns: {sorted(missing)}")
    if frame["result"].le(0).any():
        raise ValueError("All performance results must be positive")
    expected_id = frame[
        ["suite", "organization", "system_name", "system_type"]
    ].astype(str).agg("|".join, axis=1)
    if not frame["system_id"].eq(expected_id).all():
        raise ValueError("system_id does not use the submitter-aware system key")
    for column in ["organization", "accelerator", "host_processor", "precision"]:
        frame[column] = frame[column].fillna("").astype(str)
    frame["cpu_only"] = frame["total_accelerators"].eq(0).astype(float)
    if FEATURE_SET == "memory":
        frame = attach_memory_features(frame)
    return frame


def feature_set_manifest() -> dict[str, object]:
    """Fields every run manifest records so feature sets are never confused."""
    manifest: dict[str, object] = {
        "feature_set": FEATURE_SET,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
    }
    if FEATURE_SET == "memory":
        manifest["memory_table_sha256"] = sha256_file(DEFAULT_MEMORY_TABLE)
    return manifest


def load_memory_table(path: Path = DEFAULT_MEMORY_TABLE) -> pd.DataFrame:
    table = pd.read_csv(path, keep_default_na=False, na_values={"accelerator_memory_gb": [""]})
    expected = {"system_id", *MEMORY_CATEGORICAL_COLUMNS, *MEMORY_NUMERIC_COLUMNS}
    missing = expected.difference(table.columns)
    if missing:
        raise ValueError(f"Memory table is missing columns: {sorted(missing)}")
    if table["system_id"].duplicated().any():
        raise ValueError("Memory table is not unique by system_id")
    return table


def attach_memory_features(
    frame: pd.DataFrame, path: Path = DEFAULT_MEMORY_TABLE
) -> pd.DataFrame:
    """Join per-system memory features; missing values stay missing."""
    table = load_memory_table(path)[
        ["system_id", *MEMORY_CATEGORICAL_COLUMNS, *MEMORY_NUMERIC_COLUMNS]
    ]
    uncovered = set(frame["system_id"]) - set(table["system_id"])
    if uncovered:
        raise ValueError(
            f"Memory table lacks {len(uncovered)} systems; rerun prepare_memory_features.py"
        )
    merged = frame.merge(table, on="system_id", how="left", validate="many_to_one")
    merged["accelerator_memory_type"] = (
        merged["accelerator_memory_type"].fillna("").astype(str)
    )
    return merged


def load_family_map(path: Path = DEFAULT_FAMILY_MAP) -> pd.DataFrame:
    mapping = pd.read_csv(path, keep_default_na=False)
    expected = {"accelerator", "accelerator_family", "review_status", "rationale"}
    if set(mapping.columns) != expected:
        raise ValueError(f"Accelerator map columns must be {sorted(expected)}")
    mapping["accelerator_key"] = mapping["accelerator"].replace(
        {"<blank>": ""}
    )
    if mapping["accelerator_key"].duplicated().any():
        duplicated = mapping.loc[
            mapping["accelerator_key"].duplicated(keep=False), "accelerator_key"
        ]
        raise ValueError(f"Duplicate accelerator mappings: {sorted(set(duplicated))}")
    return mapping


def _join_unique(values: pd.Series, numeric: bool = False) -> str:
    unique = sorted(set(values.dropna()))
    if numeric:
        return ";".join(f"{float(value):g}" for value in unique)
    return ";".join(sorted(set(str(value) for value in unique)))


def system_table(
    frame: pd.DataFrame, family_map: pd.DataFrame
) -> pd.DataFrame:
    mapping = family_map.set_index("accelerator_key")
    rows = []
    for system_id, group in frame.groupby("system_id", sort=True):
        first = group.iloc[0]
        accelerators = sorted(set(group["accelerator"].fillna("").astype(str)))
        if len(accelerators) != 1:
            raise ValueError(
                f"{system_id} has multiple accelerator names: {accelerators}"
            )
        accelerator = accelerators[0]
        if accelerator not in mapping.index:
            raise ValueError(f"No accelerator-family mapping for {accelerator!r}")
        count_values = _join_unique(group["total_accelerators"], numeric=True)
        sensitivity_id = "|".join(
            [
                str(first["organization"]),
                accelerator,
                str(first["host_processor"]),
            ]
        )
        mapped = mapping.loc[accelerator]
        rows.append(
            {
                "system_id": system_id,
                "suite": first["suite"],
                "submitter": first["organization"],
                "system_name": first["system_name"],
                "system_type": first["system_type"],
                "accelerator": accelerator,
                "accelerator_family": mapped["accelerator_family"],
                "family_review_status": mapped["review_status"],
                "family_rationale": mapped["rationale"],
                "accelerator_count_values": count_values,
                "processor": first["host_processor"],
                "sensitivity_group_id": sensitivity_id,
                "measurements": len(group),
            }
        )
    systems = pd.DataFrame(rows)
    if systems["system_id"].duplicated().any():
        raise ValueError("System table is not unique by system_id")
    return systems

