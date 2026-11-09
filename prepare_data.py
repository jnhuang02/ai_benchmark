#!/usr/bin/env python3
"""Prepare only the latest MLPerf Datacenter and Edge result CSV exports."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


SUITE_FILES = {
    "datacenter": "Table - Inference_Full Data_data.csv",
    "edge": "Table - Inference_edge.csv",
}

COLUMNS = [
    "suite",
    "version",
    "division",
    "benchmark",
    "model",
    "task_code",
    "scenario",
    "units",
    "result",
    "accuracy",
    "public_id",
    "organization",
    "availability",
    "system_name",
    "system_type",
    "host_processor",
    "host_processor_cores",
    "processors_per_node",
    "accelerator",
    "accelerators_per_node",
    "nodes",
    "total_accelerators",
    "precision",
    "software",
    "operating_system",
    "has_power",
    "system_id",
    "processor_id",
    "accelerator_id",
    "target_group",
    "log_performance",
    "direction",
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", ""),
        errors="coerce",
    )


def first_available(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for name in names:
        if name in frame.columns:
            values = frame[name].map(clean_text)
            result = result.mask(result.eq(""), values)
    return result


def normalize_direction(units: pd.Series) -> pd.Series:
    lowered = units.fillna("").str.lower()
    lower_is_better = lowered.str.contains(
        r"latency|energy|joule|\buj\b|\bmj\b|\bmw\b|\bw\b", regex=True
    )
    higher_is_better = lowered.str.contains(
        r"tokens/s|samples/s|queries/s|tasks/s|qps|throughput", regex=True
    )
    if (lower_is_better & higher_is_better).any():
        raise ValueError("A unit was classified as both higher- and lower-is-better")
    unknown = ~(lower_is_better | higher_is_better)
    if unknown.any():
        values = sorted(units[unknown].fillna("<blank>").unique())
        raise ValueError(f"Unrecognized result unit(s): {values}")
    return np.where(lower_is_better, -1.0, 1.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_integrity(source: Path) -> pd.DataFrame:
    paths = [source / filename for filename in SUITE_FILES.values()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files: " + ", ".join(missing))
    hashes = [file_sha256(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("The Datacenter and Edge CSV files are byte-identical")
    return pd.DataFrame(
        {
            "suite": list(SUITE_FILES),
            "file": paths,
            "sha256": hashes,
            "bytes": [path.stat().st_size for path in paths],
        }
    )


def load_long_csv(path: Path, suite: str) -> pd.DataFrame:
    """Read one UTF-16, tab-separated MLPerf result export."""
    frame = pd.read_csv(path, sep="\t", encoding="utf-16")
    frame["suite"] = suite
    frame["version"] = first_available(frame, ["Version"])
    frame["division"] = first_available(frame, ["Division"])
    frame["benchmark"] = first_available(frame, ["Benchmark"])
    frame["model"] = first_available(frame, ["Model MLC", "Model (copy)"])
    frame["task_code"] = first_available(frame, ["Task"])
    frame["scenario"] = first_available(frame, ["Scenario"])
    frame["units"] = first_available(frame, ["Units"])
    frame["result"] = numeric(first_available(frame, ["Result"]))
    frame["accuracy"] = np.nan
    frame["public_id"] = first_available(frame, ["Public ID"])
    frame["organization"] = first_available(frame, ["Organization"])
    frame["availability"] = first_available(frame, ["Availability"])
    frame["system_name"] = first_available(
        frame, ["System Name (click + for details)", "System Name"]
    )
    frame["system_type"] = first_available(
        frame, ["System Type", "System Type Revised"]
    )
    frame["host_processor"] = first_available(
        frame, ["Processor (click + for details)", "Host Processor Model Name"]
    )
    frame["host_processor_cores"] = numeric(
        first_available(frame, ["Host Processor Core Count"])
    )
    frame["processors_per_node"] = numeric(
        first_available(frame, ["Processors per node", "Host Processors Per Node"])
    )
    frame["accelerator"] = first_available(
        frame, ["Accelerator Model Name", "Accelerator (click + for details)"]
    )
    frame["accelerators_per_node"] = numeric(
        first_available(frame, ["Accelerators per node", "Accelerators Per Node"])
    )
    frame["nodes"] = numeric(first_available(frame, ["# of Nodes"]))
    frame["total_accelerators"] = numeric(
        first_available(frame, ["Total Accelerators1", "Total Accelerators"])
    )
    frame["precision"] = first_available(frame, ["Precision Detailed"])
    frame["software"] = first_available(frame, ["Software1", "Software"])
    frame["operating_system"] = first_available(frame, ["Operating System"])
    frame["has_power"] = first_available(frame, ["Has Power"])
    return frame


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    for column in COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    for column in [
        "suite",
        "version",
        "division",
        "benchmark",
        "model",
        "task_code",
        "scenario",
        "units",
        "public_id",
        "organization",
        "availability",
        "system_name",
        "system_type",
        "host_processor",
        "accelerator",
        "precision",
        "software",
        "operating_system",
        "has_power",
    ]:
        frame[column] = frame[column].map(clean_text)

    frame = frame.copy()
    frame["version"] = frame["version"].str.lower()
    frame["division"] = frame["division"].str.lower().replace({"": "not stated"})
    frame["has_power"] = frame["has_power"].str.lower().replace(
        {"true": "yes", "1": "yes", "false": "no", "0": "no", "": "unknown"}
    )
    frame["result"] = pd.to_numeric(frame["result"], errors="coerce")
    frame = frame.loc[frame["result"].gt(0) & frame["model"].ne("")].copy()

    frame["target_group"] = frame[
        [
            "suite",
            "version",
            "division",
            "benchmark",
            "model",
            "task_code",
            "scenario",
            "units",
        ]
    ].agg("|".join, axis=1)
    # The submitter is part of system identity. Without it, identically named
    # systems operated by different organizations are incorrectly merged.
    frame["system_id"] = frame[
        ["suite", "organization", "system_name", "system_type"]
    ].agg("|".join, axis=1)
    frame["processor_id"] = frame[["suite", "host_processor"]].agg("|".join, axis=1)
    frame["accelerator_id"] = frame[["suite", "accelerator"]].agg("|".join, axis=1)
    frame["direction"] = normalize_direction(frame["units"])
    frame["log_performance"] = frame["direction"] * np.log(frame["result"])
    return frame[COLUMNS].reset_index(drop=True)


def build_dataset(source: Path) -> pd.DataFrame:
    """Load exactly the two supported CSV exports and no other files."""
    source_integrity(source)
    frames = [
        load_long_csv(source / filename, suite)
        for suite, filename in SUITE_FILES.items()
    ]
    result = canonicalize(pd.concat(frames, ignore_index=True, sort=False))
    if set(result["suite"]) != set(SUITE_FILES):
        raise ValueError("Both Datacenter and Edge suites must be present")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Downloads" / "mlcommons_results_latest",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "mlperf_long.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    integrity = source_integrity(args.source)
    result = build_dataset(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    print(integrity[["suite", "file", "sha256", "bytes"]].to_string(index=False))
    print(f"\nWrote {len(result):,} rows to {args.out}")
    print(
        result.groupby(["suite", "version", "division"], dropna=False)
        .agg(
            rows=("result", "size"),
            systems=("system_id", "nunique"),
            public_ids=("public_id", "nunique"),
            target_groups=("target_group", "nunique"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
