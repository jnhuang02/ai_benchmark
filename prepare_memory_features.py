#!/usr/bin/env python3
"""Build per-system accelerator memory features from MLPerf v6.1 system JSONs.

Each canonical system in ``data/mlperf_long.csv`` is linked to the
``closed/<submitter>/systems/<platform>.json`` description that the upstream
``summary.csv`` names for the same submitter, scenario, and result value. The
script writes one reviewed row per ``system_id`` to
``data/system_memory.csv``. It never modifies ``data/mlperf_long.csv``.

Normalization rules (recorded per row in ``normalization_note``):

- ``accelerator_memory_gb`` is the per-accelerator capacity in nominal GB.
  Values reported in MiB (nvidia-smi style) are converted to GiB and snapped
  to the nominal capacity in the accelerator name when within 5%.
- CPU-only systems receive 0 GB and memory type ``none``.
- Heterogeneous systems with two accelerator models receive the
  count-weighted mean capacity and memory type ``mixed``.
- Unified host/accelerator memory (GB10, Jetson Thor) keeps the unified pool
  size and type ``LPDDR5X``.
- An accelerated system whose description reports ``N/A`` keeps a missing
  capacity and an empty type; models impute it inside each training fold.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path.home() / "Documents" / "GitHub" / "inference_results_v6.1"
DEFAULT_DATA = ROOT / "data" / "mlperf_long.csv"
DEFAULT_OUT = ROOT / "data" / "system_memory.csv"
FAMILY_MAP = ROOT / "config" / "accelerator_families.csv"

# Systems whose rows cannot be matched by (submitter, scenario, result) because
# summary.csv rounds or omits them. Each mapping was checked by hand against
# the accelerator name and count in the system description.
MANUAL_SYSTEM_JSON = {
    "edge|Dell|Dell Pro Max (1x GB10)|edge": "Dell/systems/Dell_Pro_Max_GB10x1.json",
    "edge|HPE|HPE ProLiant Compute DL345 Gen12 (1x NVIDIA RTX PRO 4500 "
    "Blackwell Server Edition 32GB TensorRT)|edge": (
        "HPE/systems/HPE_PROLIANT_DL345_GEN12_RTX_PRO_4500_PCIE_32GBx1.json"
    ),
    "edge|HPE|HPE ProLiant Compute DL380 Gen12 (1x NVIDIA RTX PRO 4500 "
    "Blackwell Server Edition 32GB)|edge": (
        "HPE/systems/HPE_PROLIANT_DL380_RTX_PRO_4500_PCIE_32GBx1_TRT.json"
    ),
}

# Heterogeneous systems: (capacity GB, accelerator count) per accelerator model,
# taken from the accelerator_model_name / memory fields of the system JSON.
HETEROGENEOUS_SYSTEMS = {
    "Cisco/systems/Cisco_UCS_H200x8_MI350Xx8_G200_2x2_TOR.json": [
        (141.0, 8),
        (288.0, 8),
    ],
    "Dell_MangoBoost/systems/2x8xMI355X_2x8xMI300X.json": [
        (192.0, 16),
        (288.0, 16),
    ],
}

MEMORY_TYPES = ("HBM2e", "HBM3", "HBM3e", "HBM4", "GDDR6", "GDDR7", "LPDDR5X")
OUTPUT_COLUMNS = [
    "system_id",
    "accelerator",
    "accelerator_memory_gb",
    "accelerator_memory_type",
    "raw_capacity",
    "raw_configuration",
    "source_json",
    "match_method",
    "normalization_note",
]


def summary_json_paths(source: Path) -> pd.DataFrame:
    summary = pd.read_csv(source / "summary.csv", dtype=str)
    summary = summary.loc[summary["Division"].eq("closed")].copy()
    submitter_dir = summary["Location"].str.split("/").str[2]
    summary["source_json"] = (
        "closed/" + submitter_dir + "/systems/" + summary["Platform"] + ".json"
    )
    summary["result_key"] = summary["Result"].astype(float).round(6)
    return summary[["Organization", "Scenario", "result_key", "source_json"]]


def link_systems(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    rows = frame[["system_id", "organization", "scenario", "result"]].copy()
    rows["result_key"] = rows["result"].astype(float).round(6)
    merged = rows.merge(
        summary_json_paths(source),
        left_on=["organization", "scenario", "result_key"],
        right_on=["Organization", "Scenario", "result_key"],
        how="left",
    )
    linked = (
        merged.dropna(subset=["source_json"])
        .groupby("system_id")["source_json"]
        .agg(lambda values: sorted(set(values)))
    )
    records = []
    for system_id in sorted(frame["system_id"].unique()):
        if system_id in linked.index:
            records.append((system_id, linked[system_id], "summary_row_match"))
        elif system_id in MANUAL_SYSTEM_JSON:
            records.append(
                (system_id, ["closed/" + MANUAL_SYSTEM_JSON[system_id]], "manual")
            )
        else:
            raise ValueError(f"No system description found for {system_id}")
    return pd.DataFrame(records, columns=["system_id", "source_json", "match_method"])


def normalize_type(raw: str | None, cpu_only: bool) -> str:
    if cpu_only:
        return "none"
    text = str(raw or "").upper()
    for memory_type in sorted(MEMORY_TYPES, key=len, reverse=True):
        if memory_type.upper() in text:
            return memory_type
    raise ValueError(f"Unrecognized accelerator memory configuration: {raw!r}")


def nominal_from_name(accelerator: str) -> float | None:
    values = [float(item) for item in re.findall(r"(\d+)\s*GB", accelerator)]
    return max(values) if values else None


def normalize_capacity(
    raw: str | None, accelerator: str, host_capacity: str | None
) -> tuple[float, str]:
    text = str(raw or "").strip()
    if text.lower() == "shared with host":
        value = float(re.search(r"([\d.]+)\s*GB", str(host_capacity)).group(1))
        return value, "unified memory; host capacity used"
    gigabytes = re.search(r"([\d.]+)\s*(GB|Gi)\b", text)
    if gigabytes:
        note = "unified memory pool" if "unified" in text.lower() else ""
        return float(gigabytes.group(1)), note
    mebibytes = re.search(r"([\d.]+)\s*MiB", text)
    if mebibytes:
        value = float(mebibytes.group(1)) / 1024.0
        nominal = nominal_from_name(accelerator)
        if nominal is not None and abs(value - nominal) / nominal <= 0.05:
            return nominal, f"{text} snapped to nominal {nominal:g} GB"
        return round(value, 1), f"{text} converted to GiB"
    raise ValueError(f"Unparseable accelerator memory capacity: {raw!r}")


def _append_note(existing: str, addition: str) -> str:
    return f"{existing}; {addition}" if existing else addition


def reconcile_by_accelerator(table: pd.DataFrame) -> pd.DataFrame:
    """Make one accelerator model report one memory type and nominal capacity.

    Memory type is voted within the reviewed accelerator family from
    ``config/accelerator_families.csv`` (which pools spelling variants such as
    "RTX PRO 4500 Blackwell" and "... Server Edition"); a system that
    disagrees with a strict family majority takes the majority type.
    Capacity is reconciled per exact accelerator name. Capacities are only
    replaced when they came from a MiB conversion and lie within 5% of the
    majority, so genuine SKU differences (e.g. GB200 186 GB vs 192 GB) remain.
    """
    table = table.copy()
    table["normalization_note"] = table["normalization_note"].fillna("")
    eligible = table["accelerator"].ne("") & ~table[
        "accelerator_memory_type"
    ].isin(["mixed", "none", ""])
    families = pd.read_csv(FAMILY_MAP, keep_default_na=False)
    family_of = dict(zip(families["accelerator"], families["accelerator_family"]))
    table["_family"] = table["accelerator"].map(family_of).fillna(table["accelerator"])
    for family, group in table.loc[eligible].groupby("_family"):
        types = group["accelerator_memory_type"].value_counts()
        if len(types) > 1 and types.iloc[0] > len(group) / 2:
            majority = types.index[0]
            for index in group.index[group["accelerator_memory_type"].ne(majority)]:
                reported = table.at[index, "accelerator_memory_type"]
                table.at[index, "accelerator_memory_type"] = majority
                table.at[index, "normalization_note"] = _append_note(
                    table.at[index, "normalization_note"],
                    f"type {reported} overridden by {majority} reported by "
                    f"{types.iloc[0]}/{len(group)} {family} systems",
                )
    for accelerator, group in table.loc[eligible].groupby("accelerator"):
        from_mib = group["raw_capacity"].astype(str).str.contains("MiB")
        reported_gb = group.loc[~from_mib, "accelerator_memory_gb"].value_counts()
        if reported_gb.empty:
            continue
        nominal = float(reported_gb.index[0])
        for index in group.index[from_mib]:
            value = table.at[index, "accelerator_memory_gb"]
            if value != nominal and abs(value - nominal) / nominal <= 0.05:
                table.at[index, "accelerator_memory_gb"] = nominal
                table.at[index, "normalization_note"] = _append_note(
                    table.at[index, "normalization_note"],
                    f"snapped to {nominal:g} GB reported by other "
                    f"{accelerator} systems",
                )
    return table.drop(columns="_family")


def build_memory_table(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    links = link_systems(frame, source)
    systems = frame.drop_duplicates("system_id").set_index("system_id")
    records = []
    for link in links.itertuples(index=False):
        system = systems.loc[link.system_id]
        cpu_only = float(system["total_accelerators"]) == 0
        accelerator = "" if cpu_only else str(system["accelerator"])
        values = []
        for relative in link.source_json:
            description = json.loads((source / relative).read_text())
            values.append(
                (
                    relative,
                    description.get("accelerator_memory_capacity"),
                    description.get("accelerator_memory_configuration"),
                    description.get("host_memory_capacity"),
                )
            )
        relative, raw_capacity, raw_configuration, host_capacity = values[0]
        relative_key = relative.removeprefix("closed/")
        if cpu_only:
            capacity, memory_type, note = 0.0, "none", "CPU-only system"
        elif relative_key in HETEROGENEOUS_SYSTEMS:
            parts = HETEROGENEOUS_SYSTEMS[relative_key]
            capacity = float(
                np.average([gb for gb, _ in parts], weights=[n for _, n in parts])
            )
            memory_type = "mixed"
            note = "heterogeneous accelerators; count-weighted mean capacity"
        elif str(raw_capacity).strip().upper() in {"N/A", "NONE", ""}:
            capacity, memory_type = np.nan, ""
            note = "accelerated system reports N/A; left missing"
        else:
            capacity, note = normalize_capacity(
                raw_capacity, accelerator, host_capacity
            )
            memory_type = normalize_type(raw_configuration, cpu_only)
            for other in values[1:]:
                other_capacity, _ = normalize_capacity(
                    other[1], accelerator, other[3]
                )
                if other_capacity != capacity or (
                    normalize_type(other[2], cpu_only) != memory_type
                ):
                    raise ValueError(
                        f"Conflicting memory descriptions for {link.system_id}"
                    )
        records.append(
            {
                "system_id": link.system_id,
                "accelerator": accelerator,
                "accelerator_memory_gb": capacity,
                "accelerator_memory_type": memory_type,
                "raw_capacity": raw_capacity,
                "raw_configuration": raw_configuration,
                "source_json": ";".join(link.source_json),
                "match_method": link.match_method,
                "normalization_note": note,
            }
        )
    table = reconcile_by_accelerator(pd.DataFrame(records, columns=OUTPUT_COLUMNS))
    if table["system_id"].duplicated().any():
        raise AssertionError("Memory table is not unique by system_id")
    if set(table["system_id"]) != set(frame["system_id"]):
        raise AssertionError("Memory table does not cover every system")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data, dtype={"public_id": str})
    table = build_memory_table(frame, args.source)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} systems to {args.out}")
    print(table["match_method"].value_counts().to_string())
    print(table["accelerator_memory_type"].value_counts().to_string())
    print(
        table.groupby("accelerator_memory_type")["accelerator_memory_gb"]
        .agg(["min", "median", "max"])
        .to_string()
    )
    flagged = table.loc[table["normalization_note"].ne("")]
    print("\nRows with a normalization note (review these):")
    print(
        flagged[
            ["system_id", "raw_capacity", "accelerator_memory_gb", "normalization_note"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
