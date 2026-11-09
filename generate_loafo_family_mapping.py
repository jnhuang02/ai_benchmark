#!/usr/bin/env python3
"""Generate the proposed 15-family LOAFO map without fitting any model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.data import (
    DEFAULT_DATA,
    load_dataset,
    load_family_map,
    sha256_file,
    system_table,
)
from evaluation.features import StrictTargetScaler


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "folds"

FAMILY_RENAMES = {
    "amd_mi300x": "amd_cdna3_mi300x",
    "amd_mi350x": "amd_cdna4_mi350x",
    "amd_mi355x": "amd_cdna4_mi355x",
    "amd_radeon_8060s": "amd_rdna35_integrated_radeon_8060s",
    "cpu_only": "cpu_only_intel_xeon6_granite_rapids",
    "intel_arc_pro_b70": "intel_xe2_battlemage_arc_pro_b70",
    "mixed_amd_mi300x_mi355x": "mixed_amd_cdna3_mi300x_cdna4_mi355x",
    "mixed_nvidia_h200_amd_mi350x": "mixed_nvidia_hopper_h200_amd_cdna4_mi350x",
    "nvidia_blackwell_b200": "nvidia_blackwell_b200",
    "nvidia_blackwell_b300": "nvidia_blackwell_b300",
    "nvidia_blackwell_gb10": "nvidia_blackwell_integrated_gb10",
    "nvidia_blackwell_jetson_thor": "nvidia_blackwell_integrated_thor",
    "nvidia_blackwell_rtx_pro": "nvidia_blackwell_rtx_pro",
    "nvidia_hopper_h200": "nvidia_hopper_h200",
    "nvidia_rubin_vr200": "nvidia_rubin_vr200",
}

FAMILY_METADATA = {
    "amd_cdna3_mi300x": ("AMD", "CDNA3 MI300X", "discrete"),
    "amd_cdna4_mi350x": ("AMD", "CDNA4 MI350X", "discrete"),
    "amd_cdna4_mi355x": ("AMD", "CDNA4 MI355X", "discrete"),
    "amd_rdna35_integrated_radeon_8060s": (
        "AMD",
        "RDNA 3.5 integrated Radeon 8060S",
        "integrated-device",
    ),
    "cpu_only_intel_xeon6_granite_rapids": (
        "Intel",
        "Xeon 6 Granite Rapids",
        "cpu-only",
    ),
    "intel_xe2_battlemage_arc_pro_b70": (
        "Intel",
        "Xe2 Battlemage Arc Pro B70",
        "discrete",
    ),
    "mixed_amd_cdna3_mi300x_cdna4_mi355x": (
        "AMD",
        "CDNA3 MI300X + CDNA4 MI355X",
        "mixed",
    ),
    "mixed_nvidia_hopper_h200_amd_cdna4_mi350x": (
        "NVIDIA + AMD",
        "Hopper H200 + CDNA4 MI350X",
        "mixed",
    ),
    "nvidia_blackwell_b200": (
        "NVIDIA",
        "Blackwell B200",
        "discrete",
    ),
    "nvidia_blackwell_b300": (
        "NVIDIA",
        "Blackwell B300",
        "discrete",
    ),
    "nvidia_blackwell_integrated_gb10": (
        "NVIDIA",
        "Blackwell integrated GB10",
        "integrated-device",
    ),
    "nvidia_blackwell_integrated_thor": (
        "NVIDIA",
        "Blackwell integrated Jetson Thor",
        "integrated-device",
    ),
    "nvidia_blackwell_rtx_pro": (
        "NVIDIA",
        "Blackwell RTX PRO",
        "discrete",
    ),
    "nvidia_hopper_h200": (
        "NVIDIA",
        "Hopper H200",
        "discrete",
    ),
    "nvidia_rubin_vr200": (
        "NVIDIA",
        "Rubin VR200",
        "discrete",
    ),
}


def main() -> None:
    frame = load_dataset(DEFAULT_DATA)
    family_map = load_family_map()
    systems = system_table(frame, family_map)
    systems["proposed_family_id"] = systems["accelerator_family"].map(
        FAMILY_RENAMES
    )
    if systems["proposed_family_id"].isna().any():
        missing = sorted(
            systems.loc[
                systems["proposed_family_id"].isna(), "accelerator_family"
            ].unique()
        )
        raise AssertionError(f"Unmapped existing families: {missing}")

    all_system_ids = set(systems["system_id"])
    family_rows = []
    for family_id, local in systems.groupby(
        "proposed_family_id", sort=True
    ):
        vendor, architecture, device_class = FAMILY_METADATA[family_id]
        train_systems = all_system_ids - set(local["system_id"])
        train = frame.loc[frame["system_id"].isin(train_systems)]
        test = frame.loc[frame["system_id"].isin(local["system_id"])]
        if device_class == "mixed":
            supported_rows = pd.NA
            supported_groups = pd.NA
        else:
            scaler = StrictTargetScaler.fit(train)
            support = test["target_group"].isin(scaler.supported_groups)
            supported_rows = int(support.sum())
            supported_groups = int(
                test.loc[support, "target_group"].nunique()
            )
        if device_class == "mixed":
            disposition = "always_train"
            headline = False
            report_separately = False
        elif device_class == "cpu-only":
            disposition = "stress_case_outside_headline"
            headline = False
            report_separately = True
        elif len(local) >= 3:
            disposition = "headline_holdout"
            headline = True
            report_separately = False
        else:
            disposition = "small_family_reported_separately"
            headline = False
            report_separately = True
        family_rows.append(
            {
                "family_id": family_id,
                "vendor": vendor,
                "architecture_generation": architecture,
                "device_class": device_class,
                "systems": len(local),
                "rows": int(local["measurements"].sum()),
                "supported_rows_if_held_out": supported_rows,
                "supported_target_groups_if_held_out": supported_groups,
                "holdout_disposition": disposition,
                "headline_eligible": headline,
                "reported_separately": report_separately,
                "system_ids": ";".join(sorted(local["system_id"])),
            }
        )
    family_table = pd.DataFrame(family_rows)
    if len(family_table) != 15:
        raise AssertionError(f"Expected 15 families, found {len(family_table)}")
    if family_table["systems"].sum() != 105:
        raise AssertionError("Family system counts do not sum to 105")

    system_table_output = systems[
        [
            "system_id",
            "suite",
            "submitter",
            "accelerator",
            "processor",
            "accelerator_family",
            "proposed_family_id",
            "accelerator_count_values",
            "measurements",
        ]
    ].sort_values("system_id")

    OUTPUT.mkdir(exist_ok=True)
    family_path = OUTPUT / "loafo_family_mapping_proposed.csv"
    assignment_path = OUTPUT / "loafo_system_assignment_proposed.csv"
    family_table.to_csv(family_path, index=False)
    system_table_output.to_csv(assignment_path, index=False)
    manifest = {
        "families": len(family_table),
        "systems": len(system_table_output),
        "rows": int(family_table["rows"].sum()),
        "proposed_only": True,
        "loafo_model_runs": 0,
        "family_mapping_sha256": sha256_file(family_path),
        "system_assignment_sha256": sha256_file(assignment_path),
        "data_sha256": sha256_file(DEFAULT_DATA),
    }
    (OUTPUT / "loafo_family_mapping_proposed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
