#!/usr/bin/env python3
"""Generate LOAFO roles from the approved 15-family map without fitting."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file
from evaluation.folds import PHASE2_LOSO_SEED, _validation_systems


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "folds"
FAMILY_MAP = OUTPUT / "loafo_family_mapping_proposed.csv"
SYSTEM_ASSIGNMENT = OUTPUT / "loafo_system_assignment_proposed.csv"
FOLD_ASSIGNMENTS = OUTPUT / "loafo_fold_assignments.csv"
FOLD_SUMMARY = OUTPUT / "loafo_fold_summary.csv"
FOLD_MANIFEST = OUTPUT / "loafo_fold_manifest.json"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> None:
    families = pd.read_csv(FAMILY_MAP)
    systems = pd.read_csv(SYSTEM_ASSIGNMENT)
    expected_family_columns = {
        "family_id",
        "device_class",
        "holdout_disposition",
        "system_ids",
    }
    if not expected_family_columns.issubset(families.columns):
        raise ValueError("Approved LOAFO family map is missing columns")
    if len(families) != 15 or len(systems) != 105:
        raise AssertionError("Approved LOAFO map must cover 15 families and 105 systems")
    family_by_system = systems.set_index("system_id")["proposed_family_id"]
    if set(families["family_id"]) != set(family_by_system.unique()):
        raise AssertionError("Family table and system assignment disagree")

    metadata = systems.set_index("system_id")
    all_systems = np.array(sorted(metadata.index.tolist()), dtype=object)
    mixed_families = set(
        families.loc[families["device_class"].eq("mixed"), "family_id"]
    )
    mixed_systems = set(
        system_id
        for system_id, family in family_by_system.items()
        if family in mixed_families
    )
    if len(mixed_systems) != 2:
        raise AssertionError("Expected exactly two always-train mixed systems")

    rows = []
    summary_rows = []
    for family_number, family in enumerate(families.itertuples(index=False)):
        if family.family_id in mixed_families:
            continue
        test_systems = set(family.system_ids.split(";"))
        outer_train = np.array(
            [system for system in all_systems if system not in test_systems],
            dtype=object,
        )
        validation_pool = np.array(
            [system for system in outer_train if system not in mixed_systems],
            dtype=object,
        )
        validation = _validation_systems(
            validation_pool,
            seed=PHASE2_LOSO_SEED + 3000 + family_number,
        )
        fold_id = f"family_{family_number:02d}_{family.family_id}"
        for system_id in all_systems:
            role = (
                "test"
                if system_id in test_systems
                else "validation"
                if system_id in set(validation)
                else "train"
            )
            rows.append(
                {
                    "protocol": "loafo",
                    "fold_id": fold_id,
                    "family_id": family.family_id,
                    "device_class": family.device_class,
                    "holdout_disposition": family.holdout_disposition,
                    "family_number": family_number,
                    "validation_seed": PHASE2_LOSO_SEED + 3000 + family_number,
                    "system_id": system_id,
                    "role": role,
                    "suite": metadata.loc[system_id, "suite"],
                    "accelerator_family": family_by_system.loc[system_id],
                    "measurements": int(metadata.loc[system_id, "measurements"]),
                }
            )
        summary_rows.append(
            {
                "protocol": "loafo",
                "fold_id": fold_id,
                "family_id": family.family_id,
                "device_class": family.device_class,
                "holdout_disposition": family.holdout_disposition,
                "systems": len(test_systems),
                "rows": int(family.rows),
                "validation_seed": PHASE2_LOSO_SEED + 3000 + family_number,
            }
        )

    assignments = pd.DataFrame(rows)
    for fold_id, local in assignments.groupby("fold_id", sort=True):
        if local["role"].eq("test").sum() == 0:
            raise AssertionError(f"{fold_id} has no test system")
        mixed_roles = local.loc[
            local["system_id"].isin(mixed_systems), "role"
        ].unique()
        if set(mixed_roles) != {"train"}:
            raise AssertionError(f"{fold_id} does not keep mixed systems in train")
        test_families = set(local.loc[local["role"].eq("test"), "family_id"])
        if len(test_families) != 1:
            raise AssertionError(f"{fold_id} holds out multiple families")
    if assignments["fold_id"].nunique() != 13:
        raise AssertionError("Expected 13 non-mixed LOAFO folds")

    assignments.to_csv(FOLD_ASSIGNMENTS, index=False)
    pd.DataFrame(summary_rows).to_csv(FOLD_SUMMARY, index=False)
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "family_mapping_sha256": sha256_file(FAMILY_MAP),
        "system_assignment_sha256": sha256_file(SYSTEM_ASSIGNMENT),
        "fold_assignments_sha256": sha256_file(FOLD_ASSIGNMENTS),
        "fold_summary_sha256": sha256_file(FOLD_SUMMARY),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "families": 15,
        "heldout_folds": 13,
        "mixed_always_train_systems": sorted(mixed_systems),
        "validation_seed_base": PHASE2_LOSO_SEED + 3000,
        "loafo_model_runs": 0,
    }
    FOLD_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
