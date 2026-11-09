#!/usr/bin/env python3
"""Write the pre-registered Phase 2 fold file without fitting a model."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from evaluation.data import (
    DEFAULT_DATA,
    load_dataset,
    load_family_map,
    sha256_file,
    system_table,
)
from evaluation.folds import (
    PHASE2_CV5_PARTITION_SEEDS,
    PHASE2_LOSO_SEED,
    PHASE2_SENSITIVITY_SEED,
    build_phase2_fold_specs,
    fold_assignment_table,
    fold_summary,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "folds"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> None:
    frame = load_dataset(DEFAULT_DATA)
    systems = system_table(frame, load_family_map())
    specs = build_phase2_fold_specs(systems)
    assignments = fold_assignment_table(systems, specs)

    seed_map = {
        f"cv5_partition_{index}": seed
        for index, seed in enumerate(PHASE2_CV5_PARTITION_SEEDS)
    }
    seed_map["loso"] = PHASE2_LOSO_SEED
    seed_map["sensitivity"] = PHASE2_SENSITIVITY_SEED
    assignments.insert(
        1, "split_seed", assignments["protocol"].map(seed_map).astype(int)
    )
    assignments.insert(
        2,
        "partition_seed",
        assignments["protocol"].map(
            {
                f"cv5_partition_{index}": seed
                for index, seed in enumerate(PHASE2_CV5_PARTITION_SEEDS)
            }
        ),
    )
    assignments["partition_seed"] = assignments["partition_seed"].astype("Int64")

    expected_systems = set(systems["system_id"])
    for partition_index, seed in enumerate(PHASE2_CV5_PARTITION_SEEDS):
        protocol = f"cv5_partition_{partition_index}"
        local = assignments.loc[assignments["protocol"].eq(protocol)]
        if local["fold_id"].nunique() != 5:
            raise AssertionError(f"{protocol} does not contain five folds")
        test_systems = local.loc[local["role"].eq("test"), "system_id"]
        if len(test_systems) != len(expected_systems):
            raise AssertionError(f"{protocol} test folds do not cover every system")
        if set(test_systems) != expected_systems:
            raise AssertionError(f"{protocol} test folds have wrong systems")
        if set(local["split_seed"]) != {seed}:
            raise AssertionError(f"{protocol} does not record seed {seed}")

    OUTPUT.mkdir(exist_ok=True)
    fold_path = OUTPUT / "phase2_fold_assignments.csv"
    assignments.to_csv(fold_path, index=False)
    summary = fold_summary(assignments)
    summary_path = OUTPUT / "phase2_fold_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "cv5_partition_seeds": list(PHASE2_CV5_PARTITION_SEEDS),
        "loso_seed": PHASE2_LOSO_SEED,
        "sensitivity_seed": PHASE2_SENSITIVITY_SEED,
        "rows": len(assignments),
        "systems": assignments["system_id"].nunique(),
        "protocols": sorted(assignments["protocol"].unique().tolist()),
        "fold_assignments_sha256": sha256_file(fold_path),
        "fold_summary_sha256": sha256_file(summary_path),
        "phase2_model_runs": 0,
    }
    (OUTPUT / "phase2_fold_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
