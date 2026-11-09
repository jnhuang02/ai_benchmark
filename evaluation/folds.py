"""Canonical, reusable system-grouped fold assignments."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


PHASE2_CV5_PARTITION_SEEDS = (
    20261107,
    20261108,
    20261109,
    20261110,
    20261111,
)
PHASE2_LOSO_SEED = 20261107
PHASE2_SENSITIVITY_SEED = PHASE2_LOSO_SEED + 4000


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    heldout_group: str
    test_systems: tuple[str, ...]
    validation_systems: tuple[str, ...]


def _validation_systems(
    systems: np.ndarray, seed: int, fraction: float = 0.12
) -> tuple[str, ...]:
    if len(systems) < 2:
        return ()
    count = max(1, round(fraction * len(systems)))
    count = min(count, len(systems) - 1)
    rng = np.random.default_rng(seed)
    return tuple(sorted(rng.choice(systems, count, replace=False).tolist()))


def _suite_stratified_chunks(
    systems: pd.DataFrame, folds: int, seed: int
) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    chunks: list[list[str]] = [[] for _ in range(folds)]
    for _, suite_systems in systems.groupby("suite", sort=True):
        identifiers = np.array(sorted(suite_systems["system_id"].tolist()))
        rng.shuffle(identifiers)
        for offset, system_id in enumerate(identifiers):
            chunks[offset % folds].append(str(system_id))
    return [sorted(chunk) for chunk in chunks]


def _grouped_chunks(
    systems: pd.DataFrame,
    group_column: str,
    folds: int,
    seed: int,
) -> list[list[str]]:
    groups = (
        systems[["system_id", group_column]]
        .drop_duplicates()
        .sort_values(group_column)
    )
    if groups["system_id"].duplicated().any():
        raise ValueError(f"Systems cross groups in {group_column}")
    labels = groups[group_column].drop_duplicates().to_numpy()
    if len(labels) < folds:
        raise ValueError(f"Only {len(labels)} groups for {folds} folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    assignment = {label: index % folds for index, label in enumerate(labels)}
    return [
        sorted(
            systems.loc[
                systems[group_column].map(assignment).eq(fold_index), "system_id"
            ].tolist()
        )
        for fold_index in range(folds)
    ]


def five_fold_specs(
    systems: pd.DataFrame, seed: int
) -> list[FoldSpec]:
    all_systems = np.array(sorted(systems["system_id"].tolist()))
    five_chunks = _suite_stratified_chunks(systems, folds=5, seed=seed)
    five_specs = []
    for fold, test_systems in enumerate(five_chunks):
        outer_train = np.array(
            [item for item in all_systems if item not in set(test_systems)]
        )
        validation = _validation_systems(outer_train, seed + 1000 + fold)
        five_specs.append(
            FoldSpec(
                fold_id=f"fold_{fold}",
                heldout_group="random_system_group",
                test_systems=tuple(test_systems),
                validation_systems=validation,
            )
        )
    return five_specs


def build_fold_specs(systems: pd.DataFrame, seed: int) -> dict[str, list[FoldSpec]]:
    all_systems = np.array(sorted(systems["system_id"].tolist()))
    specs: dict[str, list[FoldSpec]] = {}

    five_specs = five_fold_specs(systems, seed)
    specs["cv5"] = five_specs

    loso_specs = []
    for fold, test_system in enumerate(all_systems):
        outer_train = all_systems[all_systems != test_system]
        loso_specs.append(
            FoldSpec(
                fold_id=f"holdout_{fold:03d}",
                heldout_group=str(test_system),
                test_systems=(str(test_system),),
                validation_systems=_validation_systems(
                    outer_train, seed + 2000 + fold
                ),
            )
        )
    specs["loso"] = loso_specs

    loafo_specs = []
    for fold, (family, group) in enumerate(
        systems.groupby("accelerator_family", sort=True)
    ):
        test_systems = tuple(sorted(group["system_id"].tolist()))
        outer_train = np.array(
            [item for item in all_systems if item not in set(test_systems)]
        )
        loafo_specs.append(
            FoldSpec(
                fold_id=f"family_{fold:02d}",
                heldout_group=str(family),
                test_systems=test_systems,
                validation_systems=_validation_systems(
                    outer_train, seed + 3000 + fold
                ),
            )
        )
    specs["loafo"] = loafo_specs

    sensitivity_chunks = _grouped_chunks(
        systems, "sensitivity_group_id", folds=5, seed=seed + 4000
    )
    sensitivity_specs = []
    for fold, test_systems in enumerate(sensitivity_chunks):
        outer_train = np.array(
            [item for item in all_systems if item not in set(test_systems)]
        )
        sensitivity_specs.append(
            FoldSpec(
                fold_id=f"fold_{fold}",
                heldout_group="submitter_accelerator_processor",
                test_systems=tuple(test_systems),
                validation_systems=_validation_systems(
                    outer_train, seed + 5000 + fold
                ),
            )
        )
    specs["sensitivity"] = sensitivity_specs
    return specs


def build_phase2_fold_specs(
    systems: pd.DataFrame,
) -> dict[str, list[FoldSpec]]:
    specs: dict[str, list[FoldSpec]] = {}
    for partition_index, seed in enumerate(PHASE2_CV5_PARTITION_SEEDS):
        protocol = f"cv5_partition_{partition_index}"
        specs[protocol] = [
            replace(spec, fold_id=f"p{partition_index}_{spec.fold_id}")
            for spec in five_fold_specs(systems, seed)
        ]
    shared = build_fold_specs(systems, seed=PHASE2_LOSO_SEED)
    specs["loso"] = shared["loso"]
    specs["sensitivity"] = shared["sensitivity"]
    return specs


def fold_assignment_table(
    systems: pd.DataFrame, specs: dict[str, list[FoldSpec]]
) -> pd.DataFrame:
    metadata = systems.set_index("system_id")
    rows = []
    for protocol, protocol_specs in specs.items():
        for spec in protocol_specs:
            test = set(spec.test_systems)
            validation = set(spec.validation_systems)
            if test & validation:
                raise ValueError(f"{protocol}/{spec.fold_id} has test/validation overlap")
            for system_id in sorted(metadata.index):
                role = (
                    "test"
                    if system_id in test
                    else "validation"
                    if system_id in validation
                    else "train"
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "fold_id": spec.fold_id,
                        "heldout_group": spec.heldout_group,
                        "system_id": system_id,
                        "role": role,
                        "suite": metadata.loc[system_id, "suite"],
                        "accelerator_family": metadata.loc[
                            system_id, "accelerator_family"
                        ],
                        "sensitivity_group_id": metadata.loc[
                            system_id, "sensitivity_group_id"
                        ],
                    }
                )
    result = pd.DataFrame(rows)
    if set(result["role"]) != {"train", "validation", "test"}:
        raise ValueError("Fold table must contain all three roles")
    return result


def systems_for_role(
    assignments: pd.DataFrame, protocol: str, fold_id: str, role: str
) -> list[str]:
    subset = assignments.loc[
        assignments["protocol"].eq(protocol)
        & assignments["fold_id"].eq(fold_id)
        & assignments["role"].eq(role),
        "system_id",
    ]
    return sorted(subset.tolist())


def fold_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    summary = (
        assignments.groupby(["protocol", "fold_id", "role"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    heldout = (
        assignments.loc[assignments["role"].eq("test")]
        .groupby(["protocol", "fold_id"], as_index=False)
        .agg(heldout_group=("heldout_group", "first"))
    )
    return summary.merge(heldout, on=["protocol", "fold_id"], how="left")
