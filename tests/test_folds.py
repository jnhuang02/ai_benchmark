"""Tests for the pre-registered Phase 2 repeated fold structure."""

from __future__ import annotations

import unittest

from evaluation.data import load_dataset, load_family_map, system_table
from evaluation.folds import (
    PHASE2_CV5_PARTITION_SEEDS,
    build_phase2_fold_specs,
    fold_assignment_table,
)


class TestPhase2Folds(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.systems = system_table(load_dataset(), load_family_map())
        cls.assignments = fold_assignment_table(
            cls.systems, build_phase2_fold_specs(cls.systems)
        )

    def test_five_repeated_partitions_cover_every_system(self) -> None:
        expected = set(self.systems["system_id"])
        for index, seed in enumerate(PHASE2_CV5_PARTITION_SEEDS):
            protocol = f"cv5_partition_{index}"
            local = self.assignments.loc[
                self.assignments["protocol"].eq(protocol)
            ]
            self.assertEqual(local["fold_id"].nunique(), 5)
            tested = local.loc[local["role"].eq("test"), "system_id"]
            self.assertEqual(set(tested), expected)
            self.assertEqual(seed, PHASE2_CV5_PARTITION_SEEDS[index])

    def test_partitions_are_distinct(self) -> None:
        signatures = set()
        for index in range(len(PHASE2_CV5_PARTITION_SEEDS)):
            protocol = f"cv5_partition_{index}"
            local = self.assignments.loc[
                self.assignments["protocol"].eq(protocol)
                & self.assignments["role"].eq("test")
            ].sort_values("fold_id")
            signature = tuple(
                tuple(group.sort_values("system_id")["system_id"])
                for _, group in local.groupby("fold_id", sort=True)
            )
            signatures.add(signature)
        self.assertEqual(len(signatures), len(PHASE2_CV5_PARTITION_SEEDS))

    def test_loso_and_sensitivity_are_present(self) -> None:
        self.assertEqual(
            self.assignments.loc[
                self.assignments["protocol"].eq("loso"), "fold_id"
            ].nunique(),
            105,
        )
        self.assertEqual(
            self.assignments.loc[
                self.assignments["protocol"].eq("sensitivity"), "fold_id"
            ].nunique(),
            5,
        )


if __name__ == "__main__":
    unittest.main()
