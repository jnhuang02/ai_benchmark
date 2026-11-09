"""Tests for system identity, sensitivity groups, and feature representation."""

from __future__ import annotations

import unittest

from evaluation.data import load_dataset, load_family_map, system_table


class TestSystemTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_dataset()
        cls.systems = system_table(cls.frame, load_family_map())

    def test_sensitivity_key_excludes_accelerator_counts(self) -> None:
        self.assertEqual(self.systems["system_id"].nunique(), 105)
        self.assertLess(
            self.systems["sensitivity_group_id"].nunique(),
            self.systems["system_id"].nunique(),
        )
        count_signatures = self.systems.groupby("sensitivity_group_id")[
            "accelerator_count_values"
        ].nunique()
        self.assertTrue(count_signatures.gt(1).any())

    def test_multiple_count_signatures_are_retained_for_grouping(self) -> None:
        repeated = self.systems.loc[
            self.systems["accelerator_count_values"].str.contains(";")
        ]
        self.assertEqual(len(repeated), 2)
        self.assertEqual(
            set(repeated["accelerator_count_values"]),
            {"8;16", "72;288"},
        )

    def test_total_accelerators_remain_the_numeric_feature(self) -> None:
        self.assertIn("total_accelerators", self.frame.columns)
        self.assertFalse(
            self.frame["total_accelerators"]
            .astype(str)
            .str.contains(";")
            .any()
        )


if __name__ == "__main__":
    unittest.main()
