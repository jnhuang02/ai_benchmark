"""Tests for the opt-in accelerator memory feature set."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from evaluation.data import (
    DEFAULT_MEMORY_TABLE,
    attach_memory_features,
    load_dataset,
    load_memory_table,
)


ROOT = Path(__file__).resolve().parents[1]
KNOWN_TYPES = {
    "HBM2e", "HBM3", "HBM3e", "HBM4", "GDDR6", "GDDR7", "LPDDR5X", "mixed", "none", "",
}


class TestMemoryTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = load_memory_table()
        cls.frame = load_dataset()

    def test_one_row_per_system_and_full_coverage(self) -> None:
        self.assertEqual(len(self.table), 105)
        self.assertEqual(set(self.table["system_id"]), set(self.frame["system_id"]))

    def test_memory_types_are_normalized(self) -> None:
        self.assertTrue(set(self.table["accelerator_memory_type"]) <= KNOWN_TYPES)

    def test_cpu_only_systems_have_zero_memory_and_none_type(self) -> None:
        cpu_systems = set(
            self.frame.loc[self.frame["total_accelerators"].eq(0), "system_id"]
        )
        cpu_rows = self.table.loc[self.table["system_id"].isin(cpu_systems)]
        self.assertTrue(cpu_rows["accelerator_memory_gb"].eq(0).all())
        self.assertTrue(cpu_rows["accelerator_memory_type"].eq("none").all())

    def test_accelerated_capacities_are_plausible(self) -> None:
        accelerated = self.table.loc[
            self.table["accelerator_memory_type"].ne("none")
        ]["accelerator_memory_gb"].dropna()
        self.assertTrue(accelerated.between(8, 512).all())

    def test_one_accelerator_name_has_one_memory_type(self) -> None:
        named = self.table.loc[
            self.table["accelerator"].ne("")
            & ~self.table["accelerator_memory_type"].isin(["", "mixed"])
        ]
        self.assertTrue(
            named.groupby("accelerator")["accelerator_memory_type"].nunique().eq(1).all()
        )

    def test_attach_keeps_rows_and_order(self) -> None:
        base = self.frame.drop(
            columns=["accelerator_memory_gb", "accelerator_memory_type"],
            errors="ignore",
        )
        attached = attach_memory_features(base, DEFAULT_MEMORY_TABLE)
        self.assertEqual(len(attached), len(base))
        self.assertListEqual(attached["row_uid"].tolist(), base["row_uid"].tolist())


def _columns_under(feature_set: str) -> str:
    env = {**os.environ, "AI_BENCHMARK_FEATURE_SET": feature_set}
    code = (
        "from evaluation.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS;"
        "print(','.join(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS))"
    )
    return subprocess.check_output(
        [sys.executable, "-c", code], cwd=ROOT, env=env, text=True
    ).strip()


class TestFeatureSetSwitch(unittest.TestCase):
    def test_frozen_feature_set_excludes_memory(self) -> None:
        columns = _columns_under("frozen").split(",")
        self.assertNotIn("accelerator_memory_gb", columns)
        self.assertNotIn("accelerator_memory_type", columns)

    def test_memory_feature_set_appends_memory_columns(self) -> None:
        frozen = _columns_under("frozen").split(",")
        memory = _columns_under("memory").split(",")
        self.assertIn("accelerator_memory_gb", memory)
        self.assertIn("accelerator_memory_type", memory)
        self.assertEqual(len(memory), len(frozen) + 2)

    def test_unknown_feature_set_is_rejected(self) -> None:
        env = {**os.environ, "AI_BENCHMARK_FEATURE_SET": "typo"}
        result = subprocess.run(
            [sys.executable, "-c", "import evaluation.data"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
