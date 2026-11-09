"""Tests for metric aggregation and paired system-cluster bootstrap."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.metrics import (
    calculate_metrics,
    clustered_bootstrap,
    group_metrics,
    paired_bootstrap_differences,
)


class TestClusteredBootstrap(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for system in range(8):
            for method, multiplier in [("model_a", 1.0), ("model_b", 1.15)]:
                for group in ["group_0", "group_1"]:
                    result = float(10 + 2 * system + (group == "group_1"))
                    prediction = result * multiplier
                    rows.append(
                        {
                            "system_id": f"system_{system}",
                            "method": method,
                            "target_group": group,
                            "result": result,
                            "prediction": prediction,
                            "prediction_log": np.log(prediction),
                            "y": np.log(result),
                        }
                    )
        self.predictions = pd.DataFrame(rows)

    def test_paired_difference_uses_same_system_draw(self) -> None:
        summary, draws = clustered_bootstrap(
            self.predictions, replicates=100, seed=7
        )
        self.assertEqual(set(summary["method"]), {"model_a", "model_b"})
        paired = paired_bootstrap_differences(
            draws, left="model_a", right="model_b"
        )
        row = paired.iloc[0]
        self.assertLess(row["mean_difference"], 0)
        self.assertEqual(row["bootstrap_replicates"], 100)
        self.assertTrue(row["difference_95ci_low"] <= row["difference_95ci_high"])


class TestMedianGroupSpearman(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for group, size, direction in [
            ("one", 1, 1),
            ("two", 2, 1),
            ("three", 3, 1),
            ("four", 4, 1),
        ]:
            for index in range(size):
                result = float(index + 1)
                prediction = direction * result
                rows.append(
                    {
                        "system_id": f"{group}_{index}",
                        "method": "model",
                        "target_group": group,
                        "result": result,
                        "prediction": prediction,
                        "prediction_log": np.log(prediction),
                        "y": np.log(result),
                    }
                )
        self.predictions = pd.DataFrame(rows)

    def test_two_row_groups_are_excluded_by_default(self) -> None:
        one_bad_pair = self.predictions.copy()
        two_group = one_bad_pair["target_group"].eq("two")
        one_bad_pair.loc[two_group, "prediction"] = [2.0, 1.0]
        with_two_rows = group_metrics(
            one_bad_pair, spearman_minimum_rows=2
        ).set_index("target_group")
        default = group_metrics(one_bad_pair).set_index("target_group")
        self.assertAlmostEqual(
            with_two_rows.loc["two", "spearman"], -1.0, places=12
        )
        self.assertTrue(np.isnan(default.loc["two", "spearman"]))
        self.assertEqual(
            calculate_metrics(one_bad_pair)["median_group_spearman"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
