"""Integration checks for the pre-registered Phase 2 model ladder."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.evaluate import GNNConfig
from evaluation.phase2 import (
    DETERMINISTIC_METHODS,
    NEURAL_METHODS,
    fit_deterministic_fold,
    fit_neural_models,
    fit_phase2_graph_model,
    load_phase2_frame,
)


class TestPhase2Fold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_phase2_frame()
        cls.assignments = pd.read_csv("folds/phase2_fold_assignments.csv")
        cls.protocol = "cv5_partition_0"
        cls.fold_id = "p0_fold_0"

    def test_deterministic_ladder_is_finite(self) -> None:
        predictions = fit_deterministic_fold(
            self.frame,
            self.assignments,
            protocol=self.protocol,
            fold_id=self.fold_id,
        )
        self.assertEqual(
            set(predictions["method"]), set(DETERMINISTIC_METHODS)
        )
        self.assertTrue(np.isfinite(predictions["prediction_log"]).all())

    def test_deterministic_subset_omits_unrequested_models(self) -> None:
        predictions = fit_deterministic_fold(
            self.frame,
            self.assignments,
            protocol=self.protocol,
            fold_id=self.fold_id,
            methods=("group_median", "ridge", "random_forest"),
        )
        self.assertEqual(
            set(predictions["method"]),
            {"group_median", "ridge", "random_forest"},
        )

    def test_neural_ladder_and_best_gnn_are_finite(self) -> None:
        predictions = fit_neural_models(
            self.frame,
            self.assignments,
            protocol=self.protocol,
            fold_id=self.fold_id,
            seed=7,
            config=GNNConfig(epochs=2, patience=1),
        )
        expected = set(NEURAL_METHODS) | {"best_gnn"}
        self.assertEqual(set(predictions["method"]), expected)
        self.assertTrue(np.isfinite(predictions["prediction_log"]).all())
        selected = predictions.loc[
            predictions["method"].eq("best_gnn"), "selected_gnn"
        ]
        self.assertEqual(len(selected), predictions["row_uid"].nunique())
        self.assertTrue(
            set(selected).issubset({"graphsage", "gcn", "gat"})
        )

    def test_core_neural_path_omits_secondary_graph_variants(self) -> None:
        predictions = fit_neural_models(
            self.frame,
            self.assignments,
            protocol=self.protocol,
            fold_id=self.fold_id,
            seed=7,
            config=GNNConfig(epochs=2, patience=1),
            include_secondary_graph_variants=False,
        )
        self.assertEqual(
            set(predictions["method"]),
            {"mlp", "graphsage", "gcn", "gat", "best_gnn"},
        )

    def test_graph_diagnostic_reports_training_stop_state(self) -> None:
        _, diagnostics = fit_phase2_graph_model(
            self.frame,
            self.assignments,
            protocol=self.protocol,
            fold_id=self.fold_id,
            model_name="gat",
            seed=7,
            config=GNNConfig(epochs=3, patience=1),
        )
        self.assertGreaterEqual(diagnostics["best_epoch"], 1)
        self.assertGreaterEqual(diagnostics["epochs_run"], 1)
        self.assertTrue(np.isfinite(diagnostics["final_train_loss"]))
        self.assertGreater(diagnostics["prediction_variance"], 0)


if __name__ == "__main__":
    unittest.main()
