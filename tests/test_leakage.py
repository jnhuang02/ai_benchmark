"""Regression tests for system-level isolation and unknown-category training."""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from evaluation.data import (
    CATEGORICAL_COLUMNS,
    UNKNOWN,
    load_dataset,
    load_family_map,
    system_table,
)
from evaluation.evaluate import (
    GNNConfig,
    evaluate_fold_model,
    fold_role_sets,
)
from evaluation.features import (
    CategoryVocabulary,
    StrictTargetScaler,
    TabularFeatureEncoder,
    mask_identity_values,
    normalized_category,
)
from evaluation.folds import build_fold_specs, fold_assignment_table
from evaluation.graph import (
    GraphFeatures,
    graph_isolation_audit,
    make_graph,
)

warnings.filterwarnings(
    "ignore",
    message=".*type_params.*",
    category=DeprecationWarning,
)


class IsolationFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_dataset()
        cls.systems = system_table(cls.frame, load_family_map())
        cls.specs = build_fold_specs(cls.systems, seed=20261107)
        cls.assignments = fold_assignment_table(cls.systems, cls.specs)

    def frame_for(self, systems: set[str]) -> pd.DataFrame:
        return (
            self.frame.loc[self.frame["system_id"].isin(systems)]
            .copy()
            .reset_index(drop=True)
        )

    def isolated_training_graph(
        self, frame: pd.DataFrame, train_systems, validation_systems
    ):
        train = self.frame_for(set(train_systems))
        validation = self.frame_for(set(validation_systems))
        scaler = StrictTargetScaler.fit(train)
        supported = set(scaler.supported_groups)
        train_labeled = train.loc[train["target_group"].isin(supported)].copy()
        validation_labeled = validation.loc[
            validation["target_group"].isin(supported)
        ].copy()
        features = GraphFeatures.fit(train)
        rows = pd.concat([train_labeled, validation_labeled], ignore_index=True)
        graph = make_graph(
            rows,
            train_labeled,
            features,
            scaler,
            seed=7,
            unknown_mask_probability=0.1,
        )
        return graph, train_labeled, validation_labeled, scaler, features


class TestEveryFoldIsolation(IsolationFixture):
    def test_roles_are_disjoint_for_every_fold(self) -> None:
        for (protocol, fold_id), group in self.assignments.groupby(
            ["protocol", "fold_id"], sort=True
        ):
            roles = {
                role: set(group.loc[group["role"].eq(role), "system_id"])
                for role in ("train", "validation", "test")
            }
            with self.subTest(protocol=protocol, fold_id=fold_id):
                self.assertFalse(roles["train"] & roles["validation"])
                self.assertFalse(roles["train"] & roles["test"])
                self.assertFalse(roles["validation"] & roles["test"])

    def test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold(
        self,
    ) -> None:
        for (protocol, fold_id), group in self.assignments.groupby(
            ["protocol", "fold_id"], sort=True
        ):
            train_systems, validation_systems, test_systems = fold_role_sets(
                self.assignments, protocol, fold_id
            )
            graph, train_labeled, _, scaler, features = self.isolated_training_graph(
                self.frame, train_systems, validation_systems
            )
            vocabulary = features.vocabulary
            forbidden = set(test_systems)
            with self.subTest(protocol=protocol, fold_id=fold_id):
                graph_isolation_audit(graph, forbidden)
                self.assertFalse(
                    (forbidden & set(graph["result"].system_id))
                    | (forbidden & set(graph["system"].system_id))
                )
                train_frame = self.frame_for(set(train_systems))
                expected_center = train_frame.loc[
                    train_frame["target_group"].isin(scaler.supported_groups)
                ].groupby("target_group")["log_performance"].median()
                observed_center = scaler.center.loc[
                    list(scaler.supported_groups)
                ]
                np.testing.assert_allclose(
                    observed_center.to_numpy(), expected_center.to_numpy()
                )
                test = self.frame_for(forbidden)
                for column in CATEGORICAL_COLUMNS:
                    test_only = set(normalized_category(test[column])) - set(
                        normalized_category(self.frame_for(set(train_systems))[column])
                    )
                    test_only.discard(UNKNOWN)
                    self.assertFalse(test_only & set(vocabulary.categories[column]))
                self.assertFalse(
                    set(train_labeled["row_uid"]) & set(test["row_uid"])
                )

    def test_baseline_fit_never_receives_test_rows_for_every_fold(self) -> None:
        for (protocol, fold_id), _ in self.assignments.groupby(
            ["protocol", "fold_id"], sort=True
        ):
            result = evaluate_fold_model(
                self.frame,
                self.assignments,
                protocol=protocol,
                fold_id=fold_id,
                model="ridge",
                policy="isolated_supported",
                seed=7,
                config=GNNConfig(epochs=1, patience=1),
            )
            with self.subTest(protocol=protocol, fold_id=fold_id):
                self.assertFalse(
                    set(result.baseline_train_row_uids)
                    & set(result.baseline_test_row_uids)
                )
                self.assertFalse(
                    set(result.train_systems) & set(result.test_systems)
                )
                self.assertFalse(
                    set(result.validation_systems) & set(result.test_systems)
                )

    def test_unknown_vectors_are_exercised_during_training(self) -> None:
        train, _, _ = fold_role_sets(self.assignments, "cv5", "fold_0")
        train_frame = self.frame_for(set(train))
        masked = mask_identity_values(train_frame, seed=7, probability=0.1)
        vocabulary = CategoryVocabulary.fit(train_frame)
        encoder = TabularFeatureEncoder.fit(masked, vocabulary)
        matrix = encoder.transform(masked)
        offset = 0
        for column in CATEGORICAL_COLUMNS:
            width = len(vocabulary.categories[column])
            unknown_index = vocabulary.categories[column].index(UNKNOWN)
            self.assertGreater(
                np.asarray(matrix[:, offset + unknown_index]).sum(),
                0,
                f"No trained unknown activation for {column}",
            )
            offset += width


class TestWeightPerturbation(IsolationFixture):
    def perturbed_test_frame(self, test_systems: set[str]) -> pd.DataFrame:
        altered = self.frame.copy()
        mask = altered["system_id"].isin(test_systems)
        altered.loc[mask, "nodes"] = altered.loc[mask, "nodes"] + 7
        altered.loc[
            mask, "accelerators_per_node"
        ] = altered.loc[mask, "accelerators_per_node"] + 3
        altered.loc[mask, "precision"] = "scrambled-test-precision"
        altered.loc[mask, "result"] = altered.loc[mask, "result"] * 3.7
        altered.loc[mask, "log_performance"] = (
            altered.loc[mask, "direction"]
            * np.log(altered.loc[mask, "result"])
        )
        return altered

    def production_result(self, frame: pd.DataFrame, policy: str):
        return evaluate_fold_model(
            frame,
            self.assignments,
            protocol="cv5",
            fold_id="fold_0",
            model="graphsage",
            policy=policy,
            seed=7,
            config=GNNConfig(),
        )

    def test_production_weights_ignore_scrambled_test_metadata(self) -> None:
        _, _, test_systems = fold_role_sets(self.assignments, "cv5", "fold_0")
        first = self.production_result(self.frame, "isolated_supported")
        graph_isolation_audit(first.training_graph, set(test_systems))
        changed = self.perturbed_test_frame(set(test_systems))
        second = self.production_result(changed, "isolated_supported")
        self.assertEqual(first.weight_sha256, second.weight_sha256)
        self.assertTrue(first.trained_weights)
        with TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.pt"
            second_path = Path(directory) / "second.pt"
            torch.save(first.trained_weights, first_path)
            torch.save(second.trained_weights, second_path)
            first_saved = torch.load(
                first_path, map_location="cpu", weights_only=True
            )
            second_saved = torch.load(
                second_path, map_location="cpu", weights_only=True
            )
        self.assertEqual(first_saved.keys(), second_saved.keys())
        for key in first_saved:
            self.assertTrue(
                torch.equal(first_saved[key], second_saved[key]),
                f"Saved weight tensor changed for {key}",
            )

    def test_reintroduced_leak_fails_the_production_weight_check(self) -> None:
        _, _, test_systems = fold_role_sets(self.assignments, "cv5", "fold_0")
        first = self.production_result(self.frame, "legacy_leaky_global")
        with self.assertRaises(AssertionError):
            graph_isolation_audit(first.training_graph, set(test_systems))
        changed = self.perturbed_test_frame(set(test_systems))
        second = self.production_result(changed, "legacy_leaky_global")
        self.assertNotEqual(first.weight_sha256, second.weight_sha256)
        differing = [
            key
            for key in first.trained_weights
            if not torch.equal(
                first.trained_weights[key], second.trained_weights[key]
            )
        ]
        self.assertTrue(
            differing,
            "Reintroduced leak did not change any production weight tensor",
        )


class TestSupportedLeakyPolicy(IsolationFixture):
    def test_loss_masks_exclude_unsupported_rows_without_isolation(self) -> None:
        result = evaluate_fold_model(
            self.frame,
            self.assignments,
            protocol="cv5",
            fold_id="fold_0",
            model="graphsage",
            policy="audit_supported_leaky",
            seed=7,
            config=GNNConfig(epochs=2, patience=1),
        )
        self.assertFalse(result.predictions.empty)
        self.assertTrue(
            np.isfinite(result.predictions["prediction"].to_numpy()).all()
        )
        self.assertTrue(np.isfinite(result.trained_weights["head.bias"]).all())


class TestFiveFoldWeightBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_dataset()
        cls.assignments = pd.read_csv(
            "folds/phase2_fold_assignments.csv"
        )
        cls.perturbations = TestWeightPerturbation()
        cls.perturbations.frame = cls.frame

    def test_production_weights_ignore_test_metadata_on_five_fold_cases(
        self,
    ) -> None:
        cases = [
            ("cv5_partition_0", "p0_fold_0", "cv5, Edge-containing"),
            ("cv5_partition_0", "p0_fold_1", "cv5, second partition fold"),
            ("loso", "holdout_104", "LOSO, Edge system"),
            ("loso", "holdout_001", "LOSO, Datacenter system"),
            ("sensitivity", "fold_0", "sensitivity, Edge-containing"),
        ]
        for protocol, fold_id, label in cases:
            _, _, test_systems = fold_role_sets(
                self.assignments, protocol, fold_id
            )
            with self.subTest(label=label):
                first = evaluate_fold_model(
                    self.frame,
                    self.assignments,
                    protocol=protocol,
                    fold_id=fold_id,
                    model="graphsage",
                    policy="isolated_supported",
                    seed=7,
                    config=GNNConfig(),
                )
                graph_isolation_audit(
                    first.training_graph, set(test_systems)
                )
                changed = self.perturbations.perturbed_test_frame(
                    set(test_systems)
                )
                second = evaluate_fold_model(
                    changed,
                    self.assignments,
                    protocol=protocol,
                    fold_id=fold_id,
                    model="graphsage",
                    policy="isolated_supported",
                    seed=7,
                    config=GNNConfig(),
                )
                self.assertEqual(first.weight_sha256, second.weight_sha256)
                self.assertEqual(
                    first.trained_weights.keys(),
                    second.trained_weights.keys(),
                )
                for key in first.trained_weights:
                    self.assertTrue(
                        torch.equal(
                            first.trained_weights[key],
                            second.trained_weights[key],
                        ),
                        f"Saved weight tensor changed for {key} in {label}",
                    )


if __name__ == "__main__":
    unittest.main()
