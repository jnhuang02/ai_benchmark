"""Regression tests for graph node identity and row-order alignment."""

from __future__ import annotations

import unittest
import warnings

import pandas as pd

from evaluation.evaluate import fold_role_sets
from evaluation.features import StrictTargetScaler
from evaluation.graph import GraphFeatures, make_graph
from evaluation.phase2 import load_phase2_frame


warnings.filterwarnings(
    "ignore",
    message=".*type_params.*",
    category=DeprecationWarning,
)


class TestGraphIdentityAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_phase2_frame()
        cls.assignments = pd.read_csv("folds/phase2_fold_assignments.csv")

    def graph_for(self, protocol: str, fold_id: str):
        train_systems, _, _ = fold_role_sets(
            self.assignments, protocol, fold_id
        )
        train = self.frame.loc[
            self.frame["system_id"].isin(train_systems)
        ].reset_index(drop=True)
        graph = make_graph(
            self.frame,
            train,
            GraphFeatures.fit(train),
            StrictTargetScaler.fit(train),
            seed=7,
            unknown_mask_probability=0.1,
        )
        return graph

    def path_cases(self):
        return (
            ("cv5_partition_0", "p0_fold_0"),
            ("loso", "holdout_000"),
        )

    def assert_result_edges_match_rows(
        self,
        graph,
        *,
        source_node: str,
        relation: str,
        destination_node: str,
        row_column: str,
        node_id_attribute: str,
    ) -> None:
        edge_index = graph[
            source_node, relation, destination_node
        ].edge_index.cpu().numpy()
        source_ids = getattr(graph[source_node], node_id_attribute, None)
        destination_ids = getattr(
            graph[destination_node], node_id_attribute, None
        )
        if source_node == "result":
            result_indices = edge_index[0]
            linked_indices = edge_index[1]
            linked_ids = destination_ids
        else:
            self.assertEqual(destination_node, "result")
            result_indices = edge_index[1]
            linked_indices = edge_index[0]
            linked_ids = source_ids

        self.assertEqual(
            set(result_indices.tolist()),
            set(range(len(self.frame))),
            f"Not every result node has a {source_node}:{relation}:{destination_node} edge",
        )
        mismatches = []
        for result_index, linked_index in zip(
            result_indices, linked_indices
        ):
            row_uid = graph["result"].row_uid[result_index]
            source_row = self.frame.loc[
                self.frame["row_uid"].eq(row_uid)
            ].iloc[0]
            expected = source_row[row_column]
            actual = linked_ids[linked_index]
            if actual != expected:
                mismatches.append(
                    (
                        result_index,
                        row_uid,
                        expected,
                        actual,
                    )
                )
        self.assertFalse(
            mismatches,
            f"{len(mismatches)} identity mismatches for "
            f"{source_node}:{relation}:{destination_node}; first ten: "
            f"{mismatches[:10]}",
        )

    def test_result_nodes_preserve_input_row_order(self) -> None:
        for protocol, fold_id in self.path_cases():
            graph = self.graph_for(protocol, fold_id)
            with self.subTest(protocol=protocol, fold_id=fold_id):
                self.assertEqual(
                    graph["result"].row_uid,
                    self.frame["row_uid"].tolist(),
                )

    def test_non_result_node_ids_match_sorted_metadata(self) -> None:
        expected = {
            "system": sorted(self.frame["system_id"].unique()),
            "accelerator": sorted(self.frame["accelerator_id"].unique()),
            "processor": sorted(self.frame["processor_id"].unique()),
            "workload": sorted(self.frame["target_group"].unique()),
        }
        attributes = {
            "system": "system_id",
            "accelerator": "accelerator_id",
            "processor": "processor_id",
            "workload": "target_group",
        }
        for protocol, fold_id in self.path_cases():
            graph = self.graph_for(protocol, fold_id)
            with self.subTest(protocol=protocol, fold_id=fold_id):
                for node_type, values in expected.items():
                    self.assertEqual(
                        getattr(graph[node_type], attributes[node_type]),
                        values,
                    )

    def test_non_result_edges_match_row_metadata(self) -> None:
        cases = (
            (
                "system",
                "uses_accelerator",
                "accelerator",
                "system_id",
                "accelerator_id",
            ),
            (
                "system",
                "uses_processor",
                "processor",
                "system_id",
                "processor_id",
            ),
            (
                "system",
                "runs_workload",
                "workload",
                "system_id",
                "target_group",
            ),
        )
        for protocol, fold_id in self.path_cases():
            graph = self.graph_for(protocol, fold_id)
            for source, relation, destination, left_column, right_column in cases:
                edge_index = graph[
                    source, relation, destination
                ].edge_index.cpu().numpy()
                source_ids = getattr(
                    graph[source],
                    {
                        "system": "system_id",
                        "accelerator": "accelerator_id",
                        "processor": "processor_id",
                        "workload": "target_group",
                    }[source],
                )
                destination_ids = getattr(
                    graph[destination],
                    {
                        "system": "system_id",
                        "accelerator": "accelerator_id",
                        "processor": "processor_id",
                        "workload": "target_group",
                    }[destination],
                )
                actual = {
                    (source_ids[left], destination_ids[right])
                    for left, right in zip(*edge_index)
                }
                expected = set(
                    zip(self.frame[left_column], self.frame[right_column])
                )
                with self.subTest(
                    protocol=protocol,
                    fold_id=fold_id,
                    relation=relation,
                ):
                    self.assertEqual(actual, expected)

    def test_result_to_metadata_edges_match_rows_for_cv5(self) -> None:
        graph = self.graph_for("cv5_partition_0", "p0_fold_0")
        for case in self.result_relations():
            with self.subTest(relation=case["relation"]):
                self.assert_result_edges_match_rows(graph, **case)

    def test_result_to_metadata_edges_match_rows_for_loso(self) -> None:
        graph = self.graph_for("loso", "holdout_000")
        for case in self.result_relations():
            with self.subTest(relation=case["relation"]):
                self.assert_result_edges_match_rows(graph, **case)

    @staticmethod
    def result_relations():
        return (
            dict(
                source_node="result",
                relation="of_system",
                destination_node="system",
                row_column="system_id",
                node_id_attribute="system_id",
            ),
            dict(
                source_node="system",
                relation="has_result",
                destination_node="result",
                row_column="system_id",
                node_id_attribute="system_id",
            ),
            dict(
                source_node="result",
                relation="uses_accelerator",
                destination_node="accelerator",
                row_column="accelerator_id",
                node_id_attribute="accelerator_id",
            ),
            dict(
                source_node="accelerator",
                relation="has_result",
                destination_node="result",
                row_column="accelerator_id",
                node_id_attribute="accelerator_id",
            ),
            dict(
                source_node="result",
                relation="uses_processor",
                destination_node="processor",
                row_column="processor_id",
                node_id_attribute="processor_id",
            ),
            dict(
                source_node="processor",
                relation="has_result",
                destination_node="result",
                row_column="processor_id",
                node_id_attribute="processor_id",
            ),
            dict(
                source_node="result",
                relation="measures",
                destination_node="workload",
                row_column="target_group",
                node_id_attribute="target_group",
            ),
            dict(
                source_node="workload",
                relation="has_result",
                destination_node="result",
                row_column="target_group",
                node_id_attribute="target_group",
            ),
        )


if __name__ == "__main__":
    unittest.main()
