from pathlib import Path

import numpy as np

from ai_benchmark import prepare_data
from pandas import DataFrame

from ai_benchmark.train_graphsage import HeteroGraphSAGE, make_graph, metrics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"


def test_source_files_are_distinct_and_complete():
    integrity = prepare_data.source_integrity(DATA)

    assert len(integrity) == 2
    assert integrity["sha256"].nunique() == 2
    assert set(integrity["suite"]) == {"datacenter", "edge"}


def test_canonical_dataset_contract():
    frame = prepare_data.build_dataset(DATA)
    counts = frame.groupby("suite").size().to_dict()

    assert len(frame) == 476
    assert frame["system_id"].nunique() == 96
    assert counts == {"datacenter": 457, "edge": 19}
    assert frame["result"].gt(0).all()
    assert np.isfinite(frame["log_performance"]).all()
    assert frame["target_group"].nunique() == 31


def test_graph_contains_all_supported_node_types():
    frame = prepare_data.build_dataset(DATA)
    train_rows = frame.loc[
        frame["suite"].eq("datacenter") & frame["division"].eq("closed")
    ]
    graph = make_graph(frame, train_rows)

    assert set(graph.node_types) == {
        "system",
        "accelerator",
        "processor",
        "workload",
        "result",
    }
    assert len(graph.edge_types) == 14
    assert graph["result"].num_nodes == len(frame)


def test_graphsage_and_ablation_models_construct():
    assert HeteroGraphSAGE(64, 2, "graphsage").head.out_features == 1
    assert HeteroGraphSAGE(64, 2, "gcn").head.out_features == 1
    assert HeteroGraphSAGE(64, 2, "gat").head.out_features == 1


def test_metrics_report_median_row_error():
    evaluation = DataFrame(
        {
            "result": [10.0, 20.0, 40.0],
            "prediction": [12.0, 10.0, 80.0],
            "target_group": ["a", "a", "b"],
            "y": [0.0, 1.0, 2.0],
        }
    )
    result = metrics(evaluation, evaluation["y"].to_numpy())

    assert result["median_row_mape"] == 0.5
