from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "thesis_figures"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


class FigureDataTests(unittest.TestCase):
    def test_scope_matches_thesis(self) -> None:
        scope = read("thesis_scope.csv").set_index("metric")["value"]
        self.assertEqual(int(scope["datacenter_result_cells"]), 457)
        self.assertEqual(int(scope["edge_result_cells"]), 19)
        self.assertEqual(int(scope["combined_systems"]), 96)
        self.assertEqual(int(scope["combined_workload_groups"]), 31)

    def test_workload_counts_reconcile(self) -> None:
        counts = read("workload_counts.csv")
        self.assertEqual(int(counts["datacenter_cells"].sum()), 457)
        self.assertEqual(int(counts["edge_cells"].sum()), 19)
        self.assertEqual(int(counts["total_cells"].sum()), 476)

    def test_headline_metrics_match_current_thesis(self) -> None:
        metrics = read("headline_comparison.csv").set_index(
            ["method", "metric"]
        )["value"]
        self.assertEqual(metrics.loc[("GraphSAGE", "pooled_mape_pct")], 460.7)
        self.assertEqual(
            metrics.loc[("GraphSAGE", "median_group_mape_pct")], 38.8
        )
        self.assertEqual(
            metrics.loc[("GraphSAGE", "median_row_mape_pct")], 27.4
        )
        self.assertEqual(
            metrics.loc[("GraphSAGE", "median_group_spearman")], 0.711
        )
        self.assertEqual(
            metrics.loc[("Random forest", "pooled_mape_pct")], 557.0
        )

    def test_diagnostic_tables_are_internally_consistent(self) -> None:
        concentration = read("error_concentration.csv")
        balance = read("selection_balance.csv")
        edge_groups = read("edge_groups.csv")
        diagnostics = read("filtered_diagnostics.csv")

        self.assertEqual(concentration["error_share_pct"].sum(), 100.0)
        self.assertEqual(concentration["groups"].sum(), 31)
        self.assertEqual(balance["total_groups"].nunique(), 1)
        self.assertEqual(balance["total_groups"].iat[0], 20)
        self.assertEqual(int(edge_groups["rows"].sum()), 19)
        self.assertEqual(int((edge_groups["rows"] < 3).sum()), 4)
        self.assertEqual(
            set(diagnostics["method"]), {"GraphSAGE", "Random forest"}
        )


if __name__ == "__main__":
    unittest.main()
