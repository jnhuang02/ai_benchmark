# Figure data provenance

These CSVs are the compact, reviewable data layer for the thesis figures.

- `thesis_scope.csv`, `workload_counts.csv`, and `edge_groups.csv` reproduce
  the MLPerf Inference v6.1 Closed-division coverage reported in the current
  thesis. The canonical source snapshot contains 457 Datacenter rows and 19
  Edge rows, for 476 measurements across 96 independent systems.
- `associations.csv`, `metadata_baseline.csv`,
  `headline_comparison.csv`, `workload_performance.csv`,
  `filtered_diagnostics.csv`, `selection_balance.csv`, and
  `error_concentration.csv` transcribe values from the current Overleaf
  thesis at commit `536f2de` (2 November 2026).

The figures intentionally do not pool latency and throughput into one metric.
The text-to-video row describes raw-latency association, so a negative
coefficient means lower latency. All other association rows describe
throughput, where higher is better.

These are model-development results derived from published MLPerf
measurements. They are not official MLPerf results, accuracy-compliance
evidence, or hardware-profiler measurements.
