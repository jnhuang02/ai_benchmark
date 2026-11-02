# Datacenter + Edge GraphSAGE run

This directory records the five-fold, complete-system-held-out development
run used in the thesis. It is reproducibility evidence, not an official
MLPerf result.

- `all_graphsage_summary.csv`: aggregate comparison of target-group median,
  Ridge regression, random forest, and heterogeneous GraphSAGE.
- `all_graphsage_groups.csv`: per-workload MAPE and Spearman rank correlation.
- `all_graphsage_predictions.csv`: held-out measured values and predictions.

The pooled MAPE is strongly affected by five singleton workload groups. Use
the per-workload report and the stated exclusions in the thesis when
interpreting the aggregate values.

The committed five-fold run reports:

| Method | Pooled MAPE | Median group MAPE | Median row MAPE | Normalized RMSE | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | 460.7% | 38.8% | 27.4% | 1.455 | 0.711 |
| Random forest | 557.0% | 67.8% | 23.5% | 1.502 | 0.732 |

Regenerate with:

```bash
ai-benchmark-prepare
ai-benchmark-train --suite all --folds 5 --model graphsage
```

Runs are deterministic for the recorded environment. The notebook and CLI
also produce byte-identical CSVs on repeated executions with the pinned
kernel.
