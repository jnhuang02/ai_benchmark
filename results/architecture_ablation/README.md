# GraphSAGE, GCN, and GAT comparison

This directory contains the five-fold architecture ablation for MLPerf
Inference v6.1 Datacenter and Edge Closed-division results. These are
model-development results, not official MLPerf submissions.

## Protocol

- Input: 476 measurements from 96 systems across 31 target groups.
- Validation: five folds with complete systems held out.
- Seed: 7, shared across all models.
- Graph: heterogeneous system, accelerator, processor, workload, and result
  nodes with 14 typed relations.
- Models: two layers, hidden width 64, dropout 0.2, Huber training loss,
  150 maximum epochs, and early stopping with patience 20.
- GAT: four attention heads with concatenation disabled and a learned
  destination self-term in every relation-specific layer.
- Baselines: target-group median, Ridge regression, and random forest on the
  same folds.

Reproduce the run:

```bash
arch -arm64 .venv/bin/python -m ai_benchmark.train_graphsage \
  --suite all --folds 5 \
  --models graphsage gcn gat \
  --out-dir results/architecture_ablation
```

## Results

MAPE values are percentages of the measured value. RMSE is computed on the
normalized signed-log target and is not a percentage.

| Method | Pooled MAPE | Median group MAPE | Median row MAPE | Norm. RMSE | Spearman rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | 460.7% | 38.8% | 27.4% | 1.455 | 0.711 |
| GAT | 762.2% | 37.2% | 27.4% | 1.444 | 0.661 |
| GCN | 450265.6% | 35.9% | 26.6% | 1.480 | 0.749 |
| Random forest | 557.0% | 67.8% | 23.5% | 1.502 | 0.732 |
| Group median | 25056.6% | 77.6% | 63.0% | 1.828 | -0.369 |
| Ridge | 8410495117756716.0% | 60.3% | 50.6% | 1.903 | 0.666 |

For rows in workload groups with at least five measurements:

| Method | Mean MAPE | Median MAPE | Within 50% |
| --- | ---: | ---: | ---: |
| GraphSAGE | 59.5% | 26.6% | 73.3% |
| GAT | 71.4% | 26.2% | 73.6% |
| GCN | 74.6% | 25.6% | 74.9% |
| Random forest | 164.3% | 20.6% | 71.4% |

## Interpretation

GCN has the best median workload-group error and median within-group rank
correlation. GraphSAGE has the lowest pooled error among the GNNs and the best
filtered mean error. GAT has the lowest normalized RMSE but does not improve
the median row error over GraphSAGE and has weaker rank correlation.

GAT has a large pooled error driven by the singleton end-to-end
question-answering group. GCN has even larger catastrophic errors in the
singleton yolo-99 and end-to-end groups, and in the three-row DLRM-v3 groups.
GraphSAGE is the most stable choice when pooled error and sensitivity to sparse
groups are emphasized.

The singleton groups are too small to support stable within-group learning.
They must be reported separately and should not be used to claim that one
architecture dominates overall.

## Files

- `all_architecture_comparison_summary.csv`: aggregate metrics for all GNNs
  and baselines.
- `all_architecture_comparison_groups.csv`: workload-level error and rank
  correlation.
- `all_architecture_comparison_predictions.csv`: row-level held-out
  predictions.
- `full_run.log`: complete training console output.
