# Thesis figure insertion guide

The matching `PNG` files are convenient previews. Use the `PDF` files in
LaTeX so the labels and lines remain sharp.

| Figure | Suggested placement | Main reading |
| --- | --- | --- |
| `fig01_data_scope.pdf` | Data Coverage and Evaluation Scope | Workload support is highly uneven; the largest four workloads contain 345 of 476 rows. |
| `fig02_accelerator_associations.pdf` | Exploratory Associations | Several throughput workloads show strong positive rank associations; text-to-video is shown in raw-latency direction. |
| `fig03_metadata_baselines.pdf` | Metadata-Only Reference Analysis | Metadata-only Ridge and k-NN errors vary substantially by workload. |
| `fig04_model_comparison.pdf` | GraphSAGE Validation Results | Neither GraphSAGE nor random forest dominates every metric. |
| `fig05_workload_diagnostics.pdf` | GraphSAGE Validation Results | Absolute error and within-workload rank ordering provide different views of model quality. |
| `fig06_sparse_group_diagnostics.pdf` | Pooled error is dominated by sparse groups | Five singleton groups contribute 85.7% of summed error; filtered results show a model tradeoff. |
| `fig07_edge_evidence.pdf` | The Edge sample is not sufficient for stable inference | Edge groups are too sparse for a strong standalone generalization claim. |
| `fig08_evaluation_pipeline.pdf` | Reproducible Evaluation Pipeline | End-to-end flow from published exports to system-held-out model evaluation. |

Example LaTeX:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\textwidth]{figures/fig04_model_comparison.pdf}
  \caption{System-held-out development results for GraphSAGE and random
  forest.  Neither model dominates every metric.  MAPE is reported on the
  original measurement scale, whereas RMSE is computed on the normalized
  logarithmic target.}
  \label{fig:graphsage-comparison}
\end{figure}
```

For the full-width five-panel comparison, use `figure*` if the thesis template
requires a page-wide float.

Figure captions should retain the exact source scope, metric direction, and
the distinction between predictive modeling and official MLPerf results.
