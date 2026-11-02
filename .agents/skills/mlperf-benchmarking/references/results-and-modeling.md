# Results Analysis and Graph Modeling

## Local Result Snapshots

The inspected `mlcommons_results_latest` directory contains:

- Inference Datacenter Closed v6.1: 457 rows, 98 submission IDs, 87 distinct
  system names; required result, accelerator, processor, and software fields
  were populated.
- Inference Edge Closed v6.1: 19 rows, 10 submissions, 9 distinct systems.
- Mobile v6.0: seven rows.
- Tiny v1.4: closed/open workbooks.

The new long-format Datacenter and Edge CSVs are the source of truth for this
snapshot. They share no Public IDs. Older wide workbooks are a different
export/snapshot and should not be joined by row order or assumed to match by ID.

Edge and Datacenter rows contain performance results but not a general accuracy
field. Treat absent accuracy as missing evidence, not passing accuracy.

## Exploratory Associations

Spearman association between total accelerator count and log performance was:

| Workload/scenario | Association |
| --- | ---: |
| VLM Server | +0.95 |
| DeepSeek-R1 Offline | +0.90 |
| Llama 3.1 8B Offline | +0.80 |
| GPT-OSS-120B Offline | +0.79 |
| Llama 2 70B Interactive | -0.27 |
| Text-to-Video SingleStream, latency | -0.81 |

These are exploratory, task-specific associations. Different units, system
families, batching strategies, precision, networking, and software stacks mean
they do not establish that adding accelerators causes proportional speedup.

## Metadata-Only Baseline

Leave-one-system-out results using metadata features produced:

| Model | Pooled MAPE | Median fold MAPE |
| --- | ---: | ---: |
| Ridge | 63.7% | 11.3% |
| k-NN | 58.8% | 6.6% |

The pooled errors are inflated by a small set of extreme values and heterogeneous
units. Performance was stronger for VLM, DeepSeek, and Llama 3.1 and weaker for
Llama 2 and GPT-OSS. Preserve both pooled and fold-level error; a strong median
does not excuse large failures.

## Modeling Order

Start with:

1. per-task/per-scenario median;
2. Ridge or Elastic Net;
3. Random Forest;
4. XGBoost or another boosted-tree baseline;
5. a multi-task MLP;
6. heterogeneous GraphSAGE;
7. GCN and GAT/GATv2 ablations;
8. a roofline-informed model only after hardware and workload features exist.

Normalize targets within task/scenario/unit. Use log throughput or inverse/log
latency so that higher remains better. Evaluate with leave-one-system-out and,
where possible, leave-one-accelerator-family-out splits. Report MAPE, RMSE,
\(R^2\), and rank correlation, plus confidence intervals.

## GraphSAGE versus “GNN”

GraphSAGE is an inductive graph-neural-network architecture. “GNN” is the model
family. Use heterogeneous GraphSAGE as the primary candidate when new systems
need predictions from metadata and neighboring accelerator/task relationships.
Use GCN as a transductive baseline and GAT when attention over specific
neighbors is a testable hypothesis.

A minimal graph can contain `System`, `Accelerator`, `Task`, and `Scenario`
nodes. Edges express containment, execution, or an observed measurement.
Two GraphSAGE layers with 64–128 hidden units, dropout around 0.2–0.3, and
relation-specific encoders is a reasonable starting point. Tune only after a
simple tabular baseline works.

Topological ordering is useful for an acyclic operator/dataflow graph: it gives
a valid execution order and exposes critical paths. Do not force a system-result
relationship graph into a topological order if it contains cycles. Use graph
joins and typed edges there instead.

## Limits of Extrapolation

Given hardware metadata, a trained graph model can emit a numeric estimate for a
system absent from the training table. That estimate is an extrapolation, not
validation. A withheld measured result is required to claim predictive accuracy.
Without ground truth for the target system, clearly label the output as
hypothetical and report distribution-shift risk.

## Roofline Requirements

Published result tables alone are insufficient for a roofline plot. A defensible
roofline requires:

- accelerator peak compute for the actual precision;
- memory bandwidth and relevant cache/interconnect limits;
- workload FLOPs and bytes moved for the implemented model;
- batching, sequence length, sparsity, and concurrency;
- achieved performance mapped to the same workload and hardware configuration.

In the inspected trace set, 63 `mlperf_log_trace.json` files were found: 10 were
non-empty and 53 were empty. They contained no utilization counters, hardware
memory-bandwidth measurements, interconnect traffic, or controlled intervention.
Therefore causal bottleneck claims require external profiler telemetry or an
experiment that changes one factor.

Useful figures are predicted versus measured performance, residuals by
accelerator family, aggregate model comparison, GraphSAGE/GCN/GAT ablation, and
uncertainty by task. Keep the final discussion tied to the exact rows and
versions that support it.
