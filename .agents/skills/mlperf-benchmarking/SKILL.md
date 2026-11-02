---
name: mlperf-benchmarking
description: "Analyze, run, and interpret MLPerf Mobile, Tiny, Edge, and Datacenter benchmarks; use for benchmark selection, device setup, scenario and metric interpretation, result comparison, or graph and roofline performance modeling. Do not use for generic machine-learning benchmarks unrelated to MLPerf."
metadata:
  short-description: "Run and analyze MLPerf benchmarks"
---

# MLPerf Benchmarking

Use the local benchmark repository and the matching results round as the source
of truth. Do not assume that a workload, scenario, unit, or result count from a
different MLPerf round still applies.

## Establish Scope

Before answering or running anything, identify:

- suite: Mobile, Tiny, Inference Edge, or Inference Datacenter;
- release/version and division, such as `v6.1` and `closed`;
- workload, model, scenario, and reported unit;
- target device or system, including the relevant hardware and software stack;
- whether the goal is a development run, a valid submission run, or analysis of
  published results only.

Do not compare latency, throughput, and accuracy as if they were one metric.
Normalize throughput and latency per task and record whether higher is better.

## Working Method

1. Inspect the local suite documentation, task README, result metadata, and
   existing logs before proposing commands.
2. Trace every claim to a versioned result table, benchmark configuration, or
   reproducible run. Distinguish an observation from a causal explanation.
3. Use quick runs only to verify integration. Never label them as official
   MLPerf results.
4. Preserve the required dataset, model, precision, compiler, runtime, device,
   build, and power conditions.
5. For modeling, establish simple baselines before a GNN and validate with
   grouped splits. Report missing labels and extrapolation explicitly.

## Core Distinctions

- Closed division constrains the model and accuracy and normally requires
  compliance testing. Open division permits more implementation freedom but
  still follows its current rules.
- A physical iPhone and an iOS Simulator are different systems. A meaningful
  Mobile result requires the physical device named in the submission.
- MobileNetV4 is a workload/model. Core ML, CPU, GPU, and Apple Neural Engine
  are execution backends or compute units.
- Tiny commonly reports latency and energy alongside accuracy; Inference
  Edge/Datacenter typically report throughput or latency for a scenario.
- MLPerf result exports are not hardware profilers. LoadGen traces do not
  provide utilization, memory-bandwidth, or interconnect counters by
  themselves.
- GraphSAGE is a GNN. It is inductive and usually the primary candidate for a
  system graph; GCN and GAT are useful ablations, not implementation synonyms.

## References

- Read [references/benchmarks-and-metrics.md](references/benchmarks-and-metrics.md)
  for the observed suite catalog, scenarios, divisions, and metrics.
- Read [references/execution-and-devices.md](references/execution-and-devices.md)
  for server, iOS, and Tiny execution guidance and troubleshooting.
- Read [references/results-and-modeling.md](references/results-and-modeling.md)
  for the analyzed result snapshots, baselines, GraphSAGE design, validation,
  trace limitations, and roofline requirements.

Load only the reference needed for the current task. For result analysis,
retain exact version and unit labels in outputs; for a new benchmark run, report
the commands and environment instead of relying on remembered setup.
