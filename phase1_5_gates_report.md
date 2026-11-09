# Phase 1.5 Acceptance-Gate Report

Gate evidence bundle:
`mlperf_gnn/results/phase1_5-gates-f82e4d2/`.

The bundle was produced at clean code commit `f82e4d2`, has 11 verified output
hashes, and records `phase2_model_runs: 0`.

## Gate 1: Spearman reconciliation

### Exact definitions

Phase 1 metric, which produced the 0.4956 aggregation:

- group size: at least 2 rows;
- singleton groups are not calculated and therefore cannot enter the median;
- at least 2 distinct measured values and 2 distinct predictions are required;
- the group value is the Pearson correlation of average ranks;
- the seed-level value is the median over all finite qualifying group
  correlations;
- the headline is the mean of the seed-level values across seeds 7-16.

Original Table 6.4 helper:

- group size: at least 3 rows;
- singleton and two-row groups are excluded;
- at least 2 distinct measured values and 2 distinct predictions are required;
- the group value is the Pearson correlation of average ranks;
- the value is the median over all finite qualifying group correlations;
- Table 6.4 used seed 7.

### Reconciliation on identical predictions

| Predictions | Definition | Seeds | Mean | SD | Qualifying groups |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase 1.5 pinned-fold baseline | at least 2 rows | 7-16 | 0.495561 | 0.020163 | 26 |
| Phase 1.5 pinned-fold baseline | at least 3 rows | 7-16 | 0.484192 | 0.027474 | 25 |
| Exact Phase 0 source replay | at least 2 rows | 7 only | 0.735455 | NA | 26 |
| Exact Phase 0 source replay | at least 3 rows | 7 only | 0.711303 | NA | 25 |
| Phase 1 fixed isolation | at least 2 rows | 7-16 | 0.528660 | 0.025100 | 20 |
| Phase 1 fixed isolation | at least 3 rows | 7-16 | 0.528660 | 0.025100 | 20 |
| Phase 1 legacy leaky graph | at least 2 rows | 7-16 | 0.466085 | 0.073741 | 26 |
| Phase 1 legacy leaky graph | at least 3 rows | 7-16 | 0.434447 | 0.086580 | 25 |

Seed-7 values from the Phase 1.5 baseline replay:

- at least 2 rows: **0.4825910931** over 26 groups;
- at least 3 rows: **0.4651821862** over 25 groups.

The single group affected by the definition is the Edge Whisper Offline
Samples/s group, which has 2 held-out rows. Its two-row rank correlation is
approximately `+1.0`; the minimum-three-row definition excludes it.

### Why 0.4956 differs from 0.711

`0.4956` is not an exact seed-7 replay. It is the mean across seeds 7-16 from
the Phase 1.5 production adaptation with folds pinned and the model initialized
with the same seed in each fold. It used the minimum-two-row definition.

`0.711303` is the exact historical Phase 0 GraphSAGE source run at seed 7,
where fold construction and per-fold model initialization followed the
historical seed-coupled procedure. It used the minimum-three-row definition.

On the exact historical predictions, changing only the definition moves
Spearman from `0.735455` to `0.711303`. On the ten-seed baseline predictions,
changing only the definition moves the mean from `0.495561` to `0.484192`.
Therefore the definition contributes to the discrepancy, but the bulk of the
gap is the different prediction runs and seed aggregation, not the Spearman
definition alone.

### Correct definition for the thesis

Use the original Table 6.4 definition: **groups with at least three evaluated
rows, at least two distinct measured values, and at least two distinct
predictions enter the median; singleton and two-row groups do not enter.**

This avoids an unstable `+/-1` two-row correlation determining part of the
headline. Protocol v2 and the metric code now use this definition.

## Gate 2: Protocol v2

Version 1:

- commit `d3bc835`;
- SHA-256
  `eb14e445f34c83ee817f59b35555f086be402d031534d4415815c575579fe811`.

Version 2:

- final protocol commit `cce2f43`;
- new SHA-256
  `9062c13cfe2ae50e9a5da2a0e92f48f8d2d2ef32191cd0d38942ac13b40296f4`;
- `protocol.sha256` matches this value;
- amendment log: `mlperf_gnn/protocol_history.md`;
- full V1-to-V2 diff: `mlperf_gnn/protocol_v1_to_v2.diff`, SHA-256
  `020d25087f2f6b097a96ec69318dd0926694c40fe723d1c50289722a08904a5b`.

The V2 changes include:

- grouped five-fold repeated over five partitions is co-primary with LOSO;
- partition seeds are `20261107`, `20261108`, `20261109`, `20261110`, and
  `20261111`;
- the sole primary hypothesis is best GNN minus MLP under both co-primary
  protocols;
- equivalence margin is `+/-0.05` log units;
- result labels are better, worse, equivalent, or inconclusive under the
  pre-specified protocol frozen by commit hash before the Phase 2 model runs;
- partition and seed variability is averaged inside each system-clustered
  bootstrap draw;
- secondary comparisons receive Holm correction across the eight
  protocol-comparison tests;
- pooled MAPE is appendix-only;
- all models use `log1p(total_accelerators)` plus a CPU-only indicator;
- no tuning is used for any model;
- supported rows come from each fold's training partition and are identical
  across models;
- corrected median-group Spearman requires at least three rows;
- the sensitivity split is explicitly labeled modest because 78 of 90 groups
  are singleton systems;
- LOAFO remains approval-gated, with families below three systems reported
  separately.

The fold artifact from the pre-specified protocol frozen by commit hash before
the Phase 2 model runs is
`mlperf_gnn/folds/phase2_fold_assignments.csv`:

- 14,175 assignment rows;
- 105 systems;
- five CV partitions, LOSO, and sensitivity, with no LOAFO run;
- partition seeds recorded in `partition_seed` and `split_seed` columns;
- fold SHA-256
  `3ed4dd0bf019685255c6f6d11f309d10f147280487617cf8f49c7c437e433443`;
- manifest records `phase2_model_runs: 0`.

## Gate 3: Quick confirmations

### Loss-mask bug boundary

| Commit | Policy | Predictions finite | Primary error |
| --- | --- | --- | ---: |
| `d3bc835`, before fix | `legacy_leaky_global` | Yes | 0.368149 |
| `d3bc835`, before fix | `audit_supported_leaky` | **No** | NA |
| `d3bc835`, before fix | `isolated_supported` | Yes | 0.510997 |
| `f82e4d2`, after fix | `legacy_leaky_global` | Yes | 0.368149 |
| `f82e4d2`, after fix | `audit_supported_leaky` | Yes | 0.463906 |
| `f82e4d2`, after fix | `isolated_supported` | Yes | 0.510997 |

Only the audit-only all-row graph with strict supported labels was affected.
Its train mask included unsupported rows carrying NaN normalized targets.
Global fallback made every row labeled, and fixed isolation already restricted
the mask, so those paths were finite.

Fixed-isolation Phase 1 seed-7 comparison after the fix:

| Statistic | Saved Phase 1 | Current rerun |
| --- | ---: | ---: |
| Evaluated rows | 450 | 450 |
| Systems | 99 | 99 |
| Primary error | 0.5109971488879594 | 0.5109971488879586 |
| Median-group Spearman | 0.5127216323343433 | 0.5127216323343433 |
| Maximum raw prediction difference | NA | 1.1641532182693481e-10 |
| Maximum normalized prediction difference | NA | 4.9560546777627223e-08 |

The fixed-isolation result is unchanged at reported precision; the sub-`1e-9`
difference is floating-point nondeterminism.

### Leakage mutation result

```text
leak_reintroduced=yes
isolation_guard_failed_as_expected=True
weight_hash_changed=True
differing_weight_tensors=56
production_weight_comparison_folds=1
production_epochs=150
production_patience=20
structural_fold_checks=130
```

The production saved-weight comparison covers one representative CV fold
(`fold_0`) with four full production fits across the isolated/mutated and
leak/mutated tests. The former weight-hash loop covered 130 folds with
`epochs=1, patience=1`; that loop has been replaced. All 130 folds still
receive structural isolation, scaler, vocabulary, and row-boundary checks.

### Scalar accelerator counts for the two multi-count systems

| System | Row | Public ID | Workload / scenario | Nodes | Accel/node | `total_accelerators` |
| --- | --- | --- | --- | ---: | ---: | ---: |
| HPE XD690 | `row:0009` | `6.1-0048` | deepseek-r1 / Server | 2 | 8 | 16 |
| HPE XD690 | `row:0081` | `6.1-0049` | gpt-oss-120b / Server | 1 | 8 | 8 |
| HPE XD690 | `row:0318` | `6.1-0049` | gpt-oss-120b / Offline | 1 | 8 | 8 |
| HPE XD690 | `row:0435` | `6.1-0048` | deepseek-r1 / Offline | 2 | 8 | 16 |
| GB300 NVL72 | `row:0010` | `6.1-0074` | VLM / Server | 18 | 4 | 72 |
| GB300 NVL72 | `row:0042` | `6.1-0073` | deepseek-r1 / Server | 72 | 4 | 288 |
| GB300 NVL72 | `row:0050` | `6.1-0074` | gpt-oss-120b / Server | 18 | 4 | 72 |
| GB300 NVL72 | `row:0106` | `6.1-0074` | text-to-video / SingleStream | 18 | 4 | 72 |
| GB300 NVL72 | `row:0126` | `6.1-0074` | gpt-oss-120b / Offline | 18 | 4 | 72 |
| GB300 NVL72 | `row:0157` | `6.1-0074` | deepseek-r1 / Interactive | 18 | 4 | 72 |
| GB300 NVL72 | `row:0234` | `6.1-0074` | VLM / Offline | 18 | 4 | 72 |
| GB300 NVL72 | `row:0241` | `6.1-0074` | deepseek-r1 / Offline | 18 | 4 | 72 |
| GB300 NVL72 | `row:0316` | `6.1-0074` | text-to-video / Offline | 18 | 4 | 72 |
| GB300 NVL72 | `row:0343` | `6.1-0073` | deepseek-r1 / Offline | 72 | 4 | 288 |
| GB300 NVL72 | `row:0349` | `6.1-0074` | deepseek-r1 / Server | 18 | 4 | 72 |
| GB300 NVL72 | `row:0392` | `6.1-0074` | VLM / Interactive | 18 | 4 | 72 |
| GB300 NVL72 | `row:0416` | `6.1-0074` | gpt-oss-120b / Interactive | 18 | 4 | 72 |

Each row uses the exported `Total Accelerators` value for that measurement,
equal to `nodes * accelerators_per_node`. HPE therefore uses `8` or `16` per
row; GB300 uses `72` or `288`. The tuples `8;16` and `72;288` are audit
grouping signatures, not row features and not values to sum across unrelated
measurements.

## LOAFO proposal, not run

Family rules:

- CPU-only is one execution-regime family with 10 systems and is holdout
  eligible;
- GB10 has 2 systems and VR200 has 2 systems, so each is reported separately;
- Jetson Thor and Radeon 8060S have 1 system each and are reported separately;
- the two mixed-accelerator systems are always-train.

Full `loafo_family_proposal.csv` content:

```csv
system_id,suite,submitter,system_name,accelerator,processor,rows,proposed_vendor,proposed_architecture_generation,proposed_family_id,holdout_disposition,rationale
edge|Atlas_Inference|AMD Strix Halo desktop (Ryzen AI Max+ 395 / Radeon 8060S)|edge,edge,Atlas_Inference,AMD Strix Halo desktop (Ryzen AI Max+ 395 / Radeon 8060S),AMD Radeon 8060S Graphics,AMD Ryzen AI Max+ 395,1,AMD,RDNA 3.5 integrated Radeon 8060S,amd_rdna35_integrated_radeon_8060s,holdout_eligible,Radeon 8060S in the Strix Halo system is an integrated RDNA-generation device.
datacenter|Cisco|X410M8_1_node_4S_GNR_86C|datacenter,datacenter,Cisco,X410M8_1_node_4S_GNR_86C,,Intel(R) Xeon(R) 6788P,3,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Dell|PowerEdge R770AP_128C 24x96GB (6400 MT/s)|datacenter,datacenter,Dell,PowerEdge R770AP_128C 24x96GB (6400 MT/s),,Intel(R) Xeon(R) 6980P,2,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Dell|PowerEdge R770_86C 16x64GB (6400 MT/s)|datacenter,datacenter,Dell,PowerEdge R770_86C 16x64GB (6400 MT/s),,Intel(R) Xeon(R) 6787P,4,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Intel|1-node-2S-GNR_128C|datacenter,datacenter,Intel,1-node-2S-GNR_128C,,Intel(R) Xeon(R) 6980P,4,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Oracle|BM.Standard4.Ax.120|datacenter,datacenter,Oracle,BM.Standard4.Ax.120,,Intel(R) Xeon(R) 6987P-C,3,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Quanta_Cloud_Technology|1-node-2S-GNR_86C|datacenter,datacenter,Quanta_Cloud_Technology,1-node-2S-GNR_86C,,Intel(R) Xeon(R) 6788P,4,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|RedHat_Intel|1-node-2S-GNR_96C|datacenter,datacenter,RedHat_Intel,1-node-2S-GNR_96C,,Intel(R) Xeon(R) 6972P,3,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Supermicro|1-node-2S-GNR_128C|datacenter,datacenter,Supermicro,1-node-2S-GNR_128C,,Intel(R) Xeon(R) 6980P,3,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Supermicro|1-node-2S-GNR_86C|datacenter,datacenter,Supermicro,1-node-2S-GNR_86C,,Intel(R) Xeon(R) 6787P,4,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
datacenter|Supermicro|1-node-4S-GNR_86C|datacenter,datacenter,Supermicro,1-node-4S-GNR_86C,,Intel(R) Xeon(R) 6788P,3,Intel,Xeon 6 Granite Rapids,cpu_only_intel_xeon6_granite_rapids,holdout_eligible,No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime.
edge|Atlas_Inference|NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)|edge,edge,Atlas_Inference,NVIDIA DGX Spark (GB10 Grace Blackwell Superchip),NVIDIA GB10,NVIDIA Grace (20-core Arm v9.2),1,NVIDIA,Blackwell integrated GB10,nvidia_blackwell_integrated_gb10,holdout_eligible,"GB10 is an integrated Grace-Blackwell device, distinct from discrete B-series accelerators."
edge|Dell|Dell Pro Max (1x GB10)|edge,edge,Dell,Dell Pro Max (1x GB10),NVIDIA GB10,Cortex-X925,6,NVIDIA,Blackwell integrated GB10,nvidia_blackwell_integrated_gb10,holdout_eligible,"GB10 is an integrated Grace-Blackwell device, distinct from discrete B-series accelerators."
edge|NVIDIA|NVIDIA Jetson AGX Thor Developer Kit 128G (TensorRT)|edge,edge,NVIDIA,NVIDIA Jetson AGX Thor Developer Kit 128G (TensorRT),NVIDIA Jetson AGX Thor 128G,14-core ARM Poseidon-AE CPU,1,NVIDIA,Blackwell integrated Jetson Thor,nvidia_blackwell_integrated_jetson_thor,holdout_eligible,Jetson AGX Thor is an integrated Blackwell-generation edge device.
datacenter|NVIDIA|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)|datacenter,datacenter,NVIDIA,NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64),NVIDIA VR200,NVIDIA Vera CPU,6,NVIDIA,Rubin VR200,nvidia_rubin_vr200,holdout_eligible,The reported VR200 accelerator denotes the Rubin-generation product family.
datacenter|Nebius|Nebius VR200 NVL72 (36x VR200-288GB_aarch64 TensorRT)|datacenter,datacenter,Nebius,Nebius VR200 NVL72 (36x VR200-288GB_aarch64 TensorRT),NVIDIA VR200,NVIDIA Vera CPU,2,NVIDIA,Rubin VR200,nvidia_rubin_vr200,holdout_eligible,The reported VR200 accelerator denotes the Rubin-generation product family.
datacenter|Dell_MangoBoost|Heterogeneous Four-Cluster (2x8x MI300X + 2x8x MI355X LLMBoost)|datacenter,datacenter,Dell_MangoBoost,Heterogeneous Four-Cluster (2x8x MI300X + 2x8x MI355X LLMBoost),AMD Instinct MI300X 192GB HBM3 (x2x8) and AMD Instinct MI355X 288GB HBM3e (x2x8),AMD EPYC 9534 (on MI300X node) and AMD EPYC 9965 (on MI355X node),2,AMD,CDNA3 MI300X + CDNA4 MI355X,mixed_amd_cdna3_mi300x_cdna4_mi355x,always_train,Two AMD Instinct architecture generations are reported by one heterogeneous cluster.
datacenter|Cisco|Cisco UCS H200 x8 + MI350X x8 G200 2x2 TOR|datacenter,datacenter,Cisco,Cisco UCS H200 x8 + MI350X x8 G200 2x2 TOR,NVIDIA H200-SXM-141GB + AMD Instinct MI350X 288GB HBM3e,AMD EPYC,8,NVIDIA + AMD,Hopper H200 + CDNA4 MI350X,mixed_nvidia_hopper_h200_amd_cdna4_mi350x,always_train,Two discrete accelerator architectures are reported by the same system.
```

## Gate decision

| Gate | Status |
| --- | --- |
| Spearman reconciliation | Complete; minimum-three-row definition adopted |
| Protocol v2, hash, amendment, and diff | Complete; no Phase 2 results exist |
| Loss-mask scope and fixed-isolation invariance | Complete |
| Leak mutation and runtime coverage | Complete |
| Multi-count scalar features | Complete |
| LOAFO proposal | Complete; not run |

Phase 1.5 gates are ready for orchestrator review. No Phase 2 model run has
started.
