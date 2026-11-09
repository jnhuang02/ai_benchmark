# Phase 1.5 Report

Canonical bundle:
`mlperf_gnn/results/phase1_5-final-8fbe960/`.

Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.

## 1. What was done

- Replayed the exact Phase 0 source and data from commit `292ff7f`.
- Ran a 10-seed cumulative GraphSAGE attribution, changing one pipeline
  choice at a time from the original settings:
  original, submitter-aware key, canonical folds, unknown masking, supported
  exclusion, and fixed isolation.
- Corrected the sensitivity grouping key to
  `submitter|accelerator model|processor`.
- Replaced the one-epoch weight checks with production training, saved-state
  comparison, and a reintroduced-leak mutation check.
- Checked provenance between the Phase 1 manifest commit `5c9dfde` and HEAD
  at Phase 1 reporting `fb36169`.
- Proposed 18 LOAFO review families. No LOAFO evaluation was run.
- Froze the Phase 2 protocol before any Phase 2 run.

The attribution run used seeds 7-16. Each fold construction was pinned once and
reused across model seeds; the exact Phase 0 replay separately retained the
historical seed-coupled folds.

## 2. Files changed

Code repository, branch `phase-1-evaluation`:

| Commit | Change |
| --- | --- |
| `67fd85c` | Removed accelerator counts from the sensitivity key; added data-contract tests. |
| `4788b55` | Strengthened leakage training and mutation tests. |
| `d3bc835` | Added the Phase 1.5 runner and froze `protocol.md`. |
| `8fbe960` | Restricted supported audit loss masks and added its regression test. |

Added:

- `mlperf_gnn/run_phase1_5.py`
- `mlperf_gnn/protocol.md`
- `mlperf_gnn/protocol.sha256`
- `mlperf_gnn/tests/test_data.py`
- `mlperf_gnn/results/phase1_5-final-8fbe960/`
- This report.

The Phase 2 protocol commit is `d3bc835`. `protocol.md` SHA-256 is
`eb14e445f34c83ee817f59b35555f086be402d031534d4415815c575579fe811`.
It records: no tuning for any model, seeds 7-16 for neural models, the fixed
model list, grouped five-fold primary protocol, LOSO robustness, corrected
sensitivity check, approval-gated LOAFO, paired comparisons, and the common
supported-row set.

No thesis text, `main.tex`, `main.pdf`, or `references.bib` was edited.

## 3. Numbers that changed

### Attribution

Primary error is the per-seed median
`|ln(prediction / measured)|` over evaluated rows; mean and SD are then taken
across seeds 7-16. This is the exact aggregation used in the table.

| Step | Change from previous step | Primary error mean +/- SD | Median-group Spearman mean +/- SD | Coverage |
| --- | --- | ---: | ---: | ---: |
| Baseline | Original settings | 0.421375 +/- 0.027125 | 0.495561 +/- 0.020163 | 100.00% |
| a | Submitter-aware key | 0.414332 +/- 0.034083 | 0.577141 +/- 0.047681 | 100.00% |
| b | Canonical five-fold split | 0.470127 +/- 0.029587 | 0.493378 +/- 0.073014 | 100.00% |
| c | Unknown masking p=0.1 | 0.485124 +/- 0.041135 | 0.466085 +/- 0.073741 | 100.00% |
| d | Unsupported-group exclusion | 0.468672 +/- 0.049192 | 0.553623 +/- 0.050666 | 94.54% |
| e | Fixed train/inference isolation | 0.462314 +/- 0.034193 | 0.528660 +/- 0.025100 | 94.54% |

The evaluated row count is 476 through step c and 450 from step d onward. The
26 excluded rows are 5.46% of the dataset.

The source is
`attribution_summary.csv`; seed-level values and predictions are saved in the
same bundle.

### Exact Phase 0 regression

| Model | Primary error | Median-group MAPE | Median-row MAPE | Median-group Spearman | Edge mean APE | Singleton APE share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | 0.289413 | 38.7772% | 27.4132% | 0.711303 | 6,289.5298% | 46.2140% |
| GAT, one head | 0.244249 | 29.7565% | 23.1571% | 0.820376 | 800.3811% | 95.2324% |

The four requested Phase 0 diagnostics reproduce exactly:

| Diagnostic | Expected | Observed | Source model |
| --- | ---: | ---: | --- |
| GAT median-group MAPE | 29.8% | 29.7565% | GAT |
| GAT median-row MAPE | 23.2% | 23.1571% | GAT |
| Edge mean APE | 6,289.5% | 6,289.5298% | GraphSAGE |
| Singleton APE share | 46.2% | 46.2140% | GraphSAGE |

The Phase 1 failure to reproduce those diagnostics was caused by comparing a
Phase 0 result with the new key, new folds, trained unknown masking, strict
support policy, and changed GAT head count rather than with the exact original
run. The diagnostics were also not all produced by the same model.

### Sensitivity grouping

Old grouping: 101 groups for 105 systems. New grouping: 90 groups.

| Suite | Systems | Measurements | Groups |
| --- | ---: | ---: | ---: |
| Datacenter | 96 | 457 | 83 |
| Edge | 9 | 19 | 7 |
| Total | 105 | 476 | 90 |

| Systems per group | Number of groups |
| ---: | ---: |
| 1 | 78 |
| 2 | 9 |
| 3 | 3 |

### Leakage test

The full 11-test suite ran in 9.243 seconds inside the canonical run; total test
process wall time was 11.917 seconds. The former 130-fold weight loop was fast
because it used `epochs=1, patience=1`. The strengthened test uses the
production 150-epoch budget with patience 20, saves and reloads the actual
state dictionaries, and verifies that reintroducing the all-row graph changes
the learned weights and triggers the isolation guard.

### Provenance

`git diff --stat 5c9dfde fb36169` over run-affecting code, data, tests, and
configuration reports no differences. The intervening commit changed only
`CHANGELOG.md` and result-report metadata. No Phase 1 bundle regeneration was
required for provenance.

### Feature contract

Each result row uses its scalar `total_accelerators` value as the numeric
feature. The sorted reported-count tuple is retained only for cumulative
grouping/audit and is excluded from the sensitivity key.

| System | Reported count signature |
| --- | --- |
| `datacenter\|HPE\|HPE Compute XD690 (8x NVIDIA B300-SXM-270GB TensorRT)\|datacenter` | `8;16` |
| `datacenter\|NVIDIA\|NVIDIA GB300 NVL72 (72x GB300-288GB_aarch64 TensorRT)\|datacenter` | `72;288` |

### Proposed LOAFO families

This table is a proposal only. It was not executed.

| System | Accelerator string | Vendor / architecture generation | Proposed family | Disposition |
| --- | --- | --- | --- | --- |
| `datacenter\|Cisco\|X410M8_1_node_4S_GNR_86C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Dell\|PowerEdge R770AP_128C 24x96GB (6400 MT/s)\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Dell\|PowerEdge R770_86C 16x64GB (6400 MT/s)\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Intel\|1-node-2S-GNR_128C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Oracle\|BM.Standard4.Ax.120\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Quanta_Cloud_Technology\|1-node-2S-GNR_86C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|RedHat_Intel\|1-node-2S-GNR_96C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Supermicro\|1-node-2S-GNR_128C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Supermicro\|1-node-2S-GNR_86C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `datacenter\|Supermicro\|1-node-4S-GNR_86C\|datacenter` | blank | Intel / Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout eligible |
| `edge\|Atlas_Inference\|NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)\|edge` | `NVIDIA GB10` | NVIDIA / Blackwell integrated GB10 | `nvidia_blackwell_integrated_gb10` | holdout eligible |
| `edge\|Dell\|Dell Pro Max (1x GB10)\|edge` | `NVIDIA GB10` | NVIDIA / Blackwell integrated GB10 | `nvidia_blackwell_integrated_gb10` | holdout eligible |
| `edge\|NVIDIA\|NVIDIA Jetson AGX Thor Developer Kit 128G (TensorRT)\|edge` | `NVIDIA Jetson AGX Thor 128G` | NVIDIA / Blackwell integrated Jetson Thor | `nvidia_blackwell_integrated_jetson_thor` | holdout eligible |
| `edge\|Atlas_Inference\|AMD Strix Halo desktop (Ryzen AI Max+ 395 / Radeon 8060S)\|edge` | `AMD Radeon 8060S Graphics` | AMD / RDNA 3.5 integrated Radeon 8060S | `amd_rdna35_integrated_radeon_8060s` | holdout eligible |
| `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `NVIDIA VR200` | NVIDIA / Rubin VR200 | `nvidia_rubin_vr200` | holdout eligible |
| `datacenter\|Nebius\|Nebius VR200 NVL72 (36x VR200-288GB_aarch64 TensorRT)\|datacenter` | `NVIDIA VR200` | NVIDIA / Rubin VR200 | `nvidia_rubin_vr200` | holdout eligible |
| `datacenter\|Dell_MangoBoost\|Heterogeneous Four-Cluster (2x8x MI300X + 2x8x MI355X LLMBoost)\|datacenter` | `AMD Instinct MI300X 192GB HBM3 (x2x8) and AMD Instinct MI355X 288GB HBM3e (x2x8)` | AMD / CDNA3 MI300X + CDNA4 MI355X | `mixed_amd_cdna3_mi300x_cdna4_mi355x` | always train |
| `datacenter\|Cisco\|Cisco UCS H200 x8 + MI350X x8 G200 2x2 TOR\|datacenter` | `NVIDIA H200-SXM-141GB + AMD Instinct MI350X 288GB HBM3e` | NVIDIA + AMD / Hopper H200 + CDNA4 MI350X | `mixed_nvidia_hopper_h200_amd_cdna4_mi350x` | always train |

The CPU-only systems all report Intel Xeon 6 Granite Rapids processors, so they
are proposed as one CPU execution-regime family. GB10 and Thor remain distinct
integrated-device families rather than being merged into discrete Blackwell
B-series accelerators. The two mixed systems are never held out.

## 4. Anything surprising or contradicting the current narrative

- The Phase 0 numbers are reproducible, but the four requested diagnostics
  come from two different models: GAT supplies the 29.8%/23.2% errors;
  GraphSAGE supplies the Edge and singleton diagnostics.
- The submitter-aware key slightly lowers mean primary error
  (0.421375 -> 0.414332) and raises median-group Spearman
  (0.495561 -> 0.577141) in this pinned-fold attribution.
- Canonical folds and unknown masking both increase mean primary error in the
  cumulative sequence.
- Exclusion lowers mean primary error from 0.485124 to 0.468672 while reducing
  coverage from 100.00% to 94.54%; the two errors are not over the same rows.
- Fixed isolation lowers mean primary error from 0.468672 to 0.462314 but
  lowers median-group Spearman from 0.553623 to 0.528660. No confidence
  interval was requested or estimated for this Phase 1.5 attribution, so this
  report does not label either directional change distinguishable.
- The initial Phase 1.5 supported-exclusion run produced non-finite metrics
  because unsupported NaN targets reached the loss in the leaky audit graph.
  The mask was fixed and protected by a regression test before producing the
  canonical bundle.

## 5. Open questions

- Approve or revise the 18-system LOAFO family proposal before any LOAFO run.
- Confirm whether the CPU-only family should be interpreted as one deployment
  regime for stress testing or split by processor generation after author
  review.

## 6. Acceptance criteria met?

| Item | Met | Evidence |
| --- | --- | --- |
| 10-seed attribution table, including all combined isolation stage | Yes | `attribution_summary.csv`; all stages finite |
| Exact Phase 0 regression and reproduction decision | Yes | `regression_checks.csv`; all four checks exact |
| Sensitivity key excludes accelerator counts | Yes | `system_table`; 105 systems -> 90 groups |
| Group counts and size distribution | Yes | `sensitivity_summary.csv`, `sensitivity_group_size_distribution.csv`, `sensitivity_groups.csv` |
| Real production training in leakage test | Yes | 150 epochs, patience 20; saved state comparison |
| Leak-reintroduction mutation check | Yes | Guard fails and actual weights change |
| Leakage runtime reported and prior speed explained | Yes | 11.917 seconds; former checks used one epoch |
| Bundle provenance resolved | Yes | No run-affecting change from `5c9dfde` to `fb36169` |
| Primary error mean +/- SD defined | Yes | Per-seed median, then mean/SD over seeds 7-16 |
| 18-system LOAFO proposal, no execution | Yes | Table above; no LOAFO model run |
| Numeric accelerator feature and count-tuple contract | Yes | Scalar `total_accelerators`; tuple only audit grouping |
| Phase 2 preregistration committed with hash | Yes | `protocol.md`, commit `d3bc835`, SHA recorded |

Phase 1.5 is complete. LOAFO remains blocked on orchestrator approval, and no
Phase 2 run has started.
