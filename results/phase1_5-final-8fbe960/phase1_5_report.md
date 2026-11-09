# Phase 1.5 Report

Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.

## 1. Attribution and regression

Primary error is the median `|ln(prediction / measured)|` over the rows evaluated in a seed. The mean and SD are across seeds 7-16 after first computing that row-level median within each seed.
The attribution loop pins each fold construction once and varies only the model seed. The separate exact Phase 0 replay below retains the historical seed-coupled folds.

| Step | Change | Primary error mean | SD | Median-group Spearman mean | SD | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | original settings | 0.421375 | 0.027125 | 0.495561 | 0.020163 | 100.00% |
| a_submitter_key | a. submitter-aware system key | 0.414332 | 0.034083 | 0.577141 | 0.047681 | 100.00% |
| b_new_folds | b. canonical suite-stratified five-fold split | 0.470127 | 0.029587 | 0.493378 | 0.073014 | 100.00% |
| c_unknown_masking | c. unknown masking at p=0.1 | 0.485124 | 0.041135 | 0.466085 | 0.073741 | 100.00% |
| d_supported_exclusion | d. unsupported-group exclusion | 0.468672 | 0.049192 | 0.553623 | 0.050666 | 94.54% |
| e_fixed_isolation | e. fixed train/inference isolation | 0.462314 | 0.034193 | 0.528660 | 0.025100 | 94.54% |

| Phase 0 regression model | Primary error | Median-group MAPE | Median-row MAPE | Median-group Spearman | Edge mean APE | Singleton APE share | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| graphsage | 0.289413 | 38.7772% | 27.4132% | 0.711303 | 6289.5298% | 46.2140% | 5.300 |
| gat | 0.244249 | 29.7565% | 23.1571% | 0.820376 | 800.3811% | 95.2324% | 7.807 |

The exact Phase 0 source and data were executed from `292ff7f23eb41335881a0075337a60bef72a78c2`. GAT reproduces 29.7565%/23.1571%; GraphSAGE reproduces the 6,289.5298% Edge mean APE and 46.2140% singleton share. These were never all one model's diagnostic set.

## 2. Sensitivity grouping

The corrected key is `submitter|accelerator model|processor`; accelerator counts are excluded.

| Suite | Systems | Measurements | Groups |
| --- | ---: | ---: | ---: |
| datacenter | 96 | 457 | 83 |
| edge | 9 | 19 | 7 |
| total | 105 | 476 | 90 |

| Systems per group | Group count |
| ---: | ---: |
| 1 | 78 |
| 2 | 9 |
| 3 | 3 |

All groups are enumerated in `sensitivity_groups.csv`.

## 3. Leakage runtime

The complete suite ran in 11.917 seconds. The previous 18-second suite used `epochs=1, patience=1` in its 130-fold weight loop. The strengthened production check uses the default 150-epoch budget with patience 20, saves and reloads actual state dictionaries, and confirms the reintroduced leak changes weights.

```text
test_multiple_count_signatures_are_retained_for_grouping (test_data.TestSystemTable.test_multiple_count_signatures_are_retained_for_grouping) ... ok
test_sensitivity_key_excludes_accelerator_counts (test_data.TestSystemTable.test_sensitivity_key_excludes_accelerator_counts) ... ok
test_total_accelerators_remain_the_numeric_feature (test_data.TestSystemTable.test_total_accelerators_remain_the_numeric_feature) ... ok
test_baseline_fit_never_receives_test_rows_for_every_fold (test_leakage.TestEveryFoldIsolation.test_baseline_fit_never_receives_test_rows_for_every_fold) ... ok
test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold (test_leakage.TestEveryFoldIsolation.test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold) ... ok
test_roles_are_disjoint_for_every_fold (test_leakage.TestEveryFoldIsolation.test_roles_are_disjoint_for_every_fold) ... ok
test_unknown_vectors_are_exercised_during_training (test_leakage.TestEveryFoldIsolation.test_unknown_vectors_are_exercised_during_training) ... ok
test_loss_masks_exclude_unsupported_rows_without_isolation (test_leakage.TestSupportedLeakyPolicy.test_loss_masks_exclude_unsupported_rows_without_isolation) ... ok
test_production_weights_ignore_scrambled_test_metadata (test_leakage.TestWeightPerturbation.test_production_weights_ignore_scrambled_test_metadata) ... ok
test_reintroduced_leak_fails_the_production_weight_check (test_leakage.TestWeightPerturbation.test_reintroduced_leak_fails_the_production_weight_check) ... ok
test_paired_difference_uses_same_system_draw (test_metrics.TestClusteredBootstrap.test_paired_difference_uses_same_system_draw) ... ok

----------------------------------------------------------------------
Ran 11 tests in 9.243s

OK
```

## 4. Provenance

```text
bundle_manifest_commit=5c9dfde854d258d6848d1636564861d66e16a146
phase1_results_commit=fb36169

git diff --stat 5c9dfde fb36169 -- run-affecting files:
No run-affecting files differ between the manifest commit 5c9dfde and HEAD fb36169. The intervening commit changed only CHANGELOG.md and result-report metadata.
```

## 5. LOAFO proposal (not run)

Mixed-accelerator systems are proposed as always-train and are not held out as families.

| System | Accelerator | Vendor | Architecture generation | Proposed family | Disposition | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `edge|Atlas_Inference|AMD Strix Halo desktop (Ryzen AI Max+ 395 / Radeon 8060S)|edge` | AMD Radeon 8060S Graphics | AMD | RDNA 3.5 integrated Radeon 8060S | `amd_rdna35_integrated_radeon_8060s` | holdout_eligible | Radeon 8060S in the Strix Halo system is an integrated RDNA-generation device. |
| `datacenter|Cisco|X410M8_1_node_4S_GNR_86C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Dell|PowerEdge R770AP_128C 24x96GB (6400 MT/s)|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Dell|PowerEdge R770_86C 16x64GB (6400 MT/s)|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Intel|1-node-2S-GNR_128C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Oracle|BM.Standard4.Ax.120|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Quanta_Cloud_Technology|1-node-2S-GNR_86C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|RedHat_Intel|1-node-2S-GNR_96C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Supermicro|1-node-2S-GNR_128C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Supermicro|1-node-2S-GNR_86C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `datacenter|Supermicro|1-node-4S-GNR_86C|datacenter` | <blank> | Intel | Xeon 6 Granite Rapids | `cpu_only_intel_xeon6_granite_rapids` | holdout_eligible | No accelerator is reported; the processor identifies an Intel Xeon 6 CPU-only execution regime. |
| `edge|Atlas_Inference|NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)|edge` | NVIDIA GB10 | NVIDIA | Blackwell integrated GB10 | `nvidia_blackwell_integrated_gb10` | holdout_eligible | GB10 is an integrated Grace-Blackwell device, distinct from discrete B-series accelerators. |
| `edge|Dell|Dell Pro Max (1x GB10)|edge` | NVIDIA GB10 | NVIDIA | Blackwell integrated GB10 | `nvidia_blackwell_integrated_gb10` | holdout_eligible | GB10 is an integrated Grace-Blackwell device, distinct from discrete B-series accelerators. |
| `edge|NVIDIA|NVIDIA Jetson AGX Thor Developer Kit 128G (TensorRT)|edge` | NVIDIA Jetson AGX Thor 128G | NVIDIA | Blackwell integrated Jetson Thor | `nvidia_blackwell_integrated_jetson_thor` | holdout_eligible | Jetson AGX Thor is an integrated Blackwell-generation edge device. |
| `datacenter|NVIDIA|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)|datacenter` | NVIDIA VR200 | NVIDIA | Rubin VR200 | `nvidia_rubin_vr200` | holdout_eligible | The reported VR200 accelerator denotes the Rubin-generation product family. |
| `datacenter|Nebius|Nebius VR200 NVL72 (36x VR200-288GB_aarch64 TensorRT)|datacenter` | NVIDIA VR200 | NVIDIA | Rubin VR200 | `nvidia_rubin_vr200` | holdout_eligible | The reported VR200 accelerator denotes the Rubin-generation product family. |
| `datacenter|Dell_MangoBoost|Heterogeneous Four-Cluster (2x8x MI300X + 2x8x MI355X LLMBoost)|datacenter` | AMD Instinct MI300X 192GB HBM3 (x2x8) and AMD Instinct MI355X 288GB HBM3e (x2x8) | AMD | CDNA3 MI300X + CDNA4 MI355X | `mixed_amd_cdna3_mi300x_cdna4_mi355x` | always_train | Two AMD Instinct architecture generations are reported by one heterogeneous cluster. |
| `datacenter|Cisco|Cisco UCS H200 x8 + MI350X x8 G200 2x2 TOR|datacenter` | NVIDIA H200-SXM-141GB + AMD Instinct MI350X 288GB HBM3e | NVIDIA + AMD | Hopper H200 + CDNA4 MI350X | `mixed_nvidia_hopper_h200_amd_cdna4_mi350x` | always_train | Two discrete accelerator architectures are reported by the same system. |

## 6. Feature contract

Each result row retains its scalar `total_accelerators` value as the numeric feature. The sorted reported-count tuple is retained only for grouping/audit and is excluded from the sensitivity key.

| System | Reported count signature |
| --- | --- |
| `datacenter|HPE|HPE Compute XD690 (8x NVIDIA B300-SXM-270GB TensorRT)|datacenter` | `8;16` |
| `datacenter|NVIDIA|NVIDIA GB300 NVL72 (72x GB300-288GB_aarch64 TensorRT)|datacenter` | `72;288` |

## 7. Phase report

### What was done

- Re-executed the pinned Phase 0 source and verified all four diagnostics.
- Ran the cumulative one-change-at-a-time GraphSAGE attribution.
- Corrected and enumerated the sensitivity groups.
- Strengthened the leakage test to exercise production training.
- Recorded source/data provenance and the Phase 2 feature contract.
- Proposed, but did not run, the 18-system LOAFO review families.

### Files changed

All numerical claims above trace to CSV or text files in this directory; exact hashes are recorded in `results_manifest.json`.

### Numbers that changed

- Sensitivity groups: 101 -> 90.
- Phase 1's non-reproducing legacy diagnostics are explained and replaced by the pinned regression table above.

### Surprising or contradictory findings

- The Phase 0 figures reproduce exactly, but they belong to different model outputs: GAT owns the 29.8%/23.2% errors; GraphSAGE owns the Edge and singleton diagnostics.
- The prior leakage suite was fast because its weight comparisons used one epoch; the strengthened suite now exercises production training.

### Open questions

- Approve or revise the 18-system LOAFO proposal before any LOAFO run.

### Acceptance criteria

- Attribution table: yes.
- Phase 0 regression: yes.
- Sensitivity grouping fix and counts: yes.
- Production-training leakage mutation test: yes.
- Provenance: yes.
- Metric aggregation definition: yes.
- LOAFO table: proposed; execution intentionally withheld.
- Feature contract: yes.
