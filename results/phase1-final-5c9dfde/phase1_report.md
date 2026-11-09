# Phase 1 Evaluation Infrastructure Report

Protocol: `cv5`  
Model: `graphsage`  
Seeds: 7, 8, 9, 10, 11, 12, 13, 14, 15, 16  
Bootstrap: 2,000 system-cluster resamples  
Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge

## System-count table

| Suite | Systems | Measurements | Sensitivity groups | Accelerator families |
| --- | ---: | ---: | ---: | ---: |
| datacenter | 96 | 457 | 92 | 12 |
| edge | 9 | 19 | 9 | 4 |
| total | 105 | 476 | 101 | 15 |

The system key is `suite|submitter|system_name|system_type`. The sensitivity key is `submitter|accelerator|total accelerator count values|processor`.

Residual system-key ambiguity: three keys contain two Public IDs each, and two systems report more than one total-accelerator count. The fold key therefore remains a system-description grouping, not a hardware serial number.

## Fold inventory

| Protocol | Folds | Minimum train | Maximum train | Minimum validation | Maximum validation | Minimum test | Maximum test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cv5 | 5 | 73 | 75 | 10 | 10 | 20 | 22 |
| loafo | 15 | 63 | 92 | 9 | 12 | 1 | 33 |
| loso | 105 | 92 | 92 | 12 | 12 | 1 | 1 |
| sensitivity | 5 | 73 | 75 | 10 | 10 | 20 | 22 |

All assignments are in `fold_assignments.csv`, keyed by `system_id`.

GAT implementation: four attention heads, 16 output channels per head, concatenated to the 64-channel hidden representation.

## Old-versus-new GNN results

Primary error is median `|log(prediction / measured)|`, using natural logarithms on supported target groups only.

| Policy | Primary error mean | Seed SD | MAPE mean | Normalized RMSE mean | Median group primary error | Median group Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| isolated_supported | 0.462 | 0.034 | 166.49% | 1.488 | 0.430 | 0.529 |
| legacy_leaky_global | 0.465 | 0.035 | 164.98% | 1.496 | 0.448 | 0.492 |

Median of the ten independent, system-clustered 95% intervals (each with B=2,000):

| Policy | Primary error | 95% CI |
| --- | ---: | --- |
| isolated_supported | 0.462 | (0.339, 0.625) |
| legacy_leaky_global | 0.465 | (0.348, 0.624) |

Paired within-draw policy difference (`legacy_leaky_global` minus `isolated_supported`): 0.004 [-0.078, 0.078]; P(left better) = 0.475.

## Legacy sparse and Edge diagnostics

These are diagnostics from the legacy all-row/global-fallback policy inside the new pipeline. They are not primary metrics.

| Statistic | Mean | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Singleton APE share | 98.06% | 97.03% | 98.96% |
| Edge mean APE (%) | 187.07 | 138.19 | 316.15 |
| Edge median APE (%) | 93.04 | 81.76 | 121.38 |

Unsupported test rows: 26 distinct rows, represented by 520 seed-policy records across 10 seeds and 2 policy/model labels.

## Leakage tests

Status: **PASS**. The complete output is in [`leakage_test_output.txt`](leakage_test_output.txt).

```text
test_baseline_fit_never_receives_test_rows_for_every_fold (test_leakage.TestEveryFoldIsolation.test_baseline_fit_never_receives_test_rows_for_every_fold) ... ok
test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold (test_leakage.TestEveryFoldIsolation.test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold) ... ok
test_roles_are_disjoint_for_every_fold (test_leakage.TestEveryFoldIsolation.test_roles_are_disjoint_for_every_fold) ... ok
test_unknown_vectors_are_exercised_during_training (test_leakage.TestEveryFoldIsolation.test_unknown_vectors_are_exercised_during_training) ... ok
test_isolated_weights_ignore_scrambled_test_metadata_every_fold (test_leakage.TestWeightPerturbation.test_isolated_weights_ignore_scrambled_test_metadata_every_fold) ... ok
test_legacy_graph_fails_the_isolation_guard (test_leakage.TestWeightPerturbation.test_legacy_graph_fails_the_isolation_guard) ... ok
test_legacy_weights_change_when_test_metadata_changes (test_leakage.TestWeightPerturbation.test_legacy_weights_change_when_test_metadata_changes) ... ok
test_paired_difference_uses_same_system_draw (test_metrics.TestClusteredBootstrap.test_paired_difference_uses_same_system_draw) ... ok

----------------------------------------------------------------------
Ran 8 tests in 18.351s

OK
```

## Accelerator-family review cases

18 system rows require review or are ambiguous.

| System | Accelerator | Family | Status |
| --- | --- | --- | --- |
| `datacenter|Dell_MangoBoost|Heterogeneous Four-Cluster (2x8x MI300X + 2x8x MI355X LLMBoost)|datacenter` | AMD Instinct MI300X 192GB HBM3 (x2x8) and AMD Instinct MI355X 288GB HBM3e (x2x8) | mixed_amd_mi300x_mi355x | ambiguous |
| `datacenter|Cisco|Cisco UCS H200 x8 + MI350X x8 G200 2x2 TOR|datacenter` | NVIDIA H200-SXM-141GB + AMD Instinct MI350X 288GB HBM3e | mixed_nvidia_h200_amd_mi350x | ambiguous |
| `datacenter|Cisco|X410M8_1_node_4S_GNR_86C|datacenter` | <blank> | cpu_only | review |
| `datacenter|Dell|PowerEdge R770AP_128C 24x96GB (6400 MT/s)|datacenter` | <blank> | cpu_only | review |
| `datacenter|Dell|PowerEdge R770_86C 16x64GB (6400 MT/s)|datacenter` | <blank> | cpu_only | review |
| `datacenter|Intel|1-node-2S-GNR_128C|datacenter` | <blank> | cpu_only | review |
| `datacenter|Oracle|BM.Standard4.Ax.120|datacenter` | <blank> | cpu_only | review |
| `datacenter|Quanta_Cloud_Technology|1-node-2S-GNR_86C|datacenter` | <blank> | cpu_only | review |
| `datacenter|RedHat_Intel|1-node-2S-GNR_96C|datacenter` | <blank> | cpu_only | review |
| `datacenter|Supermicro|1-node-2S-GNR_128C|datacenter` | <blank> | cpu_only | review |
| `datacenter|Supermicro|1-node-2S-GNR_86C|datacenter` | <blank> | cpu_only | review |
| `datacenter|Supermicro|1-node-4S-GNR_86C|datacenter` | <blank> | cpu_only | review |
| `edge|Atlas_Inference|AMD Strix Halo desktop (Ryzen AI Max+ 395 / Radeon 8060S)|edge` | AMD Radeon 8060S Graphics | amd_radeon_8060s | review |
| `edge|Atlas_Inference|NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)|edge` | NVIDIA GB10 | nvidia_blackwell_gb10 | review |
| `edge|Dell|Dell Pro Max (1x GB10)|edge` | NVIDIA GB10 | nvidia_blackwell_gb10 | review |
| `edge|NVIDIA|NVIDIA Jetson AGX Thor Developer Kit 128G (TensorRT)|edge` | NVIDIA Jetson AGX Thor 128G | nvidia_blackwell_jetson_thor | review |
| `datacenter|NVIDIA|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)|datacenter` | NVIDIA VR200 | nvidia_rubin_vr200 | review |
| `datacenter|Nebius|Nebius VR200 NVL72 (36x VR200-288GB_aarch64 TensorRT)|datacenter` | NVIDIA VR200 | nvidia_rubin_vr200 | review |

## Phase report

### 1. What was done

- Generated all four fold families in one canonical keyed file.
- Ran ten fixed seeds with identical fold assignments.
- Computed the primary median absolute log ratio, MAPE, normalized RMSE, per-group metrics, and median-group Spearman correlation.
- Computed system-clustered bootstrap intervals and a paired within-draw policy difference.
- Ran the full leakage suite and generated all figures from saved outputs.

### 2. Files changed

All files in this result directory were generated by this run. See `results_manifest.json` for hashes.

### 3. Numbers that changed

The authoritative old-versus-new values are the tables above and `metrics_seed_summary.csv`; no value was transcribed from the old thesis or old result CSVs.

### 4. Surprising or contradictory findings

- The submitter-aware key increases the system count from 96 to 105.
- The legacy sparse diagnostic does not retain the Phase 0 values when rerun under the new submitter-aware folds and unknown masking; the generated values above supersede the old transcription.
- The 15 accelerator-family folds contain singleton test families; their estimates are not stable performance claims.
- The submitter-aware folds and trained unknown masking change the legacy sparse diagnostics. They should only be reported from the generated tables above.

### 5. Open questions

- Review the listed mixed-accelerator, integrated-device, and CPU-only family mappings before interpreting LOAFO.
- Decide whether the primary sparse-group policy should remain exclusion in Phase 2 or be replaced by shrinkage after comparison.

### 6. Acceptance criteria

- Fold files generated: **Yes**.
- Metrics implemented: **Yes**.
- System-clustered paired bootstrap B=2000: **Yes**.
- Ten seeds for the stochastic GNN: **Yes**.
- Old-code leakage guard fails: **Yes**.
- New isolation tests pass: **Yes**.
- Unknown categories trained: **Yes**.
- Sparse groups excluded from primary metrics: **Yes**.
- Manifest with commit, seeds, and fold hashes: **Yes**.
