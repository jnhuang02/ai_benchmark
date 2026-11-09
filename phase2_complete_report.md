# Phase 2 Complete Report

Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.

Canonical CV5: `mlperf_gnn/results/phase2-cv5-final-aeeb4b4/`  
LOSO: `mlperf_gnn/results/phase2-loso-final-a689ba5/`  
Sensitivity: `mlperf_gnn/results/phase2-sensitivity-final-a689ba5/`  
Combined inference: `mlperf_gnn/results/phase2-final-f51e57a/`  
Diagnostics: `mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/`

## 1. What was done

- Ran all requested CV5 diagnostics without changing CV5 predictions.
- Repeated GAT training for all 25 partition-fold cells and all ten seeds.
- Compared GAT with the Phase 1 numeric-feature path on the same fold.
- Reran GraphSAGE on partition 20261107 with both Phase 2 and Phase 1
  numeric features.
- Reported group-median Spearman as unavailable and enumerated the
  qualifying group count for every method.
- Listed the ten largest normalized residuals for RF, MLP, and
  GraphSAGE and identified partition/system drivers.
- Reran mixed-model fits to capture convergence and compared it with
  Ridge by target group.
- Ran LOSO with neural seeds 7, 11, and 16.
- Ran sensitivity with neural seeds 7-16.
- Applied Holm correction across the 8 CV/LOSO secondary tests.
- Did not run LOAFO.

## 2. Files changed

Key outputs:

- [GAT training summary](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/gat_training_summary.csv)
- [GAT by fold and seed](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/gat_training_by_fold_seed.csv)
- [GraphSAGE regression](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/graphsage_regression_comparison.csv)
- [Spearman counts](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/spearman_group_summary.csv)
- [Top residuals](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/top_abs_z_residuals.csv)
- [Partition errors](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/partition_primary_errors.csv)
- [Mixed specification](../mlperf_gnn/results/phase2-cv5-diagnostics-77c4752/mixed_model_specification.json)
- [LOSO table](../mlperf_gnn/results/phase2-loso-final-a689ba5/main_table.csv)
- [Sensitivity table](../mlperf_gnn/results/phase2-sensitivity-final-a689ba5/main_table.csv)
- [Co-primary decision](../mlperf_gnn/results/phase2-final-f51e57a/co_primary_decision.json)
- [Holm results](../mlperf_gnn/results/phase2-final-f51e57a/secondary_holm.csv)

The pre-existing thesis source and bibliography were not edited.

## 3. Numbers that changed

### GAT diagnostic

GAT does not collapse to a constant. Across all 25 partition-fold cells:

- Normalized prediction variance ranged from `0.008526` to `0.126357`,
  median `0.037296`.
- Unique predictions per fold-seed ranged from 69 to 112.
- Mean best epoch ranged by fold from 1 to 67;
  mean epochs trained ranged from 21 to 87.
- Mean final training Huber loss ranged from `0.282157` to `0.360742`.
- Mean final validation loss ranged from `0.254338` to `1.506562`.

The poor GAT rank correlation is therefore a ranking/outlier failure,
not a near-constant-output failure.

| Partition | Fold | Raw prediction variance | Normalized variance | Best epoch range | Epochs range | Final train loss | Final validation loss |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 0 | 0 | 23747940269.118965 | 0.029487 | 2-11 | 22-31 | 0.320480 | 0.508604 |
| 0 | 1 | 3329150387.268369 | 0.017392 | 1-7 | 21-27 | 0.338733 | 0.531287 |
| 0 | 2 | 7478676568.673811 | 0.052373 | 2-53 | 22-73 | 0.321478 | 1.506562 |
| 0 | 3 | 4754691600.798477 | 0.025856 | 1-15 | 21-35 | 0.344456 | 0.462147 |
| 0 | 4 | 2404054387.922655 | 0.013216 | 1-11 | 21-31 | 0.339347 | 0.606122 |
| 1 | 0 | 3310442002.952274 | 0.038146 | 3-14 | 23-34 | 0.360742 | 0.523711 |
| 1 | 1 | 3191624898.653304 | 0.012292 | 5-10 | 25-30 | 0.343176 | 0.254338 |
| 1 | 2 | 3010631641.843165 | 0.037296 | 1-18 | 21-38 | 0.355699 | 0.304615 |
| 1 | 3 | 4469609055.135063 | 0.057452 | 1-11 | 21-31 | 0.319275 | 0.371789 |
| 1 | 4 | 6978001440.001158 | 0.058659 | 2-13 | 22-33 | 0.327179 | 0.521882 |
| 2 | 0 | 2650885331.333700 | 0.027876 | 1-10 | 21-30 | 0.352613 | 0.559607 |
| 2 | 1 | 6303711828.500959 | 0.103112 | 10-21 | 30-41 | 0.314947 | 0.337710 |
| 2 | 2 | 3738434413.228674 | 0.018868 | 1-11 | 21-31 | 0.352103 | 0.435522 |
| 2 | 3 | 5004580751.878288 | 0.104896 | 1-21 | 21-41 | 0.337635 | 0.938181 |
| 2 | 4 | 12944821494.481199 | 0.035502 | 1-4 | 21-24 | 0.339356 | 0.511916 |
| 3 | 0 | 3582200448.003819 | 0.027143 | 1-5 | 21-25 | 0.339821 | 0.274804 |
| 3 | 1 | 6395599403.703877 | 0.072930 | 2-24 | 22-44 | 0.303435 | 1.124963 |
| 3 | 2 | 1989985077.437410 | 0.031191 | 2-19 | 22-39 | 0.353581 | 0.691226 |
| 3 | 3 | 34211707004.627941 | 0.040058 | 2-23 | 22-43 | 0.337081 | 0.536759 |
| 3 | 4 | 11848683042.980658 | 0.008526 | 1-11 | 21-31 | 0.353878 | 0.447962 |
| 4 | 0 | 4629038695.453421 | 0.098973 | 9-34 | 29-54 | 0.333221 | 1.255842 |
| 4 | 1 | 2605054526.592704 | 0.038612 | 2-6 | 22-26 | 0.350790 | 0.369279 |
| 4 | 2 | 9257081233.526730 | 0.024857 | 1-7 | 21-27 | 0.347335 | 0.671576 |
| 4 | 3 | 11180302768.100794 | 0.126357 | 9-17 | 29-37 | 0.311594 | 0.425229 |
| 4 | 4 | 6634275116.000277 | 0.075154 | 2-67 | 22-87 | 0.282157 | 0.458015 |

Same-fold seed-7 comparison on `p0_fold_0`:

| Run | Primary error | Raw variance | Normalized variance | Unique predictions | Best/epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| Phase 2 GAT, p0_fold_0 seed 7 | 0.658130 | 13811535191.977949 | 0.024403 | 103 | 2/22 |
| Phase 1 feature path, four-head GAT, same fold/seed | 0.661537 | 22079200887.459156 | 0.028956 | 103 | 7/27 |

### GraphSAGE regression

- Phase 2 GraphSAGE full-feature seed mean: `0.473849066`
- Phase 2 GraphSAGE full-feature seed SD: `0.039451679`
- Phase 2 GraphSAGE full-feature seed-ensemble primary: `0.445143709`
- Phase 2 code, Phase 1 numeric-feature seed mean: `0.462313987`
- Phase 2 code, Phase 1 numeric-feature seed SD: `0.034193466`
- Phase 1.5 fixed-isolation seed mean: `0.462313987`
- Phase 2 minus Phase 1.5 seed-mean difference: `0.011535078`

The Phase 2 CPU-only indicator alone changes the partition-0 seed
mean by `0.011535`; without it, the Phase 2 code reproduces the
Phase 1.5 value `0.4623139872076701` exactly. Pooling is not the
gap: both use the same five OOF folds and 450 rows per seed.

### Spearman availability and group counts

Group-median Spearman is reported as `NA`. Its predictions have no
within-group rank variation. No other method was globally constant.
The qualifying group count is 18-20 across the five partitions.

| Method | Groups min | Groups max | Mean group Spearman |
| --- | ---: | ---: | ---: |
| `best_gnn` | 18 | 20 | 0.619981 |
| `gat` | 18 | 20 | 0.004719 |
| `gcn` | 18 | 20 | 0.618737 |
| `gradient_boosting` | 18 | 20 | 0.740296 |
| `graphsage` | 18 | 20 | 0.634863 |
| `graphsage_no_accelerator` | 18 | 20 | 0.623835 |
| `graphsage_no_processor` | 18 | 20 | 0.627766 |
| `graphsage_no_system` | 18 | 20 | 0.658244 |
| `graphsage_no_workload` | 18 | 20 | 0.635247 |
| `graphsage_shuffled` | 18 | 20 | 0.643818 |
| `group_median` | 18 | 20 | NA |
| `hierarchical_mixed` | 18 | 20 | 0.706526 |
| `mlp` | 18 | 20 | 0.815404 |
| `random_forest` | 18 | 20 | 0.793714 |
| `ridge` | 18 | 20 | 0.827641 |

### Largest normalized residuals

Top-ten SSE shares: RF `0.851005`, MLP `0.871018`, GraphSAGE
`0.779098`. Two recurring rows dominate all three models:
Llama 2 70B Interactive on the Dell_AMD XE7745 MI350P system and
DeepSeek-R1 Interactive on the NVIDIA Vera Rubin NVL72 system.
These extreme standardized predictions drive normalized RMSE above 1.5.

| Method | Rank | Partition/fold | System | Target group | |z| |
| --- | ---: | --- | --- | --- | ---: |
| `random_forest` | 1 | cv5_partition_2/p2_fold_0 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 54.016086 |
| `random_forest` | 2 | cv5_partition_2/p2_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 25.720372 |
| `random_forest` | 3 | cv5_partition_3/p3_fold_0 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 21.769233 |
| `random_forest` | 4 | cv5_partition_4/p4_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 20.918499 |
| `random_forest` | 5 | cv5_partition_0/p0_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 20.727899 |
| `random_forest` | 6 | cv5_partition_1/p1_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 17.357846 |
| `random_forest` | 7 | cv5_partition_1/p1_fold_4 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 11.300582 |
| `random_forest` | 8 | cv5_partition_0/p0_fold_2 | `datacenter\|MiTAC\|G8825Z5\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 11.012146 |
| `random_forest` | 9 | cv5_partition_3/p3_fold_1 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 10.792147 |
| `random_forest` | 10 | cv5_partition_4/p4_fold_3 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 10.757356 |
| `mlp` | 1 | cv5_partition_2/p2_fold_0 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 54.173665 |
| `mlp` | 2 | cv5_partition_2/p2_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 24.844107 |
| `mlp` | 3 | cv5_partition_3/p3_fold_0 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 21.328548 |
| `mlp` | 4 | cv5_partition_0/p0_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 20.230146 |
| `mlp` | 5 | cv5_partition_4/p4_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 20.092052 |
| `mlp` | 6 | cv5_partition_1/p1_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 16.388398 |
| `mlp` | 7 | cv5_partition_0/p0_fold_0 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 10.841352 |
| `mlp` | 8 | cv5_partition_3/p3_fold_1 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 10.751463 |
| `mlp` | 9 | cv5_partition_1/p1_fold_4 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 10.735488 |
| `mlp` | 10 | cv5_partition_4/p4_fold_3 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 10.510847 |
| `graphsage` | 1 | cv5_partition_2/p2_fold_0 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 54.648065 |
| `graphsage` | 2 | cv5_partition_2/p2_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 24.549952 |
| `graphsage` | 3 | cv5_partition_3/p3_fold_0 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 21.379720 |
| `graphsage` | 4 | cv5_partition_0/p0_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 20.102450 |
| `graphsage` | 5 | cv5_partition_4/p4_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 19.562786 |
| `graphsage` | 6 | cv5_partition_1/p1_fold_1 | `datacenter\|NVIDIA\|NVIDIA Vera Rubin NVL72 (72x VR-288GB_aarch64)\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 16.209851 |
| `graphsage` | 7 | cv5_partition_1/p1_fold_4 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 11.554310 |
| `graphsage` | 8 | cv5_partition_0/p0_fold_0 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 11.462374 |
| `graphsage` | 9 | cv5_partition_3/p3_fold_1 | `datacenter\|Dell_AMD\|XE7745_8xMI350P_2xEPYC_9455\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 11.406132 |
| `graphsage` | 10 | cv5_partition_0/p0_fold_2 | `datacenter\|MiTAC\|G8825Z5\|datacenter` | `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 11.312961 |

### Partition sensitivity

| Method | P0 | P1 | P2 | P3 | P4 | Partition SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_forest` | 0.230595 | 0.172587 | 0.242461 | 0.186130 | 0.177491 | 0.032297 |
| `mlp` | 0.253798 | 0.261248 | 0.265858 | 0.288952 | 0.235500 | 0.019410 |
| `graphsage` | 0.445144 | 0.268147 | 0.339907 | 0.489107 | 0.309048 | 0.093310 |
| `gcn` | 0.425807 | 0.249714 | 0.315997 | 0.462557 | 0.319611 | 0.087236 |
| `gat` | 0.594694 | 0.543353 | 0.675100 | 0.696522 | 0.653544 | 0.062715 |

GraphSAGE is highest on partition 3 (`0.489107`) and partition 0
(`0.445144`), and lowest on partition 1 (`0.268147`). The largest
cross-partition system ranges are shown below.

| Method | System | Error range | Mean error |
| --- | --- | ---: | ---: |
| `graphsage` | `datacenter\|Crusoe\|Crusoe Cloud 64x8 MI355X (512 GPU) — gpt-oss-120b / vLLM\|datacenter` | 2.648014 | 0.789373 |
| `graphsage` | `datacenter\|TTA\|ManyCoreSoft DeepGadget dg5W (4x NVIDIA RTX PRO 6000 Blackwell 96GB TensorRT)\|datacenter` | 1.658952 | 2.154358 |
| `graphsage` | `datacenter\|Nebius\|Nebius VR200 NVL72 (36x VR200-288GB_aarch64 TensorRT)\|datacenter` | 1.587276 | 0.680116 |
| `graphsage` | `datacenter\|Azure\|Azure GB200 NVL72 (288x GB200-186GB_aarch64 TensorRT)\|datacenter` | 1.515578 | 0.481254 |
| `graphsage` | `datacenter\|AMD\|9x8xMI355X_2xEPYC_9575F\|datacenter` | 1.390455 | 0.863770 |
| `gcn` | `datacenter\|Crusoe\|Crusoe Cloud 64x8 MI355X (512 GPU) — gpt-oss-120b / vLLM\|datacenter` | 2.198838 | 0.652728 |
| `gcn` | `datacenter\|TTA\|ManyCoreSoft DeepGadget dg5W (4x NVIDIA RTX PRO 6000 Blackwell 96GB TensorRT)\|datacenter` | 2.022920 | 2.214286 |
| `gcn` | `datacenter\|MangoBoost\|MangoBoost 2x8 MI355X (16x MI355X LLMBoost P/D Disaggregated Interactive Engine)\|datacenter` | 1.429852 | 0.734773 |
| `gcn` | `datacenter\|AMD\|9x8xMI355X_2xEPYC_9575F\|datacenter` | 1.166528 | 0.798139 |
| `gcn` | `datacenter\|Dell\|PowerEdge XE7745 (8x NVIDIA RTX PRO 4500 Blackwell TensorRT)\|datacenter` | 1.164417 | 2.700101 |
| `mlp` | `datacenter\|AMD\|smci355-ccs-aus-m13-13 (AS -4126GS-NMR-LCC)\|datacenter` | 1.638709 | 0.910701 |
| `mlp` | `datacenter\|Supermicro\|1-node-4x-BMG-B70\|datacenter` | 1.508140 | 2.242165 |
| `mlp` | `datacenter\|Intel\|1-node-4x-BMG-B70\|datacenter` | 1.275222 | 2.366499 |
| `mlp` | `datacenter\|Crusoe\|Crusoe Cloud 64x8 MI355X (512 GPU) — gpt-oss-120b / vLLM\|datacenter` | 1.231358 | 0.483597 |
| `mlp` | `datacenter\|Oracle\|BM.Standard4.Ax.120\|datacenter` | 1.091827 | 0.711631 |
| `random_forest` | `datacenter\|Crusoe\|Crusoe Cloud 64x8 MI355X (512 GPU) — gpt-oss-120b / vLLM\|datacenter` | 1.939020 | 0.864834 |
| `random_forest` | `datacenter\|Cisco\|Cisco UCS X210 M8 - X580P PCIe Node (4x NVIDIA RTX PRO 4500 Blackwell-32GB TensorRT)\|datacenter` | 1.701883 | 1.287834 |
| `random_forest` | `datacenter\|Supermicro\|1-node-4x-BMG-B70\|datacenter` | 1.236103 | 2.715865 |
| `random_forest` | `datacenter\|Intel\|1-node-4x-BMG-B70\|datacenter` | 1.089602 | 2.595075 |
| `random_forest` | `datacenter\|MangoBoost\|MangoBoost 2x8 MI355X (16x MI355X LLMBoost P/D Disaggregated Interactive Engine)\|datacenter` | 1.009350 | 0.814570 |
| `gat` | `datacenter\|Intel\|1-node-2S-GNR_128C\|datacenter` | 2.083456 | 1.441881 |
| `gat` | `datacenter\|Supermicro\|1-node-4S-GNR_86C\|datacenter` | 2.065252 | 1.241227 |
| `gat` | `datacenter\|Cisco\|X410M8_1_node_4S_GNR_86C\|datacenter` | 1.968147 | 2.101471 |
| `gat` | `datacenter\|Quanta_Cloud_Technology\|1-node-2S-GNR_86C\|datacenter` | 1.944980 | 2.438199 |
| `gat` | `datacenter\|Supermicro\|1-node-2S-GNR_86C\|datacenter` | 1.927461 | 2.127067 |

Full top-10 driver tables are in `partition_system_drivers.csv`.

### Mixed model

Formula: `y ~ C(target_group) + log_total + is_cpu + accelerator_memory`.
Random intercepts: accelerator family, submitter, and system.
Held-out system random effect is fixed at zero; predictions use fixed
effects only, and held-out systems do not update variance components.

Convergence: 25/25 CV5 fold fits converged. Optimizers: `lbfgs` 25. LOSO subsequently required the logged Powell fallback on one fold.

Per-group mean absolute log error, mixed minus Ridge:

| Target group | Mixed | Ridge | Difference |
| --- | ---: | ---: | ---: |
| `datacenter\|v6.1\|closed\|Inference\|VLM\|\|Interactive\|Queries/s` | 1.412845 | 0.904864 | +0.507981 |
| `datacenter\|v6.1\|closed\|Inference\|VLM\|\|Offline\|Samples/s` | 0.442991 | 0.171370 | +0.271621 |
| `datacenter\|v6.1\|closed\|Inference\|VLM\|\|Server\|Queries/s` | 0.485173 | 0.158702 | +0.326471 |
| `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Interactive\|Tokens/s` | 0.061738 | 0.497522 | -0.435784 |
| `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Offline\|Tokens/s` | 0.438762 | 0.201066 | +0.237696 |
| `datacenter\|v6.1\|closed\|Inference\|deepseek-r1\|\|Server\|Tokens/s` | 0.442988 | 0.216771 | +0.226218 |
| `datacenter\|v6.1\|closed\|Inference\|gpt-oss-120b\|\|Interactive\|Tokens/s` | 0.324340 | 1.629814 | -1.305474 |
| `datacenter\|v6.1\|closed\|Inference\|gpt-oss-120b\|\|Offline\|Tokens/s` | 0.548020 | 0.284303 | +0.263717 |
| `datacenter\|v6.1\|closed\|Inference\|gpt-oss-120b\|\|Server\|Tokens/s` | 0.609094 | 0.312958 | +0.296136 |
| `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Interactive\|Tokens/s` | 0.223034 | 0.086267 | +0.136767 |
| `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Offline\|Tokens/s` | 0.529897 | 0.408557 | +0.121340 |
| `datacenter\|v6.1\|closed\|Inference\|llama2-70b-99.9\|\|Server\|Tokens/s` | 0.553323 | 0.393837 | +0.159485 |
| `datacenter\|v6.1\|closed\|Inference\|llama3.1-8b\|\|Interactive\|Tokens/s` | 0.539420 | 0.536045 | +0.003375 |
| `datacenter\|v6.1\|closed\|Inference\|llama3.1-8b\|\|Offline\|Tokens/s` | 0.472580 | 0.748636 | -0.276056 |
| `datacenter\|v6.1\|closed\|Inference\|llama3.1-8b\|\|Server\|Tokens/s` | 0.577234 | 0.930593 | -0.353359 |
| `datacenter\|v6.1\|closed\|Inference\|rgat\|\|Offline\|Samples/s` | 0.268109 | 0.210938 | +0.057171 |
| `datacenter\|v6.1\|closed\|Inference\|text_to_video\|\|Offline\|Samples/s` | 0.390489 | 0.244303 | +0.146186 |
| `datacenter\|v6.1\|closed\|Inference\|text_to_video\|\|SingleStream\|Latency (s)` | 0.284668 | 0.191074 | +0.093594 |
| `datacenter\|v6.1\|closed\|Inference\|whisper\|\|Offline\|Samples/s` | 0.511602 | 0.387077 | +0.124525 |
| `edge\|v6.1\|closed\|Inference\|agentic edge\|\|SingleStream\|Latency (ms)` | 0.406773 | 0.460056 | -0.053283 |

### Protocol and coverage

- Protocol SHA-256:
  `d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e`
- Recorded as unchanged in the CV5 manifest:
  `True`
- The CV5 average coverage is `94.495798%`, not `450/476 =
  94.537815%`, because support differs by partition.

| Partition | Evaluated rows | Coverage |
| --- | ---: | ---: |
| cv5_partition_0 | 450 | 94.537815% |
| cv5_partition_1 | 451 | 94.747899% |
| cv5_partition_2 | 452 | 94.957983% |
| cv5_partition_3 | 448 | 94.117647% |
| cv5_partition_4 | 448 | 94.117647% |
| mean | 449.8 | 94.495798% |

The five partition coverages average to 449.8 rows and 94.495798%.

### Primary co-primary decision

| Protocol | Difference | 95% CI | Label |
| --- | ---: | --- | --- |
| cv5_partitioned | 0.115597 | [0.047136, 0.198051] | inconclusive |
| loso | 0.104155 | [0.023774, 0.192784] | inconclusive |
| sensitivity | 0.158657 | [0.058606, 0.257561] | worse |

Combined primary label: **inconclusive**. Both
co-primary protocols return `inconclusive`; sensitivity is labeled
`worse` but is not co-primary.

### Secondary comparisons after Holm

| Protocol | Comparison | Difference [95% CI] | Raw label | Holm p | Significant |
| --- | --- | --- | --- | ---: | --- |
| cv5_partitioned | `graphsage - random_forest` | 0.171529 [0.111863, 0.237339] | worse | 0.000000 | True |
| cv5_partitioned | `gat - graphsage` | 0.267067 [0.145077, 0.450114] | worse | 0.000000 | True |
| loso | `graphsage - random_forest` | 0.163960 [0.105163, 0.223994] | worse | 0.000000 | True |
| loso | `gat - graphsage` | 0.328409 [0.192823, 0.498889] | worse | 0.000000 | True |
| cv5_partitioned | `graphsage - graphsage_shuffled` | 0.024523 [0.004590, 0.045019] | equivalent | 0.096000 | False |
| loso | `graphsage - graphsage_shuffled` | -0.032199 [-0.077127, 0.012450] | inconclusive | 0.387000 | False |
| cv5_partitioned | `gcn - graphsage` | -0.016638 [-0.045146, 0.007482] | equivalent | 0.387000 | False |
| loso | `gcn - graphsage` | 0.022510 [-0.030877, 0.095708] | inconclusive | 0.485000 | False |

### Protocol tables

| Method | CV5 primary | LOSO primary | Sensitivity primary |
| --- | ---: | ---: | ---: |
| `best_gnn` | 0.376393 | 0.340865 | 0.416584 |
| `gat` | 0.632643 | 0.658164 | 0.742656 |
| `gcn` | 0.354737 | 0.356236 | 0.417344 |
| `gradient_boosting` | 0.336582 | 0.367885 | 0.336125 |
| `graphsage` | 0.370271 | 0.338161 | 0.365091 |
| `graphsage_no_accelerator` | 0.370623 | 0.321748 | 0.401364 |
| `graphsage_no_processor` | 0.353855 | 0.334312 | 0.360944 |
| `graphsage_no_system` | 0.348254 | 0.359857 | 0.379694 |
| `graphsage_no_workload` | 0.358888 | 0.343862 | 0.381687 |
| `graphsage_shuffled` | 0.341685 | 0.366378 | 0.386209 |
| `group_median` | 0.617066 | 0.580975 | 0.631946 |
| `hierarchical_mixed` | 0.506282 | 0.493480 | 0.503984 |
| `mlp` | 0.261071 | 0.246446 | 0.251640 |
| `random_forest` | 0.201853 | 0.165012 | 0.174637 |
| `ridge` | 0.331671 | 0.299884 | 0.372580 |

Full metric columns and intervals are in each protocol's
`main_table.csv`.

## 4. Anything surprising or contradicting the current narrative

- GAT does not collapse numerically; it produces 69-112 distinct
  predictions per fold but has near-zero rank correlation.
- The GraphSAGE difference from Phase 1.5 is entirely attributable
  to the added CPU-only indicator in the same-fold ablation.
- Random forest remains best under CV5 and LOSO.
- Best GNN minus MLP is inconclusive under both co-primary protocols
  because the positive intervals cross the `+0.05` boundary.
- Sensitivity is worse, materially stronger than its modest-group
  warning suggested.
- Two interactive-throughput rows dominate normalized RMSE.
- The mixed model converges in every CV5 fold but is consistently
  worse than Ridge on most target groups.

## 5. Open questions

- LOAFO remains unrun despite approval and requires explicit
  instruction before execution.
- Any model redesign after these results is exploratory.

## 6. Acceptance criteria

| Item | Status |
| --- | --- |
| GAT diagnostic and Phase 1 comparison | Complete |
| CV5 partition-20261107 GraphSAGE regression | Complete |
| Spearman NA and group counts | Complete |
| Top-10 residual audit | Complete |
| Per-partition GNN variance drivers | Complete |
| Mixed-model specification and convergence | Complete |
| Protocol hash and coverage reconciliation | Complete |
| LOSO, seeds 7/11/16 | Complete |
| Sensitivity, seeds 7-16 | Complete |
| Combined co-primary label | Inconclusive |
| LOAFO | Not run by instruction |

Phase 2 protocol execution is complete. Work stops before LOAFO.
