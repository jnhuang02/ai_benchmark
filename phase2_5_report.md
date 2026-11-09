# Phase 2.5 Report

Protocol: MLPerf Inference v6.1, Closed division, Datacenter and Edge. The frozen protocol hash is `d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e`. No Phase 3 thesis text was edited.

## 1. What was done

- Ran LOAFO on the approved 15-family map: 13 non-mixed family folds; both mixed systems were always in training.
- Fitted group median, Ridge, random forest, MLP, GraphSAGE, GCN, and GAT. The validation-selected best GNN was also retained.
- Used neural seeds 7, 11, 16.
- Computed each family separately and did not pool the headline, small/integrated, or CPU-only families.
- Used 2000 whole-system bootstrap resamples for family-specific paired intervals.
- Audited supported-row parity across models within every family.
- Computed RF, MLP, and GraphSAGE absolute-log tails from the saved CV5 and LOSO prediction files.
- Named the six LOSO folds with no supported held-out row and replayed the hierarchical mixed-model Powell fallback.
- Logged Protocol v2.2 as a history-only addendum: the orchestration LOAFO choices were not in the hash-pinned file, LOAFO has no pooling rule and is descriptive, and the protocol compute table was corrected.
- Evaluated the partial-pooling secondary against the frozen definition and marked it `not_run` under deviation `S3A-SHRINK-001` because the estimator is under-specified.
- Replaced interval display with a threshold rule based on supported systems: intervals only when at least 8 systems were tested.
- Recomputed LOAFO system coverage from the prediction file and reconciled all 105 systems.
- Recomputed the full 21-test suite and reconciled it with the earlier 8-test targeted run.
- Added group-median-baseline differences for every LOAFO family and counted best-GNN point improvements over MLP in 2 of 6 headline families.
- Generated a number-to-source map covering the Phase 2 and Phase 2.5 result artifacts, including a decision for every NA.

## 2. Files changed

Code and folds:

- `mlperf_gnn/generate_loafo_folds.py`
- `mlperf_gnn/run_phase2_loafo.py`
- `mlperf_gnn/analyze_phase2_5.py`
- `mlperf_gnn/run_stage3a_corrections.py`
- `mlperf_gnn/write_phase2_5_report.py`
- `mlperf_gnn/protocol_history.md` (Version 2.2 history addendum)
- `mlperf_gnn/folds/loafo_fold_assignments.csv`
- `mlperf_gnn/folds/loafo_fold_summary.csv`
- `mlperf_gnn/folds/loafo_fold_manifest.json`
- `mlperf_gnn/evaluation/phase2.py` (method selectors only; default Phase 2 behavior unchanged)
- `mlperf_gnn/tests/test_phase2.py`
- `mlperf_gnn/CHANGELOG.md`

New result bundles:

- `mlperf_gnn/results/phase2-loafo-final-1a65664/`
- `mlperf_gnn/results/phase2-5-analysis-4faac06/`
- `mlperf_gnn/results/phase2-5-corrections-e97bb93/`
- This report: `thesis_overleaf/phase2_5_report.md`
- Source map: `thesis_overleaf/phase2_5_number_source_map.csv`
- NA register: `thesis_overleaf/phase2_5_na_register.csv`

Existing result bundles were not modified. The thesis files `main.tex`, `main.pdf`, and `references.bib` were not edited.

## 3. Numbers that changed (old -> new)

| Quantity | Old | New |
| --- | --- | --- |
| LOAFO execution | Not run | 13 family folds completed |
| B200 supported rows | 47 | 45 |
| RF/MLP/GraphSAGE tail summaries | Not computed | Computed under CV5 and LOSO |
| Empty LOSO folds | Not enumerated in a result table | 6 folds named |
| Powell fallback system | Named only in the changelog | Replayed and saved |
| Partial-pooling secondary | Specified but not uniquely estimable | Logged as not run; deviation `S3A-SHRINK-001` |
| Full test suite | 8 targeted tests reported | 21 tests passed; all pasted in Appendix D |
| LOAFO system reconciliation | 97 stated without a full manifest | 97 tested out of 105; all eight non-tested systems identified |
| Family intervals | Shown regardless of system count | Intervals only for 5 families with >=8 supported systems |

No CV5, LOSO, sensitivity, co-primary, or Holm-corrected Phase 2 point estimate was changed. The Phase 2 co-primary rows remain:

| Protocol | Best GNN − MLP | 95% CI | Frozen label |
| --- | ---: | --- | --- |
| cv5_partitioned | 0.115597 | [0.047136, 0.198051] | inconclusive |
| loso | 0.104155 | [0.023774, 0.192784] | inconclusive |
| sensitivity | 0.158657 | [0.058606, 0.257561] | worse |

The combined co-primary label remains `inconclusive`: the point estimate is positive, each interval excludes zero, and each interval crosses the `+0.05` margin.

## 4. Anything surprising or contradicting the narrative

- B300 is an extreme extrapolation: 33 systems and 198 of 476 rows (41.596639%) were held out together.
- GB10 and Jetson Thor are same-generation Blackwell tests of new integrated form factors; VR200 is the new-generation case. They are reported separately and not pooled.
- The B200 proposal recorded 47 supported rows, while the actual train-role partition supports 45. The difference is retained as an audit discrepancy.
- The family point estimates do not support a uniform directional statement: GNNs are worse than MLP in MI350 and B200, while GraphSAGE, GCN, and best GNN have better point estimates than MLP in MI355 and CPU-only.
- Best-GNN has a lower primary-error point estimate than MLP in 2 of 6 headline families (`amd_cdna4_mi355x;intel_xe2_battlemage_arc_pro_b70`). This is a descriptive point-estimate count, not an equivalence decision.
- One of those two families is Intel Arc Pro B70, which had only 3 supported systems; its paired interval is therefore not shown.
- B300 is uniformly poor in absolute terms across the reported ladder, and its best-GNN-minus-MLP interval crosses zero.
- RTX PRO has high errors for every reported model, including random forest and MLP.
- Integrated GB10, Jetson Thor, Radeon 8060S, and H200 have at most two supported rows, so their intervals are degenerate or nearly degenerate and are descriptive rather than inferential.
- Direct recomputation from the LOAFO prediction file finds 97 unique tested systems, not a count inferred from the LOSO unsupported-fold list. The full 105-system reconciliation is reported below.

## Detailed LOAFO results

### (a) Headline families (at least four systems)

| Family | Systems | Supported systems | Rows | Supported rows | Group median | Ridge | Random forest | MLP | GraphSAGE | GCN | GAT | Best GNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd_cdna4_mi350x` | 4 | 3 | 24 | 20 | 0.764164 | 0.372809 | 0.330457 | 0.223854 | 0.877529 | 0.811681 | 0.735374 | 0.796515 |
| `amd_cdna4_mi355x` | 12 | 12 | 66 | 61 | 0.153894 | 0.534892 | 0.681118 | 0.404295 | 0.341609 | 0.288889 | 0.284788 | 0.288889 |
| `intel_xe2_battlemage_arc_pro_b70` | 4 | 3 | 19 | 17 | 1.691060 | 1.643861 | 0.836444 | 1.750370 | 1.419600 | 1.151481 | 1.698588 | 1.359058 |
| `nvidia_blackwell_b200` | 14 | 14 | 47 | 45 | 0.808193 | 0.325631 | 0.300555 | 0.201595 | 0.303524 | 0.289497 | 0.827602 | 0.303524 |
| `nvidia_blackwell_b300` | 33 | 33 | 198 | 198 | 0.938501 | 1.095572 | 1.008014 | 0.796966 | 0.852926 | 0.794819 | 0.992099 | 0.856458 |
| `nvidia_blackwell_rtx_pro` | 18 | 15 | 58 | 50 | 1.287673 | 1.067957 | 1.263829 | 1.358398 | 1.667265 | 1.663046 | 1.538275 | 1.693099 |

Paired differences, GNN minus MLP:
| Family | Supported systems | GraphSAGE − MLP | GCN − MLP | GAT − MLP | Best GNN − MLP |
| --- | ---: | --- | --- | --- | --- |
| `amd_cdna4_mi350x` | 3 | 0.653675; NA (too few systems) | 0.587827; NA (too few systems) | 0.511520; NA (too few systems) | 0.572661; NA (too few systems) |
| `amd_cdna4_mi355x` | 12 | -0.062686 [-0.447063, -0.006632] | -0.115406 [-0.504774, -0.038053] | -0.119507 [-0.467154, 0.108317] | -0.115406 [-0.504774, -0.038053] |
| `intel_xe2_battlemage_arc_pro_b70` | 3 | -0.330771; NA (too few systems) | -0.598889; NA (too few systems) | -0.051782; NA (too few systems) | -0.391312; NA (too few systems) |
| `nvidia_blackwell_b200` | 14 | 0.101929 [-0.024416, 0.243393] | 0.087902 [-0.010824, 0.272295] | 0.626007 [0.221833, 1.285488] | 0.101929 [-0.024416, 0.243393] |
| `nvidia_blackwell_b300` | 33 | 0.055960 [-0.089119, 0.242517] | -0.002147 [-0.167320, 0.275239] | 0.195132 [-0.116410, 0.417373] | 0.059492 [-0.135334, 0.239440] |
| `nvidia_blackwell_rtx_pro` | 15 | 0.308868 [-0.240216, 0.530187] | 0.304649 [-0.259646, 0.574935] | 0.179877 [-0.893381, 0.355269] | 0.334701 [-0.291528, 0.520475] |

Model differences from the group-median baseline:
| Family | Group median | Ridge − GM | Random forest − GM | MLP − GM | GraphSAGE − GM | GCN − GM | GAT − GM | Best GNN − GM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd_cdna4_mi350x` | 0.764164 | -0.391355 | -0.433707 | -0.540310 | 0.113365 | 0.047517 | -0.028790 | 0.032351 |
| `amd_cdna4_mi355x` | 0.153894 | 0.380998 | 0.527224 | 0.250401 | 0.187715 | 0.134995 | 0.130894 | 0.134995 |
| `intel_xe2_battlemage_arc_pro_b70` | 1.691060 | -0.047198 | -0.854615 | 0.059311 | -0.271460 | -0.539578 | 0.007528 | -0.332002 |
| `nvidia_blackwell_b200` | 0.808193 | -0.482562 | -0.507638 | -0.606598 | -0.504669 | -0.518696 | 0.019409 | -0.504669 |
| `nvidia_blackwell_b300` | 0.938501 | 0.157071 | 0.069513 | -0.141535 | -0.085575 | -0.143682 | 0.053598 | -0.082043 |
| `nvidia_blackwell_rtx_pro` | 1.287673 | -0.219716 | -0.023844 | 0.070724 | 0.379592 | 0.375373 | 0.250602 | 0.405425 |

### (b) Small/integrated families

| Family | Systems | Supported systems | Rows | Supported rows | Group median | Ridge | Random forest | MLP | GraphSAGE | GCN | GAT | Best GNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd_cdna3_mi300x` | 1 | 1 | 2 | 2 | 1.376473 | 0.810770 | 1.200972 | 1.236041 | 1.453123 | 1.435732 | 1.297602 | 1.490590 |
| `amd_rdna35_integrated_radeon_8060s` | 1 | 1 | 1 | 1 | 0.966204 | 0.742417 | 1.028630 | 0.695477 | 0.519781 | 0.571530 | 0.893041 | 0.535731 |
| `nvidia_blackwell_integrated_gb10` | 2 | 1 | 7 | 1 | 0.322527 | 0.315018 | 0.213816 | 0.281803 | 0.090855 | 0.162582 | 0.517604 | 0.003088 |
| `nvidia_blackwell_integrated_thor` | 1 | 1 | 1 | 1 | 0.631946 | 1.243431 | 1.143215 | 1.255449 | 1.582791 | 1.472277 | 0.838044 | 1.581338 |
| `nvidia_hopper_h200` | 1 | 1 | 2 | 2 | 1.010825 | 0.113516 | 0.791017 | 0.719870 | 0.334389 | 0.247812 | 1.333063 | 0.390006 |
| `nvidia_rubin_vr200` | 2 | 2 | 8 | 8 | 2.800540 | 1.114532 | 0.835638 | 1.026695 | 0.838571 | 0.795890 | 2.249656 | 0.701498 |

Paired differences, GNN minus MLP:
| Family | Supported systems | GraphSAGE − MLP | GCN − MLP | GAT − MLP | Best GNN − MLP |
| --- | ---: | --- | --- | --- | --- |
| `amd_cdna3_mi300x` | 1 | 0.217082; NA (too few systems) | 0.199691; NA (too few systems) | 0.061561; NA (too few systems) | 0.254550; NA (too few systems) |
| `amd_rdna35_integrated_radeon_8060s` | 1 | -0.175696; NA (too few systems) | -0.123947; NA (too few systems) | 0.197564; NA (too few systems) | -0.159746; NA (too few systems) |
| `nvidia_blackwell_integrated_gb10` | 1 | -0.190948; NA (too few systems) | -0.119221; NA (too few systems) | 0.235801; NA (too few systems) | -0.278715; NA (too few systems) |
| `nvidia_blackwell_integrated_thor` | 1 | 0.327342; NA (too few systems) | 0.216827; NA (too few systems) | -0.417405; NA (too few systems) | 0.325889; NA (too few systems) |
| `nvidia_hopper_h200` | 1 | -0.385481; NA (too few systems) | -0.472058; NA (too few systems) | 0.613193; NA (too few systems) | -0.329864; NA (too few systems) |
| `nvidia_rubin_vr200` | 2 | -0.188125; NA (too few systems) | -0.230805; NA (too few systems) | 1.222960; NA (too few systems) | -0.325197; NA (too few systems) |

Model differences from the group-median baseline:
| Family | Group median | Ridge − GM | Random forest − GM | MLP − GM | GraphSAGE − GM | GCN − GM | GAT − GM | Best GNN − GM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `amd_cdna3_mi300x` | 1.376473 | -0.565703 | -0.175501 | -0.140432 | 0.076649 | 0.059259 | -0.078871 | 0.114117 |
| `amd_rdna35_integrated_radeon_8060s` | 0.966204 | -0.223787 | 0.062426 | -0.270726 | -0.446423 | -0.394674 | -0.073162 | -0.430472 |
| `nvidia_blackwell_integrated_gb10` | 0.322527 | -0.007508 | -0.108711 | -0.040724 | -0.231672 | -0.159945 | 0.195078 | -0.319439 |
| `nvidia_blackwell_integrated_thor` | 0.631946 | 0.611484 | 0.511269 | 0.623503 | 0.950845 | 0.840330 | 0.206098 | 0.949392 |
| `nvidia_hopper_h200` | 1.010825 | -0.897309 | -0.219808 | -0.290955 | -0.676436 | -0.763013 | 0.322238 | -0.620819 |
| `nvidia_rubin_vr200` | 2.800540 | -1.686008 | -1.964902 | -1.773844 | -1.961969 | -2.004650 | -0.550884 | -2.099041 |

### (c) CPU-only Intel Xeon 6 stress case

| Family | Systems | Supported systems | Rows | Supported rows | Group median | Ridge | Random forest | MLP | GraphSAGE | GCN | GAT | Best GNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu_only_intel_xeon6_granite_rapids` | 10 | 10 | 33 | 28 | 2.707038 | 1.204246 | 2.041449 | 1.696620 | 1.277336 | 1.477151 | 2.994144 | 1.298226 |

Paired differences, GNN minus MLP:
| Family | Supported systems | GraphSAGE − MLP | GCN − MLP | GAT − MLP | Best GNN − MLP |
| --- | ---: | --- | --- | --- | --- |
| `cpu_only_intel_xeon6_granite_rapids` | 10 | -0.419285 [-0.851562, 0.078548] | -0.219469 [-0.679566, 0.116348] | 1.297523 [0.989712, 1.502870] | -0.398394 [-0.802894, -0.040203] |

Model differences from the group-median baseline:
| Family | Group median | Ridge − GM | Random forest − GM | MLP − GM | GraphSAGE − GM | GCN − GM | GAT − GM | Best GNN − GM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu_only_intel_xeon6_granite_rapids` | 2.707038 | -1.502793 | -0.665590 | -1.010418 | -1.429702 | -1.229887 | 0.287105 | -1.408812 |

All values are median `|ln(prediction / measured)|`; lower is better. System-cluster intervals are displayed only when at least 8 supported systems were tested. Families below that threshold retain point estimates and supported-system counts, with the interval reported as `NA (too few systems)`.

### Tail summaries

CV5 tail metrics are computed per partition after averaging neural seeds within row, then averaged across the five partitions. LOSO uses the pooled supported out-of-fold rows after averaging seeds within row. The CV5 maximum is the mean of the five per-partition maxima; the LOSO maximum is the pooled out-of-fold maximum. The ±25% window is `|ln(prediction/measured)| <= ln(1.25)`, equivalently `0.8*measured <= prediction <= 1.25*measured`. The ±50% window is `|ln(prediction/measured)| <= ln(1.5)`, equivalently `measured/1.5 <= prediction <= 1.5*measured`.

| Protocol | Method | Median | 75th | 90th | 95th | Max (CV5: mean of per-partition maxima; LOSO: pooled max) | Within ±25% | Within ±50% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cv5_partitioned | `random_forest` | 0.201853 | 0.566923 | 1.147904 | 1.824215 | 3.384369 | 52.519592% | 66.519667% |
| cv5_partitioned | `mlp` | 0.261071 | 0.588696 | 1.089365 | 1.415805 | 2.973499 | 44.997516% | 64.742552% |
| cv5_partitioned | `graphsage` | 0.370271 | 0.960684 | 1.765408 | 2.487238 | 3.757758 | 37.116327% | 52.506156% |
| loso | `random_forest` | 0.165012 | 0.497415 | 1.101461 | 1.588001 | 3.877578 | 57.709251% | 68.722467% |
| loso | `mlp` | 0.246446 | 0.494670 | 1.102937 | 1.359019 | 3.392854 | 47.136564% | 69.162996% |
| loso | `graphsage` | 0.338161 | 0.830410 | 1.627857 | 2.314123 | 3.901103 | 37.004405% | 56.828194% |

The medians reproduce the saved CV5 and LOSO primary errors. Random forest has the better median and a larger share within ±25%, but MLP has lower 90th and 95th percentile errors under both protocols.

### LOSO unsupported folds and mixed-model fallback

The six LOSO folds with no supported held-out row were:

| Fold | System |
| --- | --- |
| `holdout_000` | `datacenter|AMD|8xMI350X_2xEPYC_9575F|datacenter` |
| `holdout_049` | `datacenter|Intel|1-node-2S-Xeon_6787P|datacenter` |
| `holdout_098` | `edge|Dell|Dell Pro Max (1x GB10)|edge` |
| `holdout_099` | `edge|HPE|HPE ProLiant Compute DL345 Gen12 (1x NVIDIA RTX PRO 4500 Blackwell Server Edition 32GB TensorRT)|edge` |
| `holdout_101` | `edge|HPE|HPE ProLiant Compute DL380 Gen12 (1x NVIDIA RTX PRO 4500 Blackwell Server Edition 32GB)|edge` |
| `holdout_102` | `edge|HPE|HPE ProLiant Compute DL380 Gen12 (5x NVIDIA RTX PRO 4500 Blackwell Server Edition 32GB TensorRT)|edge` |

The hierarchical mixed-model Powell fallback occurred on `holdout_089` for system `datacenter|Supermicro|AS-4126GS-NMR-LCC|datacenter`. The replay converged with optimizer `powell` and produced finite predictions; its supported-row count was 7.

### Holm p-values versus equivalence labels

The Holm-adjusted p-value tests whether a paired secondary difference is zero, whereas the equivalence label evaluates whether the same 95% system-clustered interval lies entirely within the frozen `[-0.05, +0.05]` log-unit margin.

### Partial-pooling secondary analysis

Requested scope: the full CV5 and LOSO ladder. Status: `not_run` as a secondary analysis, logged under deviation `S3A-SHRINK-001`. Protocol v2.1 names a 'shrinkage posterior' but does not define the prior or posterior formula, the group-size shrinkage parameter, variance estimation, whether both center and scale are shrunk independently, or how the normalized primary error is evaluated after shrinkage. Do not invent an empirical-Bayes convention after results are known. Record the secondary analysis as not executable from the frozen specification and retain primary exclusion results.
No shrinkage performance number is reported because choosing a formula after observing the primary results would constitute a new exploratory analysis.

### LOAFO system coverage reconciliation

The 105 systems divide into 97 tested systems, two mixed systems kept in training for every family fold, and six systems whose held-out family rows were all unsupported:

| Family | System | Coverage status |
| --- | --- | --- |
| `amd_cdna4_mi350x` | `datacenter|AMD|8xMI350X_2xEPYC_9575F|datacenter` | `heldout_no_supported_rows` |
| `mixed_nvidia_hopper_h200_amd_cdna4_mi350x` | `datacenter|Cisco|Cisco UCS H200 x8 + MI350X x8 G200 2x2 TOR|datacenter` | `always_train_mixed` |
| `mixed_amd_cdna3_mi300x_cdna4_mi355x` | `datacenter|Dell_MangoBoost|Heterogeneous Four-Cluster (2x8x MI300X + 2x8x MI355X LLMBoost)|datacenter` | `always_train_mixed` |
| `intel_xe2_battlemage_arc_pro_b70` | `datacenter|Intel|1-node-2S-Xeon_6787P|datacenter` | `heldout_no_supported_rows` |
| `nvidia_blackwell_integrated_gb10` | `edge|Dell|Dell Pro Max (1x GB10)|edge` | `heldout_no_supported_rows` |
| `nvidia_blackwell_rtx_pro` | `edge|HPE|HPE ProLiant Compute DL345 Gen12 (1x NVIDIA RTX PRO 4500 Blackwell Server Edition 32GB TensorRT)|edge` | `heldout_no_supported_rows` |
| `nvidia_blackwell_rtx_pro` | `edge|HPE|HPE ProLiant Compute DL380 Gen12 (1x NVIDIA RTX PRO 4500 Blackwell Server Edition 32GB)|edge` | `heldout_no_supported_rows` |
| `nvidia_blackwell_rtx_pro` | `edge|HPE|HPE ProLiant Compute DL380 Gen12 (5x NVIDIA RTX PRO 4500 Blackwell Server Edition 32GB TensorRT)|edge` | `heldout_no_supported_rows` |

### Protocol record correction

`protocol.md` remains hash-pinned at Version 2.1. `protocol_history.md` adds Version 2.2 to record that the LOAFO models, seeds 7/11/16, omission of the mixed model, and family-level paired inference were specified in orchestration before the run rather than in the pinned file. It also records that the pinned phrase "LOAFO stress-test estimate" contains no pooling rule, so LOAFO results are descriptive.

The compute erratum uses 25 repeated-CV outer folds and 5 sensitivity outer folds. At the documented `0.36` seconds per fit, corrected totals are 6,210 fits and 2,235.6 seconds (37.26 minutes): repeated CV 2,375 fits; LOSO 3,360; sensitivity 475.

## 5. Open questions

- Should the thesis summarize family-level findings descriptively without attempting a pooled LOAFO claim, given the explicit instruction not to pool the three family sets?
- Should the author supply the missing shrinkage posterior and parameterization if a future exploratory partial-pooling analysis is desired?
- Should the B200 proposal-versus-actual supported-row difference be documented as a protocol-map audit note rather than changing the approved mapping?
- Should the CPU-only stress fold remain outside every headline claim, consistent with the frozen protocol?

These questions are for orchestrator review; no Phase 3 work or exploratory semantic block was started.

## 6. Acceptance criteria

| Item | Met? | Evidence |
| --- | --- | --- |
| Run LOAFO on approved 15-family map; mixed systems always train | Yes | Fold table and manifest; 13 non-mixed holdouts |
| Report headline, small/integrated, CPU-only separately; no pooling | Yes | Detailed LOAFO results |
| Systems, rows, supported rows, primary error per model | Yes | LOAFO family tables |
| Paired GNN-minus-MLP system-clustered intervals | Yes | LOAFO paired tables and `paired_differences_by_family.csv` |
| Flag B300 extreme extrapolation | Yes | Section 4 |
| Describe GB10/Thor and VR200 cases | Yes | Section 4 |
| Tail summaries for RF, MLP, GraphSAGE | Yes | Tail summary table and `tail_summary.csv` |
| Six empty LOSO folds and Powell system | Yes | LOSO special-case table and analysis bundle |
| Holm p-value versus equivalence sentence | Yes | Holm/equivalence subsection |
| Partial-pooling secondary handled or deviation logged | Yes | Not run; deviation `S3A-SHRINK-001` |
| Intervals only for families with >=8 supported systems | Yes | `loafo_family_support.csv` and family tables |
| LOAFO tested-system count recomputed and reconciled | Yes | 97 tested; full 105-system reconciliation |
| Full 21-test output and 8-test reconciliation | Yes | Appendix D |
| Tail labels, window definitions, and RF/MLP comparison | Yes | Tail summary subsection |
| Every NA has a resolve-or-do-not-cite decision | Yes | Appendix B and NA register |
| Headline best-GNN < MLP count and group-median comparisons | Yes | Detailed LOAFO tables and Section 4 |
| Paste protocol.md in full | Yes | Appendix A |
| Number-to-source map | Yes | Appendix E and accompanying CSV |
| Do not edit existing result bundles or thesis text | Yes | New bundles plus report/map only |
| Stop before Phase 3 or optional exploratory block | Yes | No Phase 3 or exploratory run performed |

Verification:

- `arch -arm64 python3 -m unittest tests.test_phase2 tests.test_folds`: 8 targeted tests passed; full discovery ran 21 tests (see Appendix D).
- LOAFO manifest: clean commit and matching protocol, data, family, system-assignment, and fold hashes.
- Analysis manifest: clean commit and saved source hashes.
- Corrections manifest: `e97bb939b3488766a17d4ad310f734c97944688e`; partial-pooling status `not_run`; test count `21`.

## Appendix A. Protocol v2.1 (full)

```text
# Phase 2 Pre-registration, Version 2.1

Frozen: 2026-11-07, before any Phase 2 model run.

This version supersedes Version 2 from commit `cce2f43`, whose file SHA-256 was
`9062c13cfe2ae50e9a5da2a0e92f48f8d2d2ef32191cd0d38942ac13b40296f4`.
Version 2 superseded Version 1 from commit `d3bc835`, SHA-256
`eb14e445f34c83ee817f59b35555f086be402d031534d4415815c575579fe811`.
The amendment log is `protocol_history.md`. Any future design change must be
committed and labeled exploratory unless it is made before any Phase 2 result
exists and is recorded as a new pre-registered version.

Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge, 476
published measurements from 105 submitter-aware system descriptions.

Canonical data SHA-256:
`a6853e1e2dc777404ddf9f5e0546679fef758df38bc4ebe5bba21e0c0bb735dd`.

Accelerator-family map SHA-256:
`f52b4021cadbec2582d5b2b5fa5931c146630db068b9ac4ba871b21bc1703c2d`.

## Data and identity

- `system_id = suite|submitter|system_name|system_type`.
- Each result row uses its scalar exported `total_accelerators` count.
- Every model uses `log1p(total_accelerators)` plus a CPU-only indicator.
  CPU-only is defined as `total_accelerators == 0`. No zero-accelerator row is
  silently dropped.
- The sorted tuple of reported counts is retained only for audit grouping. It
  is not a numeric feature and is not part of the sensitivity split key.
- The sensitivity split key is `submitter|accelerator|processor`.
- Empty categorical metadata is represented by `<unknown>`. Training masks
  categorical values to `<unknown>` with probability 0.1.
- All vocabulary, imputation, scaling, target normalization, graph, and
  baseline fitting uses training systems only. Validation systems are used for
  early stopping and the pre-registered best-GNN selection. Test systems enter
  only the inference graph after weights are fixed.

## Fold protocols and reuse

One saved fold file keyed by `system_id` is the source of every Phase 2 model
split: `folds/phase2_fold_assignments.csv`.

Co-primary protocols:

1. Grouped five-fold cross-validation repeated over five partitions using
   partition seeds `20261107`, `20261108`, `20261109`, `20261110`, and
   `20261111`. The seeds and the resulting assignments are recorded in the fold
   file.
2. Leave-one-system-out (LOSO), 105 folds, generated with seed `20261107` for
   its training-only validation selection.

Conservative check:

3. Five-fold sensitivity split on `submitter|accelerator|processor`, generated
   with seed `20265107`. This check is labeled modest because 78 of its 90
   groups are singleton systems.

Stress test:

4. Leave-one-accelerator-family-out (LOAFO), only after approval of the
   family table. Eligible families with at least three systems define the
   LOAFO stress-test estimate. Eligible smaller families are reported
   separately rather than pooled into that estimate. CPU-only is one
   execution-regime family. Mixed-accelerator systems are always in training
   and are never held out as a family.

The proposed 15-family map is
`folds/loafo_family_mapping_proposed.csv`, SHA-256
`1651770d0b6fd8d5c462bf1dd6f8b5fee5c44066cbf1fa1404d8d9745ffb3bd4`.
CPU-only is a labeled stress case outside the headline LOAFO estimate.
GB10 and Jetson Thor are explicit integrated-device families.

Within each outer fold, validation is selected only from outer-training
systems. Roles are disjoint. The five CV partitions reuse the same 105
systems but change the system-to-fold assignment.

## Evaluated set

For each outer fold, a target group is supported only if its training
partition has at least three rows and nonzero training standard deviation.
The supported row set is determined from that fold's training partition and is
identical across every model.

Unsupported test rows are excluded from the primary evaluated set and their
share is reported for every protocol. No global fallback normalization is used
in the primary analysis.

## Metrics

Primary error:

`median(|ln(prediction / measured)|)`

over supported out-of-fold rows. Lower is better.

Median-group Spearman uses the corrected thesis definition:

- a group enters only if it has at least three evaluated rows and at least two
  distinct measured values and two distinct predictions;
- singleton groups and two-row groups do not enter the median;
- the reported value is the median of the qualifying group-level Spearman
  correlations.

This is the definition used by the original Table 6.4 helper. It supersedes
the Phase 1 metric that admitted two-row groups.

Reported secondary metrics are:

- median row absolute percentage error;
- normalized RMSE on the signed within-group log target;
- median over target groups of the group median absolute log ratio;
- corrected median-group Spearman;
- coverage = evaluated supported rows / all 476 rows.

Pooled MAPE is an appendix diagnostic only. It is not used for model
selection, hypothesis testing, or the main ranking.

Neural stochastic models use seeds 7-16. For descriptive tables, report the
mean and SD of the ten seed-level values. For repeated CV, also report
partition-level variability.

Metric computation follows the protocol:

- repeated CV: compute the metric separately for each partition from its
  out-of-fold predictions; then average the five partition metrics;
- LOSO: compute each seed's metric from the pooled 105-fold out-of-fold
  predictions. Do not average 105 fold-level metrics as the headline.

Neural seeds for the repeated-CV core are 7-16. To control the LOSO cost, LOSO
uses three pre-specified neural seeds: **7, 11, and 16**. Sensitivity uses seeds
7-16. No seed selection may depend on results.

## System-clustered inference across partition and seed variability

Confidence intervals use 2,000 bootstrap resamples of whole systems. The same
selected systems are used for every model in a paired comparison.

Within each bootstrap draw, partition and seed variability is averaged inside
each selected system before pooling. Specifically:

1. Form each model's row prediction as the mean signed-log prediction across
   neural seeds.
2. For repeated CV, average that system's row contribution across the five
   partitions inside the draw.
3. For LOSO, use the single system-held-out prediction and average across
   seeds.
4. Compute the paired difference from the same system resample and draw.

Partitions or seeds are never treated as independent bootstrap units. Rows are
never resampled independently within a system.

## Compute estimate

The ladder has five deterministic model fits per outer fold (`group median`,
Ridge, random forest, gradient boosting, hierarchical mixed) and nine neural
fits per outer fold and seed (MLP, GraphSAGE, GCN, GAT, shuffled-edge
GraphSAGE, and four relation-removal GraphSAGE models).

| Protocol | Outer folds | Neural seeds | Deterministic fits | Neural fits | Total fits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5-fold x 5 partitions | 125 | 10 | 625 | 11,250 | 11,875 |
| LOSO | 105 | 3 | 525 | 2,835 | 3,360 |
| sensitivity | 25 | 10 | 125 | 2,250 | 2,375 |
| Total | 255 | - | 1,275 | 16,335 | 17,610 |

The Phase 1.5 audit measured 120.05 seconds for 300 GraphSAGE fold fits plus
regression, tests, and orchestration. Using the conservative planning rate
`0.36` seconds per fit gives:

| Protocol | Estimated compute |
| --- | ---: |
| 5-fold x 5 partitions | 4,275 seconds, 71.25 minutes |
| LOSO, three seeds | 1,209.6 seconds, 20.16 minutes |
| sensitivity, ten seeds | 855 seconds, 14.25 minutes |
| Total | 6,339.6 seconds, 105.66 minutes |

This is a planning estimate based on one GNN's measured runtime. It does not
claim equal fitting cost across model families. With the ten-seed LOSO
alternative, LOSO would contain 9,975 fits and the total would be 24,225 fits.

## Model fitting policy

Policy: no tuning for any model. The same fixed budget applies to every model.

- Group median: within-fold training median of the signed log target.
- Ridge: alpha `10.0`.
- Random forest: 300 trees, minimum leaf size 2, maximum features `0.7`,
  `random_state=7`.
- Gradient boosting: 300 iterations, learning rate `0.05`, maximum 15 leaves,
  L2 regularization `1.0`, `random_state=7`.
- MLP: the same metadata features and learned categorical identifier
  embeddings, no edges; two 64-unit hidden layers, dropout `0.2`, AdamW
  learning rate `0.01`, weight decay `1e-4`, Huber loss delta `1.0`, at most
  150 epochs, patience 20.
- GraphSAGE, GCN, and GAT: metadata graph with real edges; hidden size 64, two
  layers, dropout `0.2`, AdamW learning rate `0.01`, weight decay `1e-4`,
  Huber loss delta `1.0`, at most 150 epochs, patience 20. GAT uses four
  attention heads of 16 channels.
- Shuffled graph: GraphSAGE with degree-preserving shuffled directed edges.
  Per relation, source out-degrees and destination in-degrees are retained
  exactly.
- Relation removal: separate GraphSAGE models remove every edge incident to
  one metadata node type in turn: system, accelerator, processor, or workload.
  Result nodes and result numeric features remain.
- Hierarchical mixed model: fixed effects for workload target group,
  `log1p(total_accelerators)`, CPU-only indicator, and accelerator memory;
  random intercepts for accelerator family, submitter, and system. An unseen
  test system receives a zero random intercept and contributes no variance
  estimation.

Learned identifier embedding width is fixed at
`min(16, max(2, round(sqrt(category_count))))` per categorical identifier,
fitted from training systems only. Values absent from training use the trained
`<unknown>` representation.

Within-group target center and scale come from the fold's training partition.
Groups with fewer than three training rows or no defined scale are unsupported
in the primary analysis.

## Single primary hypothesis

The sole primary hypothesis is:

`primary error(best GNN) - primary error(MLP)`

under both co-primary protocols. Lower primary error is better.

Best GNN is selected without test labels, separately for each outer fold and
neural seed:

- candidates are GraphSAGE, GCN, and GAT using the fixed fitting policy above;
- validation primary error is computed on the outer fold's validation systems;
- the candidate with lowest validation primary error is selected;
- ties use the fixed order GraphSAGE, GCN, GAT.

The selected architecture supplies the test predictions for that fold and
seed. For LOSO, the same rule is applied using only the outer-training and
validation systems.

The equivalence margin is `+/-0.05` log units on the paired difference.
Classification uses the 95% system-clustered paired interval:

| Interval position relative to `[-0.05, +0.05]` | Label |
| --- | --- |
| Entire interval below `-0.05` | better |
| Entire interval above `+0.05` | worse |
| Entire interval inside `[-0.05, +0.05]`, whether or not it excludes 0 | equivalent |
| Crosses either boundary, including intervals that exclude 0 but straddle a boundary | inconclusive |

An interval entirely inside the margin but excluding zero is still
`equivalent`: the equivalence decision concerns practical magnitude, not
whether zero is inside the interval. An interval outside zero but crossing
`-0.05` or `+0.05` is `inconclusive`.

Because there are two co-primary protocols, both must produce the same label
before a primary claim is made. The hypothesis is called better only if both
intervals are entirely below `-0.05`; equivalent only if both are entirely
inside the margin; worse only if both are entirely above `+0.05`. Any
disagreement is `inconclusive`. There is no multiplicity correction for the
single primary hypothesis.

## Secondary comparisons

The fixed GraphSAGE model is the reference for these secondary comparisons:

1. GraphSAGE - random forest.
2. GraphSAGE real edges - GraphSAGE degree-preserving shuffled edges.
3. GCN - GraphSAGE.
4. GAT - GraphSAGE.

Each comparison is evaluated under both co-primary protocols using the same
system-clustered paired interval. Holm correction is applied across the eight
protocol-comparison tests. Adjusted results are labeled secondary. A
zero-containing interval is `not distinguishable`; it is not equivalence.

All other comparisons, subgroup analyses, pooled MAPE, and post hoc model
choices are exploratory.

## Sparse-group secondary comparison

Primary analysis excludes unsupported sparse groups. A pre-specified secondary
analysis uses partial pooling normalization: target-group center and scale are
shrunk toward global training values according to training group size. The
same shrinkage posterior is used for every model, and test labels never enter
the estimate. Results are labeled secondary and never replace primary
exclusions.

## Deliverables

- Separate main results for repeated CV and LOSO.
- All models by primary error, median row error, corrected median-group
  Spearman, normalized RMSE, and coverage, each with clustered 95% intervals.
- Partition-specific CV results.
- The primary best-GNN minus MLP decision table.
- The four Holm-corrected secondary comparisons under both co-primary
  protocols.
- Per-target-group results for every model in an appendix.
- Per-fold support and exclusion counts.
- Pooled MAPE in an appendix only.
- A manifest with commit, seeds, fold hash, data hash, and output hashes.

## Semantic-layer boundary

The later semantic-layer experiment is not part of Phase 2. It requires the
author's mapping file and a separate pre-registration. Measured performance
must not define semantic clusters unless the mapping is refit inside each
training fold.
```

## Appendix B. Every NA in the number-to-source map

There are 146 NA entries. The full register is also saved as `thesis_overleaf/phase2_5_na_register.csv`.

| Map ID | Context | Decision | Rationale |
| --- | --- | --- | --- |
| N0128 | cv5_partitioned main table, gradient_boosting, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0129 | cv5_partitioned main table, gradient_boosting, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0130 | cv5_partitioned main table, gradient_boosting, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0131 | cv5_partitioned main table, gradient_boosting, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0132 | cv5_partitioned main table, gradient_boosting, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0344 | cv5_partitioned main table, group_median, median_group_spearman_mean | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0359 | cv5_partitioned main table, group_median, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0360 | cv5_partitioned main table, group_median, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0361 | cv5_partitioned main table, group_median, median_group_spearman_seed_sd | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0362 | cv5_partitioned main table, group_median, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0363 | cv5_partitioned main table, group_median, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0369 | cv5_partitioned main table, group_median, median_group_spearman_95ci_low | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0370 | cv5_partitioned main table, group_median, median_group_spearman_95ci_high | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0392 | cv5_partitioned main table, hierarchical_mixed, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0393 | cv5_partitioned main table, hierarchical_mixed, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0394 | cv5_partitioned main table, hierarchical_mixed, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0395 | cv5_partitioned main table, hierarchical_mixed, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0396 | cv5_partitioned main table, hierarchical_mixed, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0458 | cv5_partitioned main table, random_forest, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0459 | cv5_partitioned main table, random_forest, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0460 | cv5_partitioned main table, random_forest, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0461 | cv5_partitioned main table, random_forest, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0462 | cv5_partitioned main table, random_forest, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0491 | cv5_partitioned main table, ridge, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0492 | cv5_partitioned main table, ridge, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0493 | cv5_partitioned main table, ridge, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0494 | cv5_partitioned main table, ridge, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0495 | cv5_partitioned main table, ridge, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0521 | loso main table, best_gnn, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0522 | loso main table, best_gnn, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0548 | loso main table, gat, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0549 | loso main table, gat, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0575 | loso main table, gcn, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0576 | loso main table, gcn, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0595 | loso main table, gradient_boosting, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0597 | loso main table, gradient_boosting, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0599 | loso main table, gradient_boosting, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0601 | loso main table, gradient_boosting, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0602 | loso main table, gradient_boosting, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0603 | loso main table, gradient_boosting, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0629 | loso main table, graphsage, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0630 | loso main table, graphsage, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0656 | loso main table, graphsage_no_accelerator, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0657 | loso main table, graphsage_no_accelerator, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0683 | loso main table, graphsage_no_processor, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0684 | loso main table, graphsage_no_processor, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0710 | loso main table, graphsage_no_system, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0711 | loso main table, graphsage_no_system, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0737 | loso main table, graphsage_no_workload, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0738 | loso main table, graphsage_no_workload, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0764 | loso main table, graphsage_shuffled, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0765 | loso main table, graphsage_shuffled, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0779 | loso main table, group_median, median_group_spearman | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0784 | loso main table, group_median, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0786 | loso main table, group_median, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0787 | loso main table, group_median, median_group_spearman_seed_mean | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0788 | loso main table, group_median, median_group_spearman_seed_sd | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0790 | loso main table, group_median, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0791 | loso main table, group_median, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0792 | loso main table, group_median, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0798 | loso main table, group_median, median_group_spearman_95ci_low | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0799 | loso main table, group_median, median_group_spearman_95ci_high | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N0811 | loso main table, hierarchical_mixed, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0813 | loso main table, hierarchical_mixed, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0815 | loso main table, hierarchical_mixed, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0817 | loso main table, hierarchical_mixed, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0818 | loso main table, hierarchical_mixed, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0819 | loso main table, hierarchical_mixed, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0845 | loso main table, mlp, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0846 | loso main table, mlp, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0865 | loso main table, random_forest, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0867 | loso main table, random_forest, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0869 | loso main table, random_forest, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0871 | loso main table, random_forest, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0872 | loso main table, random_forest, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0873 | loso main table, random_forest, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0892 | loso main table, ridge, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0894 | loso main table, ridge, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0896 | loso main table, ridge, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0898 | loso main table, ridge, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0899 | loso main table, ridge, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0900 | loso main table, ridge, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N0926 | sensitivity main table, best_gnn, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0927 | sensitivity main table, best_gnn, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0953 | sensitivity main table, gat, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0954 | sensitivity main table, gat, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0980 | sensitivity main table, gcn, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N0981 | sensitivity main table, gcn, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1000 | sensitivity main table, gradient_boosting, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1002 | sensitivity main table, gradient_boosting, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1004 | sensitivity main table, gradient_boosting, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1006 | sensitivity main table, gradient_boosting, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1007 | sensitivity main table, gradient_boosting, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1008 | sensitivity main table, gradient_boosting, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1034 | sensitivity main table, graphsage, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1035 | sensitivity main table, graphsage, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1061 | sensitivity main table, graphsage_no_accelerator, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1062 | sensitivity main table, graphsage_no_accelerator, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1088 | sensitivity main table, graphsage_no_processor, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1089 | sensitivity main table, graphsage_no_processor, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1115 | sensitivity main table, graphsage_no_system, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1116 | sensitivity main table, graphsage_no_system, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1142 | sensitivity main table, graphsage_no_workload, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1143 | sensitivity main table, graphsage_no_workload, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1169 | sensitivity main table, graphsage_shuffled, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1170 | sensitivity main table, graphsage_shuffled, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1184 | sensitivity main table, group_median, median_group_spearman | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N1189 | sensitivity main table, group_median, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1191 | sensitivity main table, group_median, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1192 | sensitivity main table, group_median, median_group_spearman_seed_mean | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N1193 | sensitivity main table, group_median, median_group_spearman_seed_sd | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N1195 | sensitivity main table, group_median, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1196 | sensitivity main table, group_median, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1197 | sensitivity main table, group_median, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1203 | sensitivity main table, group_median, median_group_spearman_95ci_low | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N1204 | sensitivity main table, group_median, median_group_spearman_95ci_high | `RESOLVE_AS_NA_BY_DESIGN` | Group-median predictions have no within-group rank variation. |
| N1216 | sensitivity main table, hierarchical_mixed, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1218 | sensitivity main table, hierarchical_mixed, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1220 | sensitivity main table, hierarchical_mixed, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1222 | sensitivity main table, hierarchical_mixed, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1223 | sensitivity main table, hierarchical_mixed, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1224 | sensitivity main table, hierarchical_mixed, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1250 | sensitivity main table, mlp, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1251 | sensitivity main table, mlp, coverage_seed_sd | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1270 | sensitivity main table, random_forest, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1272 | sensitivity main table, random_forest, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1274 | sensitivity main table, random_forest, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1276 | sensitivity main table, random_forest, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1277 | sensitivity main table, random_forest, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1278 | sensitivity main table, random_forest, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1297 | sensitivity main table, ridge, primary_error_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1299 | sensitivity main table, ridge, median_row_mape_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1301 | sensitivity main table, ridge, median_group_spearman_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1303 | sensitivity main table, ridge, normalized_rmse_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1304 | sensitivity main table, ridge, coverage_seed_mean | `DO_NOT_CITE` | Coverage is fixed by fold support; use the overall coverage column. |
| N1305 | sensitivity main table, ridge, coverage_seed_sd | `DO_NOT_CITE` | Deterministic methods have one fixed fit and no seed SD. |
| N1872 | CV5 Spearman summary, row 10, median_group_spearman_partition_mean | `RESOLVE_AS_NA_BY_DESIGN` | The summary row is forced NA for the group-median baseline. |
| N1929 | CV5 top normalized residual, row 10, result | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N1930 | CV5 top normalized residual, row 10, prediction | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N1932 | CV5 top normalized residual, row 10, abs_z | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N1973 | CV5 top normalized residual, row 21, result | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N1974 | CV5 top normalized residual, row 21, prediction | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N1976 | CV5 top normalized residual, row 21, abs_z | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N2017 | CV5 top normalized residual, row 32, result | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N2018 | CV5 top normalized residual, row 32, prediction | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |
| N2020 | CV5 top normalized residual, row 32, abs_z | `DO_NOT_CITE_NULL_CELL` | Summary rows have no row-specific residual value; use numbered rows. |

## Appendix C. Protocol v2.2 history addendum

The complete addendum and erratum are in `mlperf_gnn/protocol_history.md`. The hash-pinned `protocol.md` was not edited.

## Appendix D. Full test-suite output

The earlier report listed the targeted Phase 2 and fold modules only: 5 Phase 2 tests plus 3 fold tests, totaling 8. Full discovery also ran 3 data, 8 leakage, and 2 metric tests, for 21 overall:

| Module | Tests |
| --- | ---: |
| `test_data` | 3 |
| `test_folds` | 3 |
| `test_leakage` | 8 |
| `test_metrics` | 2 |
| `test_phase2` | 5 |

Full output:

```text
test_multiple_count_signatures_are_retained_for_grouping (test_data.TestSystemTable.test_multiple_count_signatures_are_retained_for_grouping) ... ok
test_sensitivity_key_excludes_accelerator_counts (test_data.TestSystemTable.test_sensitivity_key_excludes_accelerator_counts) ... ok
test_total_accelerators_remain_the_numeric_feature (test_data.TestSystemTable.test_total_accelerators_remain_the_numeric_feature) ... ok
test_five_repeated_partitions_cover_every_system (test_folds.TestPhase2Folds.test_five_repeated_partitions_cover_every_system) ... ok
test_loso_and_sensitivity_are_present (test_folds.TestPhase2Folds.test_loso_and_sensitivity_are_present) ... ok
test_partitions_are_distinct (test_folds.TestPhase2Folds.test_partitions_are_distinct) ... ok
test_baseline_fit_never_receives_test_rows_for_every_fold (test_leakage.TestEveryFoldIsolation.test_baseline_fit_never_receives_test_rows_for_every_fold) ... ok
test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold (test_leakage.TestEveryFoldIsolation.test_no_test_system_graph_scaler_vocabulary_or_baseline_for_every_fold) ... ok
test_roles_are_disjoint_for_every_fold (test_leakage.TestEveryFoldIsolation.test_roles_are_disjoint_for_every_fold) ... ok
test_unknown_vectors_are_exercised_during_training (test_leakage.TestEveryFoldIsolation.test_unknown_vectors_are_exercised_during_training) ... ok
test_production_weights_ignore_test_metadata_on_five_fold_cases (test_leakage.TestFiveFoldWeightBoundary.test_production_weights_ignore_test_metadata_on_five_fold_cases) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
ok
test_loss_masks_exclude_unsupported_rows_without_isolation (test_leakage.TestSupportedLeakyPolicy.test_loss_masks_exclude_unsupported_rows_without_isolation) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
./tests/test_leakage.py:277: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  self.assertTrue(np.isfinite(result.trained_weights["head.bias"]).all())
ok
test_production_weights_ignore_scrambled_test_metadata (test_leakage.TestWeightPerturbation.test_production_weights_ignore_scrambled_test_metadata) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
ok
test_reintroduced_leak_fails_the_production_weight_check (test_leakage.TestWeightPerturbation.test_reintroduced_leak_fails_the_production_weight_check) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
ok
test_paired_difference_uses_same_system_draw (test_metrics.TestClusteredBootstrap.test_paired_difference_uses_same_system_draw) ... ok
test_two_row_groups_are_excluded_by_default (test_metrics.TestMedianGroupSpearman.test_two_row_groups_are_excluded_by_default) ... ok
test_core_neural_path_omits_secondary_graph_variants (test_phase2.TestPhase2Fold.test_core_neural_path_omits_secondary_graph_variants) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
ok
test_deterministic_ladder_is_finite (test_phase2.TestPhase2Fold.test_deterministic_ladder_is_finite) ... ok
test_deterministic_subset_omits_unrequested_models (test_phase2.TestPhase2Fold.test_deterministic_subset_omits_unrequested_models) ... ok
test_graph_diagnostic_reports_training_stop_state (test_phase2.TestPhase2Fold.test_graph_diagnostic_reports_training_stop_state) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
ok
test_neural_ladder_and_best_gnn_are_finite (test_phase2.TestPhase2Fold.test_neural_ladder_and_best_gnn_are_finite) ... /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/torch_geometric/inspector.py:433: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated, as it leads to incorrect behaviour when calling typing._eval_type on a stringified annotation that references a PEP 695 type parameter. It will be disallowed in Python 3.15.
  return typing._eval_type(value, _globals, None)  # type: ignore
ok

----------------------------------------------------------------------
Ran 21 tests in 15.529s

OK
```

## Appendix E. Number-to-source map

The exhaustive machine-readable map is `thesis_overleaf/phase2_5_number_source_map.csv`. Each row identifies the displayed number, context, results file, source column, row filter, and derivation. Direct values use `direct`; percentages and counts state their transformation. The `na_decision` and `na_rationale` columns classify every NA.

Key bundles:

- LOAFO manifest commit: `1a65664dc5b25d13db72656cb10ad14acf2c7811`.
- LOAFO prediction hash: `55bb110d65e68444dd5138c576e733059d4ad6c660a20bd2d9d5aad204fd43c8`.
- Tail/LOSO diagnostic manifest commit: `4faac06eec7951706571e8d1943085c423bfbd2b`.
- Tail summary hash: `d22edf72aae46bdfa326ca5f502140ab10793cdf659a293c4d5f120e42236660`.
- Corrections manifest commit: `e97bb939b3488766a17d4ad310f734c97944688e`.

