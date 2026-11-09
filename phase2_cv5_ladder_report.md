# Phase 2 Repeated-CV Stop Report

Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge.

This report stops after the first complete model ladder on the five
repeated five-fold partitions. LOSO and sensitivity have not run.

## 1. What was done

- Reconciled the seed-7 legacy/fixed table by model key, fold scheme,
  evaluated rows, and coverage, then reran both policies on the same
  supported rows.
- Froze Protocol v2.1 before the successful Phase 2 run.
- Verified the two 288-accelerator GB300 rows against the source export.
- Extended production-weight equality to five selected folds.
- Generated the proposed 15-family LOAFO map without fitting LOAFO.
- Ran 5 partitions × 5 folds × 15 methods, with ten seeds for each
  stochastic model, then computed system-clustered bootstrap summaries.
- Stopped before LOSO and sensitivity.

## 2. Files changed

Code commit `0291baa` froze Protocol v2.1. The retained core predictions
were generated at clean commit `8129f44`. The final aggregation bundle
was generated at clean commit `aeeb4b4`.

Canonical outputs:

- Phase 2 core:
  `mlperf_gnn/results/phase2-cv5-final-aeeb4b4/`
- Pre-run checks:
  `mlperf_gnn/results/phase2-prerun-1122699/`
- Fold map: `mlperf_gnn/folds/phase2_fold_assignments.csv`
- Protocol v2.1: `mlperf_gnn/protocol.md`
- Protocol V2-to-V2.1 diff:
  `mlperf_gnn/protocol_v2_to_v2_1.diff`
- Proposed LOAFO families:
  `mlperf_gnn/folds/loafo_family_mapping_proposed.csv`
- Full proposed system map:
  `mlperf_gnn/folds/loafo_system_assignment_proposed.csv`

No thesis source (`main.tex`), PDF, or bibliography file was edited.

## 3. Numbers that changed

### A. Gate 3 seed-7 reconciliation

The Gate 3 table mixed two fold contexts and one metric-filtering rule.

| Context | Fold scheme | Fold seed | Model seed | Prediction rows | Evaluated rows | Coverage | Primary error |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `rerun_Gate3_attribution_baseline_seed7_all_rows` | legacy 96-system random folds | 7 | 7 | 476 | 476 | 100.0000% | 0.368149473 |
| `Phase1_legacy_seed7_full_predictions` | Phase 1 canonical cv5 | 20261107 | 7 | 476 | 476 | 100.0000% | 0.515773273 |
| `Phase1_legacy_seed7_identical_supported` | Phase 1 canonical cv5 | 20261107 | 7 | 476 | 450 | 94.5378% | 0.485145847 |
| `Phase1_legacy_seeds7to16_identical_supported` | Phase 1 canonical cv5 | 20261107 | 7-16 | 476 | 450 | 94.5378% | 0.465427134 |
| `Phase1_fixed_seed7_full_predictions` | Phase 1 canonical cv5 | 20261107 | 7 | 450 | 450 | 94.5378% | 0.510997149 |
| `Phase1_fixed_seeds7to16_identical_supported` | Phase 1 canonical cv5 | 20261107 | 7-16 | 450 | 450 | 94.5378% | 0.462313987 |
| `rerun_legacy_leaky_global_seed7_identical_supported` | Phase 1 canonical cv5 | 20261107 | 7 | 476 | 450 | 94.5378% | 0.485145847 |
| `rerun_isolated_supported_seed7_identical_supported` | Phase 1 canonical cv5 | 20261107 | 7 | 450 | 450 | 94.5378% | 0.510997149 |

`0.368149` comes from the Phase 1.5 attribution baseline: legacy
96-system key, legacy seed-7 random folds, seed 7, all 476 rows,
100% coverage. It is not the Phase 1 `legacy_leaky_global` result.

The Phase 1 canonical split holds out 22/21/21/21/20 systems and
produces 105/77/98/98/98 test rows; training-partition support leaves
103/72/93/87/95 evaluated rows, 450 total. Model seeds are 7-16.

Phase 1 `legacy_leaky_global` produced all 476 predictions per
seed, but its reported primary metric was calculated after filtering
to the same 450 supported rows. Thus its 10-seed mean `0.465427`
was already on the identical evaluated set. Fixed isolation produced
450 predictions and evaluated all of them. Its 10-seed mean was
`0.462314`.

Seed-7 identical-set rerun: legacy `0.485146`, fixed `0.510997`.
The large-looking `0.368149` versus `0.510997` gap is therefore not
a legitimate policy comparison.

### B. Protocol v2.1

- SHA-256: `d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e`
- Frozen commit: `0291baa`
- Supersedes V2 hash:
  `9062c13cfe2ae50e9a5da2a0e92f48f8d2d2ef32191cd0d38942ac13b40296f4`
- V2-to-V2.1 diff SHA-256:
  `e8f87e52f920423c52247f6f8ef4357d382a4c847c1e65a955cbe3040f307b53`
- Fold file SHA-256:
  `3ed4dd0bf019685255c6f6d11f309d10f147280487617cf8f49c7c437e433443`

The full protocol is pasted at the end of this report.

The frozen compute plan was 17,610 fits: 11,875 for 5×5, 3,360
for three-seed LOSO, and 2,375 for ten-seed sensitivity. At the
documented 0.36 s/fit rate, the estimates were 71.25, 20.16, and
14.25 minutes, totaling 105.66 minutes.

The completed 5×5 partition-fitting wall times summed to 661.392760 seconds (11.023213 minutes). The five run
manifests are retained in their partition directories.

### C. GB300 NVL72 source verification

Source file SHA-256:
`8b5ccfe2143281e2ae266de03fbec8e57e69b86aeaa5938de0e578fef47d2d1f`.

All 13 canonical GB300 rows match the source:
`nodes × accelerators_per_node = Total Accelerators1`.

| Public ID | Nodes | Accelerators/node | Product | Reported total |
| --- | ---: | ---: | ---: | ---: |
| `6.1-0074` | 18 | 4 | 72 | 72 |
| `6.1-0073` | 72 | 4 | 288 | 288 |

The 72-versus-288 difference is the source-reported topology:
public ID `6.1-0074` reports 18 nodes × 4 accelerators = 72, while
`6.1-0073` reports 72 nodes × 4 accelerators = 288. Both rows in
the latter submission (`row:0042`, `row:0343`) therefore use 288.

### D. Production weight equality

The production check covers five folds: CV Edge-containing fold
`p0_fold_0`, CV fold `p0_fold_1`, LOSO Edge system `holdout_104`,
LOSO Datacenter system `holdout_001`, and sensitivity Edge-containing
`fold_0`. All use 150 epochs and patience 20.

Result: PASS. The test ran in 2.99 seconds inside a 5.45-second
process. For every fold, scrambling test metadata left every saved
weight tensor exactly equal.

The earlier 130-fold loop used `epochs=1, patience=1`. That loop was
removed. All 130 folds retain structural isolation checks; the full
production weight comparison covers the five selected folds.

### E. LOAFO family map

No LOAFO model was run. The proposed table has 15 families covering
105 systems and 476 rows.

| Family | Vendor | Architecture generation | Class | Systems | Rows | Supported rows if held out | Disposition |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `amd_cdna3_mi300x` | AMD | CDNA3 MI300X | discrete | 1 | 2 | 2 | small_family_reported_separately |
| `amd_cdna4_mi350x` | AMD | CDNA4 MI350X | discrete | 4 | 24 | 20 | headline_holdout |
| `amd_cdna4_mi355x` | AMD | CDNA4 MI355X | discrete | 12 | 66 | 61 | headline_holdout |
| `amd_rdna35_integrated_radeon_8060s` | AMD | RDNA 3.5 integrated Radeon 8060S | integrated-device | 1 | 1 | 1 | small_family_reported_separately |
| `cpu_only_intel_xeon6_granite_rapids` | Intel | Xeon 6 Granite Rapids | cpu-only | 10 | 33 | 28 | stress_case_outside_headline |
| `intel_xe2_battlemage_arc_pro_b70` | Intel | Xe2 Battlemage Arc Pro B70 | discrete | 4 | 19 | 17 | headline_holdout |
| `mixed_amd_cdna3_mi300x_cdna4_mi355x` | AMD | CDNA3 MI300X + CDNA4 MI355X | mixed | 1 | 2 | NA | always_train |
| `mixed_nvidia_hopper_h200_amd_cdna4_mi350x` | NVIDIA + AMD | Hopper H200 + CDNA4 MI350X | mixed | 1 | 8 | NA | always_train |
| `nvidia_blackwell_b200` | NVIDIA | Blackwell B200 | discrete | 14 | 47 | 47 | headline_holdout |
| `nvidia_blackwell_b300` | NVIDIA | Blackwell B300 | discrete | 33 | 198 | 198 | headline_holdout |
| `nvidia_blackwell_integrated_gb10` | NVIDIA | Blackwell integrated GB10 | integrated-device | 2 | 7 | 1 | small_family_reported_separately |
| `nvidia_blackwell_integrated_thor` | NVIDIA | Blackwell integrated Jetson Thor | integrated-device | 1 | 1 | 1 | small_family_reported_separately |
| `nvidia_blackwell_rtx_pro` | NVIDIA | Blackwell RTX PRO | discrete | 18 | 58 | 50 | headline_holdout |
| `nvidia_hopper_h200` | NVIDIA | Hopper H200 | discrete | 1 | 2 | 2 | small_family_reported_separately |
| `nvidia_rubin_vr200` | NVIDIA | Rubin VR200 | discrete | 2 | 8 | 8 | small_family_reported_separately |

Headline LOAFO-eligible families with at least three systems are
B300, B200, RTX PRO, MI355X, MI350X, and Intel Arc Pro B70.
MI300X, H200, Radeon 8060S, GB10, Thor, and VR200 are reported
separately because each has fewer than three systems. CPU-only is
one labeled stress case outside the headline. The two mixed systems
are always-train.

### Phase 2 completed: repeated five-fold CV

Point metrics pool all five folds within each partition, average the
five partition metrics, and average neural seeds within row before
partition metrics. Intervals use 2,000 system-cluster resamples with
partition and seed variability averaged inside each draw.

| Method | Primary error [95% CI] | Partition SD | Seed SD | Median row MAPE | Median-group Spearman | Normalized RMSE | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_forest` | 0.201853 [0.142507, 0.270546] | 0.032297 | NA | 0.194472 | 0.793714 | 1.593019 | 94.4958% |
| `mlp` | 0.261071 [0.211858, 0.321888] | 0.019410 | 0.014539 | 0.256486 | 0.815404 | 1.536847 | 94.4958% |
| `ridge` | 0.331671 [0.278574, 0.391696] | 0.034420 | NA | 0.302672 | 0.827641 | 1.587093 | 94.4958% |
| `gradient_boosting` | 0.336582 [0.272759, 0.432367] | 0.033583 | NA | 0.332135 | 0.740296 | 1.618263 | 94.4958% |
| `graphsage_shuffled` | 0.341685 [0.276537, 0.452507] | 0.096897 | 0.012989 | 0.336813 | 0.643818 | 1.666956 | 94.4958% |
| `graphsage_no_system` | 0.348254 [0.278640, 0.443082] | 0.082742 | 0.021557 | 0.334383 | 0.658244 | 1.661722 | 94.4958% |
| `graphsage_no_processor` | 0.353855 [0.284499, 0.460453] | 0.095532 | 0.016883 | 0.339056 | 0.627766 | 1.661496 | 94.4958% |
| `gcn` | 0.354737 [0.276503, 0.455569] | 0.087236 | 0.016273 | 0.340548 | 0.618737 | 1.657707 | 94.4958% |
| `graphsage_no_workload` | 0.358888 [0.289938, 0.443866] | 0.101939 | 0.017669 | 0.337749 | 0.635247 | 1.664493 | 94.4958% |
| `graphsage` | 0.370271 [0.294955, 0.479295] | 0.093310 | 0.020027 | 0.352613 | 0.634863 | 1.666696 | 94.4958% |
| `graphsage_no_accelerator` | 0.370623 [0.288840, 0.470387] | 0.103640 | 0.010088 | 0.342877 | 0.623835 | 1.662527 | 94.4958% |
| `best_gnn` | 0.376393 [0.291878, 0.489192] | 0.105183 | 0.010730 | 0.353617 | 0.619981 | 1.666412 | 94.4958% |
| `hierarchical_mixed` | 0.506282 [0.468577, 0.547471] | 0.026267 | NA | 0.406506 | 0.706526 | 1.639144 | 94.4958% |
| `group_median` | 0.617066 [0.332169, 0.874370] | 0.048762 | NA | 0.618776 | -0.297001 | 1.903985 | 94.4958% |
| `gat` | 0.632643 [0.473434, 0.881098] | 0.062715 | 0.025418 | 0.631880 | 0.004719 | 1.881176 | 94.4958% |

Pooled MAPE remains appendix-only. Full precision for every metric
and interval is in `main_table.csv`; per-group results are in
`per_group_results.csv`.

Best-GNN validation selection counts across 250 fold-seed choices:

| Selected GNN | Count |
| --- | ---: |
| `gat` | 13 |
| `gcn` | 139 |
| `graphsage` | 98 |

### Paired differences

| Left minus right | Role | Difference [95% CI] | Status | Holm |
| --- | --- | --- | --- | --- |
| `best_gnn - mlp` | primary | 0.115597 [0.047136, 0.198051] | inconclusive | not_applicable |
| `graphsage - random_forest` | secondary | 0.171529 [0.111863, 0.237339] | worse | pending_LOSO |
| `graphsage - graphsage_shuffled` | secondary | 0.024523 [0.004590, 0.045019] | equivalent | pending_LOSO |
| `gcn - graphsage` | secondary | -0.016638 [-0.045146, 0.007482] | equivalent | pending_LOSO |
| `gat - graphsage` | secondary | 0.267067 [0.145077, 0.450114] | worse | pending_LOSO |

The primary interval excludes zero but crosses the `+0.05`
equivalence boundary, so the repeated-CV label is `inconclusive`.
The co-primary claim cannot be decided until LOSO. Holm-adjusted
secondary results also remain pending because the frozen Holm family
includes both co-primary protocols.

## 4. Anything surprising or contradicting the current narrative

- Random forest is the best primary-error method (`0.201853`).
- MLP is second (`0.261071`) and outperforms every GNN.
- GraphSAGE is `0.370271`, and the validation-selected best GNN is
  `0.376393`.
- GAT is the worst method (`0.632643`) and is selected least often
  (13 of 250 fold-seed choices).
- Real GraphSAGE and shuffled-edge GraphSAGE have an interval entirely
  inside the equivalence margin (`0.024523`, 95% CI
  `[0.004590, 0.045019]`), although the interval excludes zero.
- Pooled MAPE ranks models differently because it is tail-sensitive;
  it remains appendix-only under the pre-specified protocol frozen by commit
  hash before the Phase 2 model runs.

## 5. Open questions

- LOSO must run before the two co-primary primary labels can be
  combined.
- Holm-adjusted secondary conclusions cannot be finalized until LOSO
  results enter the eight-test family.
- LOAFO remains unrun and awaits approval of the full 15-family table.

## 6. Acceptance criteria

| Item | Status |
| --- | --- |
| A. Seed-7 fold/evaluated-set reconciliation | Complete |
| B. Protocol v2.1 with new hash and logged diff | Complete |
| C. GB300 72/288 source verification | Complete |
| D. Five-fold production weight equality | Complete; PASS |
| E. Full 15-family LOAFO proposal | Complete; LOAFO not run |
| First complete 15-method ladder on 5-fold × 5 partitions | Complete |
| LOSO | Not started |
| Sensitivity | Not started |

Phase 2 core is complete. Work stops here pending approval to run LOSO.

## Appendix: Full Protocol v2.1

```markdown
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
