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
