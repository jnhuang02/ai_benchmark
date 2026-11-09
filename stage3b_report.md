# Stage 3B Report

Status: Stage 3B corrections and Stage 3C final checks completed; no exploratory semantic block was run.

## Revised abstract (inline)

Under a pre-specified protocol frozen by commit hash before the final model
comparison, graph structure did not improve prediction of MLPerf Inference
performance for unseen systems. All models and fitting budgets were fixed in
advance and were not tuned, so the result concerns these configurations. The
validation-selected best graph neural network was worse than the multilayer
perceptron by about 0.10 to 0.12 log units, and the data could not exclude a
difference as small as 0.02 to 0.05.
Repeated cross-validation estimated best-GNN minus MLP error at
+0.115597 (95% CI [0.047136,0.198051]); leave-one-system-out estimated
+0.104155 (95% CI [0.023774,0.192784]). Both intervals exclude zero but
cross the frozen +0.05 equivalence boundary, so the combined co-primary
label is *inconclusive*. The direction is nevertheless clear: the best
GNN is worse than MLP by approximately 0.10 to 0.12 log units.

Metadata prediction itself is informative. Random forest achieved primary
errors of 0.201853 under repeated cross-validation and 0.165012 under
leave-one-system-out, corresponding to typical multiplicative misses of
approximately 22% and 18%. The group-median baseline achieved 0.617066 and
0.580975, corresponding to approximately 85% and 79%. The displayed
percentage figures convert \exp(\mathrm{PE})-1 to percent for primary error
\mathrm{PE}. The analysis uses
476 published MLPerf Inference v6.1 Closed-division measurements from 105
submitter-aware systems in the Datacenter and Edge suites.

The fixed, untuned model ladder comprises group-median, Ridge,
random-forest, gradient-boosting, hierarchical-mixed, MLP, GraphSAGE, GCN, and
GAT models.
GraphSAGE is worse than random forest after Holm correction under both
co-primary protocols, and GAT is worse than GraphSAGE under both. Real-edge
GraphSAGE minus shuffled-edge GraphSAGE is +0.024523 under repeated
cross-validation, labeled *equivalent*, and -0.032199 under
leave-one-system-out, labeled *inconclusive*. GCN minus GraphSAGE is
-0.016638 under repeated cross-validation, labeled *equivalent*, and
+0.022510 under leave-one-system-out, labeled *inconclusive*. The
near-duplicate sensitivity guard estimates best GNN minus MLP at +0.158657
(95% CI [0.058606,0.257561]), labeled *worse*; this is not a
co-primary protocol. The sensitivity split is a modest check because 78 of its
90 groups are singleton systems.

The family-level extrapolation study is descriptive because the protocol has
no LOAFO pooling rule. For the six trained non-baseline models, 18 of 36
model--headline-family combinations beat the group-median baseline. The count
varies by family rather than following one narrative: MI350X 4 of 6, MI355X 0
of 6, Intel Arc Pro B70 4 of 6, B200 5 of 6, B300 3 of 6, and RTX PRO 2 of 6.
Neither MI355X nor B200 should be treated as the typical family.
Among the five families with at least eight supported systems, best-GNN minus
MLP excludes zero in the GNN's favor for MI355X (-0.115406,
95% CI [-0.504774,-0.038053]) and the CPU-only stress case (-0.398394,
95% CI [-0.802894,-0.040203]); B200, B300, and RTX PRO cross zero. These
family intervals receive no multiplicity correction.

Tail behavior depends on the statistic. MLP has the lower 95th-percentile
absolute-log error under both protocols (1.415805 versus 1.824215 for
random forest under repeated cross-validation, and 1.359019 versus
1.588001 under leave-one-system-out). MLP also has the lower 90th
percentile under repeated cross-validation (1.089365 versus 1.147904);
the leave-one-system-out 90th percentiles are effectively tied (1.101461
for random forest and 1.102937 for MLP). Two interactive-throughput rows
are shared extrapolation failures across the model families.

The thesis concludes that graph structure did not provide a reliable
predictive advantage under the frozen evaluation. It contributes an
evaluation protocol for held-out-system prediction, a complete model ladder,
family-level extrapolation evidence, and a provenance record that includes a
corrected training-metadata leak and the discrepancy between the historical
single-run rank correlation and pinned-fold results.

## Revised conclusion (inline TeX source)

```latex
\chapter{Conclusion}
\label{chap:conclusion}


This thesis evaluated whether published system and workload metadata can
predict MLPerf Inference performance for unseen systems and whether explicit
graph structure adds predictive value. The analysis used 476 MLPerf Inference
v6.1 Closed-division measurements from 105 submitter-aware systems and a
pre-specified protocol frozen by commit hash before the final model comparison.
All models and fitting budgets were fixed in advance and were not tuned; the
result concerns these configurations.

The primary result is negative for graph structure. The best GNN is worse than
MLP by approximately 0.10 to 0.12 log units. The repeated-CV difference is
$+0.115597$ with 95\% CI $[0.047136,0.198051]$, and the LOSO difference is
$+0.104155$ with 95\% CI $[0.023774,0.192784]$. Both are labeled
\emph{inconclusive} under the frozen $\pm0.05$ rule because the data cannot
exclude a difference as small as 0.02 to 0.05. This is a directional negative
finding, not an equivalence finding.

Metadata prediction itself is useful. Random forest reaches primary errors of
$0.201853$ under repeated CV and $0.165012$ under LOSO, compared with
$0.617066$ and $0.580975$ for the group median. These correspond approximately
to 22\% and 18\% typical multiplicative misses for random forest and 85\% and
79\% for the group median. The strongest overall learner is therefore a
tabular model rather than a graph model.

The graph comparisons do not reveal a uniform edge advantage. Real-edge
GraphSAGE minus shuffled-edge GraphSAGE is $+0.024523$ and labeled
\emph{equivalent} under repeated CV, but $-0.032199$ and labeled
\emph{inconclusive} under LOSO. GraphSAGE is worse than random forest under both
protocols, and GAT is worse than GraphSAGE under both. The near-duplicate
sensitivity estimate makes best GNN minus MLP more unfavorable at $+0.158657$,
but sensitivity is not co-primary. It is a modest check because 78 of its 90
groups are singleton systems.

Family extrapolation is heterogeneous. Across 36 trained-model and
headline-family combinations, 18 beat the group median. Best GNN beats MLP in
two of six headline families; among the five families with interpretable
intervals, two exclude zero in the GNN's favor and three cross zero. The family
results are descriptive, receive no multiplicity correction, and are not
pooled.

Tail behavior reinforces the need for multiple error summaries. MLP has the
lower 95th percentile under both protocols and the lower 90th percentile under
repeated CV. LOSO's 90th percentiles are effectively tied. Two
interactive-throughput rows are shared extrapolation failures and dominate
normalized RMSE.

The contributions are an evaluation protocol for held-out-system prediction, a
fixed and untuned model ladder, a negative finding for the implemented metadata
graph, family-level extrapolation evidence, and a provenance record covering
leakage, protocol versions, and tail behavior. Future work should test graph
constructions with richer node features and fully inductive evaluation,
including operator-level execution information when matching traces are
available.
```

## Revised Chapter 6 disclosures (inline TeX source)

```latex
\chapter{Evaluation Protocol}
\label{chap:protocol}


The evaluation protocol was pre-specified and frozen by commit hash before the
final model comparison. Earlier pipeline-development and audit stages exposed
the author to the dataset, motivated submitter-aware grouping and isolation
audits, and led to protocol revisions before the final comparison existed. The
protocol passed through versions v1, v2, and v2.1 before the final model
results were available.

The retained protocol hash is recorded in \texttt{protocol.sha256} and in the
result manifests. A separate protocol-history file records the v2.2 addendum
and erratum without changing the hash-pinned file.

The three recorded deviations are \texttt{S3A-SHRINK-001}, an unexecuted
secondary whose pinned definition lacks a unique shrinkage posterior; the
LOAFO addendum, under which the model set, seeds, inference set, and
family-level rule were specified in the author's analysis plan outside the
hash-pinned file; and the compute erratum, which corrects fold counts and
planning arithmetic without changing results.

\input{generated/tab_protocol}

\section{Co-primary protocols}

Repeated grouped five-fold cross-validation uses partition seeds
20261107--20261111. Neural seeds 7--16 are used for the core repeated-CV
ladder. Each partition is evaluated on its own out-of-fold predictions, and the
five partition metrics are averaged.

Leave-one-system-out contains 105 folds. For each fold, one complete
submitter-aware system is held out, approximately 12\% of the remaining systems
form a training-only validation partition, and all other systems train the
model. LOSO uses neural seeds 7, 11, and 16.

The sensitivity split groups by
\texttt{submitter|accelerator|processor} and uses five folds with neural seeds
7--16. It is a modest check because 78 of its 90 groups are singleton systems.

\section{Metrics and supported rows}

The primary metric is the median absolute log ratio over supported out-of-fold
rows. Secondary summaries are row absolute percentage error, normalized RMSE
on the signed within-group log target, corrected median-group Spearman
correlation, and coverage. Group-median Spearman is reported as unavailable
because that baseline has no within-group rank variation.

A workload group is supported only when the training partition contains at
least three rows and nonzero scale. The supported test rows are determined from
that training partition and are identical across all models. Pooled MAPE is an
appendix diagnostic and not a model-selection or hypothesis-testing metric.

\section{Primary hypothesis and uncertainty}

The sole primary hypothesis is
\begin{equation}
  \operatorname{PE}(\text{best GNN})
  -
  \operatorname{PE}(\text{MLP}).
\end{equation}
Confidence intervals use 2,000 bootstrap resamples of complete systems. Within
each draw, seed and partition variability are averaged inside a selected
system before pooling, and the same selected systems are used for both models
in a paired comparison.

The equivalence margin is $\pm0.05$ log units. Both co-primary protocols must
produce the same label before a primary claim is made. An interval entirely
below $-0.05$ is \emph{better}; one entirely above $+0.05$ is \emph{worse}; one
entirely inside $[-0.05,+0.05]$ is \emph{equivalent}, even if it excludes zero;
and one crossing either boundary is \emph{inconclusive}. The primary analysis
has no multiplicity correction because it contains one hypothesis.

\section{Secondary comparisons}

GraphSAGE is the fixed reference for four secondary comparisons:
GraphSAGE minus random forest, real-edge GraphSAGE minus shuffled-edge
GraphSAGE, GCN minus GraphSAGE, and GAT minus GraphSAGE. Each is evaluated
under repeated CV and LOSO, producing eight tests. Holm correction is applied
across the eight tests. A zero-containing interval is not evidence of
equivalence.

\section{Extrapolation protocols}

LOAFO holds out one approved accelerator family at a time. The graph is fitted
without that family, and mixed-accelerator systems remain in training for every
family fold. The hierarchical mixed model was not included in LOAFO. Because
the protocol defines no rule for pooling family errors, LOAFO is descriptive
and no pooled estimate is formed.

The protocol had requested a sparse-group partial-pooling secondary analysis.
The pinned definition does not identify a unique shrinkage posterior, prior,
variance estimator, or post-shrinkage evaluation rule. The secondary was
therefore not executed and is logged as deviation \texttt{S3A-SHRINK-001}.
No empirical-Bayes formula was chosen after observing the primary results.

\section{Leakage audit}

An earlier pipeline allowed held-out system metadata to influence training
weights through message passing and degree normalization, even though held-out
labels were masked. The production evaluation isolates training from test
metadata. For GraphSAGE, replaying the legacy and corrected pipelines on paired
system draws from the canonical five-fold partition, with the same 450
supported rows and ten neural seeds, gives a legacy-minus-fixed primary-error
difference of $+0.0043$ (95\% CI $[-0.0779,+0.0782]$). This estimate provides
no detectable benefit from the leak. The GCN degree-normalization effect was
not separately measured. The corrected isolated pipeline is used for every
primary and secondary result in this thesis.

\section{Model-selection and modeling limits}

Best-GNN selection is noisy because each outer-fold validation partition
contains roughly ten to twelve systems. The selected architecture can therefore change
with partition and seed. This is part of the evaluation, not a tuning step.

The hierarchical mixed model gives an unseen held-out system a zero random
intercept and contributes no held-out variance estimation. In the CV
diagnostics it loses to Ridge on most target groups. It is retained in the ladder for
descriptive comparison but is not used for GNN selection or LOAFO.

The protocol compute table was corrected in version 2.2. Repeated CV contains
25 outer folds, LOSO contains 105, and sensitivity contains 5. The planning
estimate of 6,210 fits and 37.26 minutes at the documented rate covers those
$25+105+5=135$ outer folds only. The 13 LOAFO folds are separate and contribute
195 planning fits under the same compute-table convention. This bookkeeping
correction changes no result.

No model is tuned. LOSO uses fewer neural seeds than the other protocols, and
the LOAFO model set, seeds, inference set, and family-level inference rule were
specified in the author's analysis plan outside the hash-pinned file. The
specific stochastic variation is reported rather than removed.
```

## Appendix B AI-tool disclosure (inline TeX source)

```latex
The AI-assisted-tools section is not present in the current author-edited Overleaf draft.
```

## 1. What was done

- Reorganized the thesis into nine chapters.
- Moved apparatus, provenance, and reproduction material to Appendices A--C.
- Removed Open-division lists, earlier releases, rented-GPU claims, and obsolete historical result claims from the compiled thesis.
- Added the corrected negative-first abstract, metadata-prediction result, sensitivity result, 18-of-36 LOAFO count, interval heterogeneity, and corrected tail statement.
- Scoped the negative result to the fixed, untuned model ladder and replaced the former graph-analysis framing.
- Expanded the AI-tool disclosure to include evaluation design, protocol drafting and amendments, analysis planning, and review of intermediate results, with an author-policy confirmation instruction.
- Corrected the four-label decision rule, the leakage sign and scope, the earlier-stage wording, and the separate LOAFO compute count.
- Revised the introduction, related-work positioning, and Discussion so mechanism explanations are labeled as hypotheses.
- Split the old results chapter into Evaluation Protocol and Results.
- Moved the tail table into the Results main text.
- Generated every result table and figure from final result bundles.
- Added protocol, leakage, model-selection, mixed-model, and AI-tool disclosures.
- Added explicit placeholders for dedication, acknowledgments, and vita.
- Compiled the thesis with pdfLaTeX and BibTeX.

## 2. Files changed

- `thesis_overleaf/main.tex`
- `thesis_overleaf/chapters/01_introduction.tex` through `09_conclusion.tex`
- `thesis_overleaf/chapters/appendix_a_apparatus.tex` through `appendix_c_reproducibility.tex`
- `thesis_overleaf/generated/` tables and figures
- `mlperf_gnn/generate_thesis_results.py`
- `mlperf_gnn/requirements-lock.txt`
- `thesis_overleaf/stage3b_report.md`
- `thesis_overleaf/stage3b_number_source_map.csv`
- `thesis_overleaf/stage3b_prose_number_audit.csv`
- `thesis_overleaf/stage3b_test_output.txt`
- `thesis_overleaf/stage3b_pdf_value_check.csv`
- `thesis_overleaf/generated/source_snapshots/`

## 3. Generated result assets

Tables:

- `tab_data_coverage.tex`
- `tab_protocol.tex`
- `tab_model_ladder.tex`
- `tab_primary_decision.tex`
- `tab_secondary_holm.tex`
- `tab_tail_summary.tex`
- `tab_loafo_headline.tex`
- `tab_loafo_other.tex`
- `tab_loafo_pairing.tex`
- `tab_partition_errors.tex`
- `tab_gat_diagnostic.tex`
- `tab_provenance.tex`
- `tab_mlperf_parameters.tex`

Figures:

- `fig_model_ladder.pdf`
- `fig_primary_decision.pdf`
- `fig_tail_errors.pdf`
- `fig_loafo_group_median.pdf`

Committed source snapshots reproduce these assets when the ignored `results/` directory is absent. Their hashes are recorded in `generated/source_snapshots/source_manifest.json`.

## 4. Claim and number inventory

| Claim | Approved statement | Source-map rows |
| --- | --- | --- |
| S1 | Corpus and protocol: 476 measurements, 105 systems, five CV partitions, ten neural seeds, 2,000 bootstrap draws, and a ±0.05 margin. | T0001, T0002, T0003, T0010, T0014, T0015 |
| S2 | Corrected planning total: 6,210 fits and 37.26 minutes for 135 repeated-CV, LOSO, and sensitivity folds; the separate 13-fold LOAFO design contributes 195 planning fits. | T0047, T0050, T0048, T0049 |
| S3 | Protocol structure includes 25 repeated-CV outer folds, 105 LOSO folds, five sensitivity folds, 13 LOAFO folds, four secondary comparison families, eight Holm tests, 78/90 singleton sensitivity groups, and three-row support. | T0006, T0007, T0008, T0009, T0025, T0026, T0044, T0045, T0027 |
| S4 | Earlier pipeline-development and audit stages exposed the author to the data. Repeated-CV partition seeds run from 20261107 through 20261111; LOSO uses seeds 7, 11, and 16; the validation split is approximately 12% and averages about 11.56 systems. The protocol record spans v1, v2, v2.1, and the v2.2 history addendum. | T0004, T0005, T0011, T0012, T0013, T0028, T0046, T0020, T0021, T0022, T0023 |
| P1 | Best GNN minus MLP under CV5 is +0.115597 with CI [0.047136, 0.198051] and label `inconclusive`. | T0096, T0097, T0098 |
| P2 | Best GNN minus MLP under LOSO is +0.104155 with CI [0.023774, 0.192784] and label `inconclusive`. | T0099, T0100, T0101 |
| P3 | Sensitivity best GNN minus MLP is +0.158657 with CI [0.058606, 0.257561], label `worse`. | T0102, T0103, T0104 |
| M1 | RF/MLP/group-median primary errors are 0.201853/0.261071/0.617066 (CV5) and 0.165012/0.246446/0.580975 (LOSO). | T0064, T0065, T0066, T0072, T0073, T0074 |
| M2 | Corresponding typical multiplicative misses are approximately 22%/18% for RF and 85%/79% for the group median. | T0092, T0093, T0094, T0095 |
| M3 | Supported-row coverage is 449.8 rows (94.495798%) under repeated CV, 454 rows (95.378151%) under LOSO, and 449 rows (94.327731%) under sensitivity; six LOSO folds have no supported held-out row. | T0054, T0056, T0057, T0060, T0061, T0063, T0058 |
| G1 | GraphSAGE is worse than RF by +0.171529 (CV5) and +0.163960 (LOSO). | T0105, T0106 |
| G2 | GAT is worse than GraphSAGE by +0.267067 (CV5) and +0.328409 (LOSO). CV5 and LOSO primary errors are 0.632643 and 0.658164. | T0107, T0108, T0070, T0078 |
| G3 | Shuffled-edge comparison: CV5 +0.024523 (`equivalent`, Holm p 0.096000); LOSO -0.032199 (`inconclusive`, Holm p 0.387000). | T0109, T0110, T0111, T0112 |
| G4 | GCN minus GraphSAGE is -0.016638 (CV5, `equivalent`) and +0.022510 (LOSO, `inconclusive`). | T0113, T0114 |
| G5 | Graph architecture and ablation errors: GraphSAGE 0.370271/0.338161, shuffled edges 0.341685/0.366378, GCN 0.354737/0.356236, GAT 0.632643/0.658164, and best GNN 0.376393/0.340865 under CV5/LOSO. | T0067, T0075, T0068, T0076, T0069, T0077, T0070, T0078, T0071, T0079 |
| T1 | MLP has lower p95 under both protocols and lower p90 under CV5; LOSO p90 is effectively tied. The tail windows are ±25% and ±50%. | T0116, T0120, T0117, T0121, T0124, T0128, T0125, T0129, T0018, T0019 |
| L1 | Eighteen of 36 trained-model/headline-family comparisons beat the group median; family counts are 4, 0, 4, 5, 3, and 2. | T0163, T0133, T0136, T0141, T0144, T0149, T0154 |
| L2 | Best-GNN-minus-MLP point differences are +0.572661 (MI350X), -0.115406 (MI355X), -0.391312 (B70), +0.101929 (B200), +0.059492 (B300), +0.334701 (RTX PRO), and -0.398394 (CPU-only). Intervals exclude zero for MI355X and CPU-only and cross zero for B200, B300, and RTX PRO. | T0135, T0138, T0139, T0140, T0143, T0146, T0147, T0148, T0151, T0152, T0153, T0156, T0157, T0158, T0160, T0161, T0162 |
| L3 | Supported systems are 3 (MI350X), 12 (MI355X), 3 (B70), 14 (B200), 33 (B300), 15 (RTX PRO), and 10 (CPU-only). | T0134, T0137, T0142, T0145, T0150, T0155, T0159 |
| L4 | Six trained models create 36 headline-family comparisons; best GNN beats MLP on the point estimate in two of six headline families; five families are interval-eligible. | T0131, T0132, T0164, T0165, T0166 |
| V1 | GraphSAGE partition SD is 0.093310; RF is 0.032297; MLP is 0.019410. | T0167, T0168, T0169 |
| O1 | Two shared interactive-throughput extrapolation failures reach |z| approximately 54 and 26. | T0172, T0177, T0178 |
| O2 | Top-ten residuals contain roughly 78--87% of squared normalized error across the reported models. | T0175, T0176 |
| A1 | Earlier leakage effect is +0.0043 (95% CI [-0.0779, +0.0782]). | T0182, T0183, T0184 |
| A2 | Historical single-run Spearman is 0.711303; pinned ten-seed reconciliation mean is 0.484192. | T0185, T0186 |

Every numeric statement in the abstract, conclusion, Evaluation Protocol chapter, Results prose, and provenance appendix is represented above or in the accompanying exhaustive `stage3b_number_source_map.csv`. Table cells are generated directly by `generate_thesis_results.py` and are not typed into LaTeX.

### Automated prose-number scan

- Source-map rows: 186.
- Numeric tokens scanned in the abstract, Chapters 1--9, and Appendix B: 239.
- Numeric tokens not covered by a map row: 0.

```text
No numeric token was uncovered.
```

### Rendered-PDF frozen-value check

| Expected text | Present in PDF |
| --- | --- |
| `0.201853` | Yes |
| `0.165012` | Yes |
| `0.261071` | Yes |
| `0.246446` | Yes |
| `0.617066` | Yes |
| `0.580975` | Yes |
| `+0.115597` | Yes |
| `[0.047136, 0.198051]` | Yes |
| `+0.104155` | Yes |
| `[0.023774, 0.192784]` | Yes |
| `+0.158657` | Yes |
| `18 of 36` | Yes |

### Random 15-row rendered-PDF spot-check

The seeded sample verifies the rendered PDF value against the mapped source column and its source value.

| PDF value | Map ID | Source file | Column | Source value | Match |
| --- | --- | --- | --- | ---: | --- |
| `0.201853` | T0115 | `mlperf_gnn/results/phase2-5-analysis-4faac06/tail_summary.csv` | `median` | `0.2018527620002152` | Yes |
| `-0.391312` | T0143 | `mlperf_gnn/results/phase2-loafo-final-1a65664/paired_differences_by_family.csv` | `point_difference` | `-0.3913123299237027` | Yes |
| `0.023774` | T0100 | `mlperf_gnn/results/phase2-final-f51e57a/primary_decision.csv` | `difference_95ci_low` | `0.0237744878397438` | Yes |
| `-0.032199` | T0111 | `mlperf_gnn/results/phase2-final-f51e57a/secondary_holm.csv` | `mean_difference` | `-0.0321990903629073` | Yes |
| `1.588001` | T0125 | `mlperf_gnn/results/phase2-5-analysis-4faac06/tail_summary.csv` | `p95` | `1.5880011364816715` | Yes |
| `0.261071` | T0065 | `mlperf_gnn/results/phase2-final-f51e57a/cv5_main_table.csv` | `primary_error_mean` | `0.2610712465396809` | Yes |
| `-0.115406` | T0138 | `mlperf_gnn/results/phase2-loafo-final-1a65664/paired_differences_by_family.csv` | `point_difference` | `-0.1154055937690552` | Yes |
| `0.580975` | T0074 | `mlperf_gnn/results/phase2-final-f51e57a/loso_main_table.csv` | `primary_error` | `0.5809749321415092` | Yes |
| `-0.802894` | T0161 | `mlperf_gnn/results/phase2-loafo-final-1a65664/paired_differences_by_family.csv` | `difference_95ci_low` | `-0.8028939689664676` | Yes |
| `0.356236` | T0077 | `mlperf_gnn/results/phase2-final-f51e57a/loso_main_table.csv` | `primary_error` | `0.3562359662750386` | Yes |
| `0.201853` | T0064 | `mlperf_gnn/results/phase2-final-f51e57a/cv5_main_table.csv` | `primary_error_mean` | `0.2018527620002152` | Yes |
| `0.165012` | T0123 | `mlperf_gnn/results/phase2-5-analysis-4faac06/tail_summary.csv` | `median` | `0.1650119636115254` | Yes |
| `0.365091` | T0083 | `mlperf_gnn/results/phase2-final-f51e57a/sensitivity_main_table.csv` | `primary_error` | `0.3650909242592064` | Yes |
| `1.147904` | T0116 | `mlperf_gnn/results/phase2-5-analysis-4faac06/tail_summary.csv` | `p90` | `1.1479043243779243` | Yes |
| `0.047136` | T0097 | `mlperf_gnn/results/phase2-final-f51e57a/primary_decision.csv` | `difference_95ci_low` | `0.0471356451420624` | Yes |

## 5. Numbers that changed (old -> new)

| Quantity | Old thesis | Rewritten thesis |
| --- | --- | --- |
| Scope | 96 systems | 105 submitter-aware systems |
| Primary finding | Qualified support for GNNs | Best GNN worse by about 0.10--0.12 log units; `inconclusive` under ±0.05 |
| Main comparator | GNN-specific metric tradeoffs | RF best overall; GraphSAGE worse than RF after Holm |
| GAT | No consolidated conclusion | Worse than GraphSAGE under both protocols |
| Tail claim | No consolidated RF/MLP tail analysis | MLP lower p95 both; p90 CV5 only; LOSO p90 tied |
| LOAFO | Development/future split language | Descriptive 13-family stress test; 18 of 36 trained-model cases beat GM |
| Leakage | Main-text omission | Disclosed with +0.0043 and corrected isolation |
| Historical Spearman | Treated as prior result | 0.711303 provenance vs 0.484192 pinned reconciliation |
| Model scope | Generic graph-versus-tabular comparison | Fixed, untuned model ladder; result limited to these configurations |
| Sensitivity scope | Conservative check | Modest check because 78 of 90 groups are singleton systems |
| Compute scope | 6,210 total could imply all protocols | 6,210 covers 135 folds only; LOAFO has 195 separate planning fits |
| AI disclosure | Code, tables, and editing | Also evaluation design, protocol drafting and amendments, analysis planning, and intermediate-result review |
| Chapter count | Eight chapters | Nine chapters with protocol/results split |
| Tables/figures | Mixed stale sources | All generated from final bundles |

## 6. Surprising or narrative-changing findings

- Metadata prediction is materially useful even though graph structure is not.
- MI355X reverses the usual family narrative: no trained model beats the group median.
- MI350X and B200 show large improvements from tabular models.
- GraphSAGE has greater partition sensitivity than RF and MLP.
- The same two interactive-throughput measurements fail across model families.
- The partial-pooling secondary cannot be recovered uniquely from the pinned text.

## 7. Open questions

- The author must confirm the AI-assisted-tool disclosure with the dissertation advisor and against applicable UCLA policy.
- Committee members and final author biography remain to be supplied.
- A future protocol may define a reproducible partial-pooling posterior and operator-level graph evaluation.

## 8. Verification and acceptance

- Thesis compiled with pdfLaTeX and BibTeX.
- Full test suite: 21 tests passed; output is saved in `stage3b_test_output.txt`.
- LaTeX undefined references: 0.
- LaTeX undefined citations: 0.
- LaTeX overfull boxes: 0.
- LaTeX underfull boxes: 2.
- Every warning or box line from `main.log`:

```text
Underfull \hbox (badness 10000) in paragraph at lines 154--154
Underfull \hbox (badness 1817) in paragraph at lines 24--28
```

- Source context for box warnings:

```text
main.tex:154
 152: \begin{document}
 153:
 154: \makeintropages
 155:
 156: \input{chapters/01_introduction}

main.bbl:24
  22:
  23: \bibitem[MLC26]{mlcommons-v61-results}
  24: {MLCommons}.
  25: \newblock ``{MLPerf Inference v6.1 Results}.''
  26: \newblock \url{https://github.com/mlcommons/inference_results_v6.1}, 2026.
```

- Lockfile and recorded runtime versions: match.
- Frozen protocol SHA-256: `d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e`.
- Final bundle hash rows verified: 62; all match: yes.
- Generated-asset hash rows verified: 35; all match: yes.
- Rendered-PDF spot-check rows: 15; all values match: yes.
- Author placeholders visibly present in the PDF: yes.
- Fresh-clone snapshot-only reproduction:

```text
temporary_root=/tmp/stage3c-fresh-final.p3QjNy
mlperf_commit=0829eda8abbe547d4e6f528288fa5f88c9e4b87d
thesis_commit=dafcf4879e4c0cff8c2366efe79e9e5424d508d1
command=arch -arm64 python3 generate_thesis_results.py --snapshot-only
/private/tmp/stage3c-fresh-final.p3QjNy/thesis_overleaf/generated
generated_assets_clean=yes
main_table_sha256=b8a6f1d319eebcae4bff7784dd32cb69518486786b06a5765391611c37da0ea1
checker_command=arch -arm64 python3 stage3c_final_checks.py --run-tests
```

- Generated files in `generated/`: 36; the manifest hashes 35 assets excluding the manifest itself.
- A second `--snapshot-only` run reproduced all 36 generated files byte-for-byte.
- Main text contains nine chapters plus three appendices.
- Every results table and figure is generated from final bundles.
- Revised abstract opens with the negative result and metadata result.
- The abstract, introduction, and conclusion state that the models were fixed and untuned and name the model ladder.
- Chapter 6 reports the full four-label rule and the corrected leakage and compute scopes.
- Discussion mechanism explanations are explicitly labeled as hypotheses.
- Tail statement matches: p95 favors MLP under both protocols; p90 favors MLP only under CV5.
- LOAFO reporting uses the verified 18-of-36 count and all five interval-eligible comparisons.
- The primary-hypothesis prose uses the frozen label and interval rather than prohibited equivalence language.
- Unsupported historical Table 6.4 values are provenance-only.
- Thesis LaTeX changed in Stage 3B; no exploratory semantic block was run.

