# Stage 3A Plan and Claims Report

Status: planning and claim approval only. No thesis LaTeX prose, abstract, table, figure, bibliography, or PDF was edited. Existing uncommitted thesis files were preserved.

## 1. What was done

- Audited the accepted Phase 2.5 evidence against the proposed Phase 3 scope.
- Defined the thesis title, framing, research questions, terminology, contributions, claim boundaries, chapter restructuring, and table/figure placement.
- Prepared exact LOAFO and tail-summary text requirements for Stage 3B.
- Kept the original Table 6.4, Phase 0 diagnostics, and historical run fragments outside the result claims.
- Stopped before thesis editing.

## 2. Proposed thesis plan

### 2.1 Proposed title

**Predicting MLPerf Inference Performance for Unseen Systems: A Comparison of Graph and Tabular Models**

The title uses "inference workloads" throughout. Claims are not restricted to LLMs.

### 2.2 Full abstract

Predicting the performance of machine-learning systems before they are benchmarked is difficult because published results combine hardware, software, workload, and submitter-specific implementation choices. This dissertation evaluates whether system and workload metadata can predict out-of-sample MLPerf Inference v6.1 performance and whether explicit graph structure improves that prediction. The analysis uses 476 published Closed-division measurements from 105 submitter-aware systems in the Datacenter and Edge suites.

The evaluation follows a pre-specified protocol frozen by commit hash before the Phase 2 model runs. The protocol defines grouped five-fold cross-validation over five partition seeds, leave-one-system-out validation, supported workload groups, system-clustered uncertainty, an equivalence margin, and no model tuning. The model ladder contains group-median, Ridge, random-forest, gradient-boosting, hierarchical mixed, MLP, GraphSAGE, GCN, and GAT models. All models use the same supported rows and the same log-transformed accelerator-count features. Test systems enter only the inference graph after model weights are fixed.

Random forest has the lowest repeated-CV primary error (`0.201853`) and the lowest LOSO primary error (`0.165012`). The corresponding MLP errors are `0.261071` and `0.246446`. The sole primary hypothesis compares the validation-selected best GNN with MLP. Its repeated-CV difference is `+0.115597`, 95% CI `[0.047136, 0.198051]`; its LOSO difference is `+0.104155`, 95% CI `[0.023774, 0.192784]`. Both intervals exclude zero but cross the frozen `+0.05` boundary, so the combined co-primary label is `inconclusive`.

Secondary comparisons show GraphSAGE worse than random forest under both protocols and GAT worse than GraphSAGE under both protocols after Holm correction. Real-edge GraphSAGE minus shuffled-edge GraphSAGE is `+0.024523` under repeated CV, labeled `equivalent`, and `-0.032199` under LOSO, labeled `inconclusive`. GCN minus GraphSAGE is `-0.016638` under repeated CV, labeled `equivalent`, and `+0.022510` under LOSO, labeled `inconclusive`.

The leave-one-accelerator-family-out stress test is descriptive because the frozen protocol contains no rule for pooling family errors. Across families, trained models do not reliably beat the group-median baseline. Best GNN has a lower primary-error point estimate than MLP in only 2 of 6 headline families; one of those two families, Intel Arc Pro B70, has only 3 supported systems and no reportable system-clustered interval. On MI355X, the group median has lower error than every trained model. Tail analysis also shows that model rankings depend on the statistic considered: random forest has the better median and larger share within ±25%, while MLP has the lower 90th and 95th percentiles.

The thesis concludes that explicit heterogeneous graph structure did not provide a reliable predictive advantage under this evaluation. The strongest tabular model, not the graph model, supplies the best overall predictions. The study contributes a leakage-audited, system-held-out benchmark, a complete model ladder, family-level extrapolation evidence, and an auditable record of protocol changes and limitations.

### 2.3 Framing

The thesis follows a pre-specified protocol frozen by commit hash before the Phase 2 model runs. Its primary empirical result is negative for graph structure under that protocol: the best GNN is worse than MLP by about 0.10-0.12 log units under the two co-primary protocols, while the intervals cannot exclude differences as small as approximately 0.02-0.05 log units because they cross the frozen `+0.05` equivalence boundary.

### 2.4 Full conclusion

This dissertation asked whether published system and workload metadata can predict MLPerf Inference performance for unseen systems and whether explicit graph structure adds value over tabular and MLP models. The evaluation was governed by a pre-specified protocol frozen by commit hash before the Phase 2 model runs. The primary result is negative for graph structure: the best GNN has higher primary error than MLP under both co-primary protocols. Under repeated CV the difference is `+0.115597` with 95% CI `[0.047136, 0.198051]`; under LOSO it is `+0.104155` with 95% CI `[0.023774, 0.192784]`. Both are labeled `inconclusive` because their intervals exclude zero but cross the frozen practical-margin boundary. The data therefore do not support a claim that graph structure improves held-out-system prediction.

The model ladder also shows that a standard tabular ensemble is the strongest overall learner: random forest reaches `0.201853` under repeated CV and `0.165012` under LOSO. GraphSAGE is worse than random forest after Holm correction under both protocols, and GAT is worse than GraphSAGE under both protocols. The shuffled-edge comparison does not provide a uniform edge-value conclusion: repeated CV is labeled `equivalent` while LOSO is labeled `inconclusive`. These protocol-specific results are part of the conclusion rather than post hoc qualifications.

The leave-one-family-out analysis exposes important scope limits. No pooled LOAFO estimate is valid because the frozen protocol has no pooling rule. The family-level results show that trained models do not reliably beat the group-median baseline. Best GNN beats MLP on the point estimate in only 2 of 6 headline families, and one of those families, Intel Arc Pro B70, has only three supported systems and therefore no reportable system-clustered interval. Large tail errors and interactive-throughput outliers further constrain claims about operational accuracy.

The contribution is therefore methodological and empirical: a leakage-audited protocol, a transparent comparison of graph and tabular models, and a negative finding reported without tuning or result-dependent revision. Future work should test richer system metadata, treatment of interactive-throughput extremes, and graph constructions that are evaluated without exposing held-out systems.

### 2.5 Research questions

1. How accurately can published system and workload metadata predict performance on previously unseen MLPerf Inference systems?
2. Does explicit graph structure add value over tabular and MLP baselines under the pre-specified protocol frozen by commit hash before the Phase 2 model runs?
3. Where do the models fail, including workload-family extrapolation, distribution tails, partition variability, and interactive-throughput outliers?

### 2.6 Definitions used once and consistently

- **Submission:** a publisher's account of one system and its measurements in a suite.
- **System:** a submitter-aware hardware/software configuration identified by suite, submitter, system name, and system type.
- **Workload:** the named MLPerf model or benchmark task.
- **Workload group:** the scenario, metric, workload, and suite combination used for target normalization and ranking.
- **Result cell:** one published scalar performance measurement.
- **Measurement:** synonymous with a result cell when referring to the numeric response.

### 2.7 Contributions

- A leakage-audited, submitter-aware evaluation protocol for 476 Closed-division measurements from 105 systems.
- A comparison of group median, Ridge, random forest, gradient boosting, hierarchical mixed, MLP, GraphSAGE, GCN, and GAT models on shared held-out systems under a pre-specified protocol frozen by commit hash before the Phase 2 model runs.
- A co-primary repeated-CV and LOSO analysis with system-clustered uncertainty and a practical equivalence margin.
- A family-level LOAFO stress test that exposes heterogeneity and failure to reliably beat a simple group-median baseline.
- An audit and reproducibility record covering leakage, protocol amendments, fold stability, sparse-group exclusions, and saved number-to-source mappings.

## 3. Claim inventory

| Claim ID | Approved claim | Source-map rows | Permitted wording |
| --- | --- | --- | --- |
| C1 | Best GNN minus MLP under CV5 is `+0.115597`, 95% CI `[0.047136, 0.198051]`, frozen label `inconclusive`. | N1317, N1318, N1319 | State point estimate, interval, and label together. |
| C2 | Best GNN minus MLP under LOSO is `+0.104155`, 95% CI `[0.023774, 0.192784]`, frozen label `inconclusive`. | N1320, N1321, N1322 | State point estimate, interval, and label together; combined co-primary label is `inconclusive`. |
| C3 | Sensitivity best GNN minus MLP is `+0.158657`, 95% CI `[0.058606, 0.257561]`, label `worse`, but sensitivity is not co-primary. | N1323, N1324, N1325 | Label as `worse`; do not use it to replace the co-primary decision. |
| C4 | GraphSAGE minus random forest is `+0.171529` under CV5 and `+0.163960` under LOSO; both are `worse` after Holm. | N1326, N1330, N1336, N1340 | Report both protocols and Holm status. |
| C5 | Real-edge GraphSAGE minus shuffled-edge GraphSAGE is `+0.024523` under CV5, label `equivalent` (Holm p `0.096000`), and `-0.032199` under LOSO, label `inconclusive` (Holm p `0.387000`). | N1346, N1350, N1351, N1355 | Do not say graph edges add value; state the protocol-specific labels. |
| C6 | GCN minus GraphSAGE is `-0.016638` under CV5, label `equivalent`, and `+0.022510` under LOSO, label `inconclusive`. | N1356, N1361 | Report both protocol-specific conclusions. |
| C7 | GAT minus GraphSAGE is `+0.267067` under CV5 and `+0.328409` under LOSO; both are `worse` after Holm. | N1331, N1335, N1341, N1345 | GAT failed without claiming numerical collapse; use the saved training diagnostics. |
| C8 | LOAFO comprises 13 descriptive family-level stress tests because Protocol v2.1 has no pooling rule. | N0004 | Never present a pooled LOAFO estimate. |
| C9 | Trained models did not reliably beat the group-median baseline; best GNN beat MLP on the point estimate in only `2` of 6 headline families. One of those families, Intel Arc Pro B70, has only 3 supported systems and no reportable system-clustered interval. | N2746, N2745, N2402, N2752 | Use a descriptive count and family examples; do not infer precision from the B70 point estimate. |
| C10 | Tail errors remain large and model rankings depend on the chosen tail statistic. | N2344, N2336, N2365, N2360 | Keep the tail table in Chapter 6 main text. |
| C11 | Two interactive-throughput rows dominate normalized RMSE. | N1892, N1896 | Treat this as a limitation of normalized RMSE, not a ranking signal. |

Provenance guardrail, not a result claim: the exact historical Table 6.4 value is not cited as a thesis result and is discussed only in the provenance appendix.

Forbidden primary language: "no difference," "competitive," or a claim that graph structure is equivalent to MLP.

## 4. Required Stage 3B content

### 4.1 LOAFO paragraph

The Stage 3B LOAFO paragraph must state:

> Across held-out accelerator families, trained models did not reliably beat the group-median baseline. Best GNN had a lower primary-error point estimate than MLP in only 2 of 6 headline families; one of those two families, Intel Arc Pro B70, has only 3 supported systems and no reportable system-clustered interval. On MI355X, the group median (`0.153894`) had a lower error than every trained model. On RTX PRO, the group median beat 5 of 7 trained models. These are descriptive family-level comparisons; no pooled LOAFO estimate is made, and intervals are restricted to families with at least 8 supported systems.

The paragraph must not imply that the LOAFO failures establish a causal mechanism.

### 4.2 Tail table

The RF/MLP/GraphSAGE tail table belongs in the Chapter 6 main text, not an appendix. It must label the CV5 maximum as the mean of the five per-partition maxima, define the ±25% and ±50% windows, and note that RF has the lower median and larger ±25% share (`52.519592%` versus `44.997516%` under CV5), while MLP has the lower 95th percentile (`1.415805` versus `1.824215`). Under LOSO the same ordering holds: RF median `0.165012`, MLP 95th percentile `1.359019` versus RF `1.588001`.

### 4.3 Chapter and section plan

1. **Chapter 1: Introduction.** Research questions, one-time definitions, contributions, and a concise statement of the negative primary result.
2. **Chapter 2: Related work.** System-performance prediction, tabular learning, heterogeneous graphs, MLPerf, and evidence gaps.
3. **Chapter 3: Data and prediction task.** v6.1 Closed Datacenter and Edge scope, result-cell construction, workload groups, system identity, and data audit.
4. **Chapter 4: Metadata and tabular models.** Features, target normalization, supported-row rule, deterministic baselines, and MLP.
5. **Chapter 5: Heterogeneous graph models.** Implemented system, accelerator, processor, workload, and result nodes; containment, execution, and measurement relations; GraphSAGE, GCN, and GAT.
6. **Chapter 6: Evaluation protocol and results.** Co-primary repeated CV and LOSO, the primary decision, the model ladder, secondary Holm comparisons, edge/shuffle and relation-removal ablations, GAT diagnostic, tail table in main text, descriptive LOAFO in three non-pooled groups, sensitivity, partition variability, and outlier limitation. Every table and figure is regenerated from final bundles with no stale footers.
7. **Chapter 7: Discussion.** Why graph structure did not outperform tabular models, family heterogeneity, group-median baseline failure, tail behavior, and the limits of normalized RMSE.
8. **Chapter 8: Conclusion.** Restate the exact co-primary result, shuffled-edge labels, GAT result, LOAFO caveat, and future work.

**Appendix moves and deletions**

| Existing material | Destination |
| --- | --- |
| Docker/LoadGen setup | Appendix A, apparatus |
| Current Table 5.1 | Appendix A, apparatus |
| QSL/QDL architecture | Appendix A, apparatus |
| Open-division test lists | Delete; outside v6.1 Closed scope |
| Versions 5/6 claims, historical runs, rented-GPU claims | Delete from main text |
| DAG/topological analysis | Short future-work subsection in Chapter 7 |
| Pooled MAPE | Appendix C, with singleton-heavy limitation |
| Leakage/audit trail, mixed-run provenance, original 0.71 Spearman | Appendix B, provenance |
| Full number-to-source map and test output | Appendix B/C and reproducibility package |

**Methods disclosure required in Chapter 6**

- Phases 0-1 exposed the research team to the same dataset before the final protocol was frozen.
- The main protocol was pre-specified and frozen by commit hash before the Phase 2 model runs.
- The v2.2 LOAFO addendum records that models, seeds `7`, `11`, and `16`, omission of the mixed model, and family-level paired inference were specified in orchestration rather than in the hash-pinned file.
- The compute erratum corrects repeated CV to 25 outer folds and sensitivity to 5 outer folds.
- The shrinkage secondary `S3A-SHRINK-001` was not executed because the pinned definition does not identify a unique posterior.
- LOAFO has no pooled estimate; headline, small/integrated, and CPU-only results are reported separately and descriptively.
- No model was tuned, LOSO uses three neural seeds while the other protocols use ten, and equivalence decisions use the frozen ±0.05 log-unit margin.

### 4.4 NA-register and draft-text audit

| NA decision | Count |
| --- | ---: |
| `DO_NOT_CITE` | 122 |
| `RESOLVE_AS_NA_BY_DESIGN` | 15 |
| `DO_NOT_CITE_NULL_CELL` | 9 |

The claim inventory cites no source-map row marked `DO_NOT_CITE` or `DO_NOT_CITE_NULL_CELL`. A source-wide search for conversational fragments and unfinished draft markers found none in the current reports. The word `placeholder` appears only in the intentional instruction to mark dedication, acknowledgments, and vita placeholders. Audited thesis sections—title, abstract, conclusion, Chapters 1-8, and appendices—cite no number marked `DO_NOT_CITE` or `DO_NOT_CITE_NULL_CELL`. Outside verbatim frozen protocol blocks, active reports use the commit-hash-frozen protocol wording requested for Stage 3B.

### 4.5 Placeholders and final checks

Dedication, acknowledgments, and vita remain visibly marked as placeholders until author text is supplied. Every result number must map to a saved file and column. Before Phase 3 completion, cross-references and bibliography must compile, Section 5.1.1 software versions must match the lockfile, and a fresh clone must reproduce the main table through documented commands.

## 5. Open questions

- Does the orchestrator approve the proposed title and claim inventory before Stage 3B edits LaTeX?
- Should the title name `v6.1` explicitly, or keep the version in the abstract and data chapter?
- Does the author want the optional semantic exploratory block supplied separately? It remains outside the main claims and was not started.

## 6. Acceptance criteria

| Item | Met? | Evidence |
| --- | --- | --- |
| Pre-specified frozen-protocol wording and new title proposed | Yes | Section 2 |
| Full abstract pasted | Yes | Section 2.2 |
| Full conclusion pasted | Yes | Section 2.4 |
| Chapter outline with appendix moves and deletions | Yes | Section 4.3 |
| Claim inventory maps every claim to source-map rows | Yes | Section 3 |
| Methods disclose Phases 0-1 exposure, v2.2 LOAFO addendum, compute erratum, shrinkage deviation, and no LOAFO pooling | Yes | Section 4.3 |
| Supported-system counts and B70 caveat included | Yes | C9 and Section 4.1 |
| NA counts and do-not-cite citation audit included | Yes | Section 4.4 |
| Draft-text audit completed | Yes | Section 4.4 |
| Research questions, definitions, and contributions planned | Yes | Sections 2.5-2.7 |
| Negative primary result stated with estimate, interval, and label | Yes | C1-C2 |
| Shuffled-edge and GAT results assigned exact language | Yes | C5 and C7 |
| LOAFO descriptive paragraph supplied | Yes | Section 4.1 |
| Tail table placed in Chapter 6 main text | Yes | Section 4.2 |
| Original Table 6.4 excluded as a result | Yes | provenance guardrail |
| Chapter restructuring mapped | Yes | Section 4.3 |
| Thesis LaTeX left untouched pending approval | Yes | No LaTeX edit performed |
| Stopped after Stage 3A | Yes | No Stage 3B work started |

**STOP.** Stage 3B must not begin without orchestrator approval of this plan and claim inventory.
