# Changelog

All evaluation changes are recorded here with their scientific reason.

## 2026-11-08

- Clarified generated table presentation rather than editing table values:
  separated the CPU-only stress row from the six headline families, added the
  best-GNN point-estimate note, explained suite-level submitter overlap, and
  stated the 25-fold-cell by 10-seed GAT fit count in its caption. Reason:
  prevent misleading row sums and make existing counts self-explanatory.

- Extended the prose-number audit fallback to the existing Phase 2.5 source map
  for the leakage comparison's 450 supported rows. Reason: trace the newly
  stated row count to map row `N0023` while preserving the accepted Stage 3B
  map total of 186 rows.

- Corrected the thesis compute statement to separate the 13 LOAFO folds from
  the 6,210-fit total for the 135 repeated-CV, LOSO, and sensitivity folds.
  Reason: the prior sentence could be read as including LOAFO, and the
  separate design contributes 195 planning fits under the documented
  convention. `protocol_history.md` and the result bundle remain unchanged.

- Expanded the Stage 3B report generator and source map for the fixed,
  untuned model-ladder scope, the complete four-label rule, the leakage
  model/sign/fold/seed disclosure, the GCN caveat, and the expanded AI-tool
  disclosure. Reason: keep the review report synchronized with the corrected
  thesis text and trace the separate 195-fit bookkeeping value.

- Added the Stage 3C final checker. Reason: verify every listed final bundle
  output hash, the frozen protocol and history hashes, lockfile and recorded
  runtime versions, the 21-test suite, generated-asset hashes, LaTeX warning
  context, visible author placeholders, and a reproducible 15-row rendered-PDF
  to source-column spot-check.

## 2026-11-07

- Added the Stage 3B thesis table and figure generator. Reason: build every
  result table and figure directly from the final Phase 2 and Phase 2.5
  bundles, including the complete ladder, co-primary and Holm decisions,
  tails, LOAFO families, partition variability, GAT diagnostics, and
  provenance checks.

- Added the Stage 3B report and prose-number map generator. Reason: review the
  revised abstract, conclusion, protocol disclosures, claim inventory, and
  every number cited in thesis prose against saved result files and columns.

- Added committed source snapshots and a snapshot-only mode to the thesis table
  generator. Reason: allow a fresh thesis checkout to regenerate every result
  table and figure even though the canonical `results/` directory is ignored.

- Added deterministic generated-asset manifests, a prose-number coverage
  audit, LaTeX warning counts, and rendered-PDF checks for the frozen result
  values. Reason: make Stage 3B review mechanically verifiable and detect any
  unscripted table or figure content.

- Added the Version 2.2 post-result protocol-history addendum. Reason: record
  the orchestration-only LOAFO specification, clarify that LOAFO has no pooled
  estimate rule and is therefore descriptive, and correct the reported
  repeated-CV and sensitivity outer-fold counts and compute estimates without
  editing the hash-pinned Protocol v2.1 file.

- Added the Stage 3A corrections runner. Reason: derive supported-system
  counts and interval eligibility from LOAFO predictions, reconcile all 105
  systems to the 97 actually tested, compare every model with the group-median
  baseline, capture the full 21-test suite, and log the under-specified
  partial-pooling secondary as not executable without inventing a posterior.

- Added the Stage 3A plan and claims report generator. Reason: freeze the
  proposed title, research questions, definitions, contributions, evidence
  claims, main-text table placement, LOAFO wording, and chapter structure
  before any thesis LaTeX edit.

- Revised the Phase 2.5 report generator for the accepted corrections. Reason:
  enforce the eight-supported-system interval rule, add supported-system and
  group-median comparisons, correct the LOAFO coverage explanation, reconcile
  the targeted and full test runs, label tail maxima and windows, list every
  NA decision, and include the partial-pooling deviation.

- Logged `S3A-SHRINK-001` after confirming the pinned partial-pooling
  description lacks a unique posterior and parameterization. Reason: preserve
  the requested secondary analysis as not executable rather than choosing an
  empirical-Bayes convention after observing the primary results.

- Re-generated the correction bundle after adding the deviation record and
  pointed the report generators at the new commit-stamped bundle. Reason:
  keep the protocol-history hash in the correction manifest synchronized.

- Expanded the Stage 3A chapter plan to require source-bundle regeneration of
  every result table and figure, pooled-MAPE appendix treatment, exact
  abstract/conclusion claims, number mapping, and fresh-clone reproduction.
  Reason: cover the remaining Phase 3 acceptance requirements before LaTeX
  editing begins.

- Completed the Stage 3A paste-ready plan. Reason: add the proposed title,
  full abstract and conclusion, detailed chapter/appendix outline, exact
  methods disclosures, source-map IDs for every claim, B70 support caveat,
  NA decision summary, and audited frozen-protocol wording without editing
  thesis LaTeX.

- Added semantic source-map contexts and report audits for the second Stage 3A
  review. Reason: map every claim to stable source rows, verify no claim uses a
  do-not-cite NA, quantify the group-median comparison, and apply the requested
  protocol wording outside verbatim frozen protocol quotations.

- Added the approved-map LOAFO fold generator and stress-test runner, plus a
  core-neural path that fits only MLP, GraphSAGE, GCN, GAT, and the
  validation-selected best GNN. Reason: execute Phase 2.5 with the two mixed
  systems always in training, identical supported rows across models, and no
  change to the frozen default Phase 2 ladder.

- Added explicit deterministic-method selection so the LOAFO runner does not
  fit the unrequested hierarchical mixed model on the CPU-only stress fold.
  Reason: the first attempted LOAFO run stopped before writing results when
  that extra fit failed to converge; no LOAFO predictions were produced.

- Added the Phase 2.5 tail-summary and LOSO special-case analyzer. Reason:
  derive requested row-tail summaries from saved CV5/LOSO predictions, record
  the six unsupported LOSO folds, and replay the logged mixed-model Powell
  fallback as a saved diagnostic.

- Added the Phase 2.5 report and number-source-map generator. Reason: render
  every requested family, tail, LOSO, protocol, and acceptance item directly
  from saved files while retaining an auditable mapping for thesis prose.

- Reordered the Phase 2.5 report to use the required six-section closeout
  format and clarified the 97 LOAFO-tested-system count. Reason: make the
  orchestrator-facing report directly reviewable without changing any result.

- Expanded the Phase 2.5 source map to enumerate every numeric column in the
  final Phase 2 main tables and key diagnostics, in addition to all requested
  LOAFO and tail values. Reason: support traceability for Phase 3 tables and
  prose without leaving secondary metrics unmapped.

## 2026-11-07

- Implemented the executable Phase 2 model ladder, repeated-CV runner, mixed
  model, graph ablation variants, fold-local best-GNN selection, and
  system-clustered repeated-partition metrics. Reason: run the frozen
  Protocol v2.1 without changing the scientific design after results appear.

- Changed the mixed-model parameterization to use family as the grouping
  intercept with submitter and system variance components. Reason: the initial
  system-group form produced a singular covariance Hessian in some folds while
  retaining the same three random intercepts.

- Fixed repeated-CV bootstrap aggregation to reuse the fixed 476-row
  denominator and tolerate bootstrap draws with no qualifying Spearman group.
  Reason: the first combine attempt repeatedly reloaded source data inside the
  replicate loop and stopped before writing summaries; no model predictions
  were changed.

- Fixed repeated-CV point aggregation to pool all five folds inside each
  partition before averaging partition metrics. Reason: the first summary
  pass computed fold-level metrics, contrary to Protocol v2.1; saved
  predictions were unchanged and were reaggregated.

- Fixed the final repeated-CV table to aggregate the five partition metrics
  into one row per method, and similarly average seed-level partition results
  before seed SDs. Reason: the preceding pass grouped by partition after
  writing them, producing five rows per method.

- Added pooled-MAPE and best-GNN-selection appendices to the repeated-CV
  bundle. Reason: keep pooled MAPE outside the main ranking while preserving
  the pre-registered appendix diagnostic and fold-local architecture choices.

- Added the generated Phase 2 repeated-CV stop report, including the complete
  Protocol v2.1 text, pre-run reconciliations, full LOAFO family proposal, main
  ladder table, paired intervals, and LOSO hold point. Reason: preserve a
  traceable report before any LOSO run begins.

- Added training-stop diagnostics and the non-modifying CV5 diagnostic runner.
  Reason: record GAT prediction variance and training epochs/losses, reconcile
  GraphSAGE feature changes, audit corrected Spearman counts and residuals,
  identify partition drivers, and verify mixed-model convergence.

- Added a same-fold GraphSAGE feature ablation between Phase 1 numeric features
  and Phase 2's CPU-only indicator. Reason: separate feature effects from
  pooling when reconciling the Phase 2 and Phase 1.5 scores.

- Added sharded LOSO/sensitivity runners and pooled-protocol aggregation.
  Reason: execute the remaining pre-registered protocols without changing the
  model ladder or bootstrap definition.

- Added the final co-primary decision and Holm-correction combiner across
  repeated-CV and LOSO secondary comparisons. Reason: produce the complete
  pre-registered inference report while keeping sensitivity non-primary.

- Added the generated Phase 2 complete report covering GAT diagnostics,
  GraphSAGE feature reconciliation, Spearman availability, residuals,
  partition drivers, mixed-model convergence, CV5/LOSO co-primary inference,
  sensitivity, and Holm-adjusted secondary comparisons. Reason: preserve the
  complete evidence trail before the separately gated LOAFO run.

- Report group-median Spearman as missing in repeated-CV, LOSO, and
  sensitivity summaries, bootstrap intervals, and per-group appendices.
  Reason: a group-median model has no within-group rank variation.

- Handled LOSO folds with no supported test rows as empty checkpoints rather
  than attempting to concatenate an empty frame. Reason: holdout-only systems
  can have no target group supported by the training partition.

- Added the frozen fallback optimizer sequence `lbfgs`, `powell`, `nm`, `bfgs`
  for hierarchical mixed-model convergence. Reason: LOSO holdout `holdout_089`
  failed the `lbfgs` convergence check; `powell` converged. The CV5 mixed fits
  all converged on `lbfgs`, so accepted CV5 outputs are unchanged.

- Corrected the diagnostic-only primary helper to compare raw prediction and
  measured logs rather than normalized residuals. Reason: the latter is a
  z-residual summary and cannot be compared with the pre-registered primary
  error. Saved CV5 model outputs were unaffected.

- Extended production-epoch weight equality checks to five fold cases spanning
  ordinary CV, Edge-containing CV, Edge and Datacenter LOSO, and sensitivity.
  Reason: establish broader checkpoint-equality coverage before the final
  Phase 2 bundle.

- Added the proposed 15-family LOAFO map with system counts, supported-row
  counts, integrated-device labels for GB10 and Jetson Thor, CPU-only as a
  stress case, and mixed systems as always-train. Reason: make the full family
  mapping reviewable without running LOAFO.

- Amended the Phase 2 protocol to Version 2 with co-primary repeated five-fold
  and LOSO protocols, a single best-GNN-versus-MLP hypothesis, a +/-0.05
  equivalence rule, Holm-corrected secondary comparisons, zero-accelerator
  handling, and corrected three-row median-group Spearman. Reason: satisfy
  the Phase 1.5 acceptance gates before any Phase 2 model run.

- Recorded the auditable Version 1 to Version 2 protocol diff.
  Reason: preserve the exact clause changes alongside the superseded and new
  protocol hashes.

- Added the pre-registered five-partition Phase 2 fold generator and fold
  tests. Reason: make partition seeds and system assignments auditable in one
  saved file before model fitting.

- Added the Phase 1.5 gate checker for Spearman reconciliation, the
  supporting loss-mask bug boundary, fixed-isolation seed-7 invariance, and
  the production leakage mutation transcript. Reason: provide saved evidence
  for the acceptance gates without running Phase 2.

- Added the canonical Phase 1.5 bundle under
  `results/phase1_5-final-8fbe960/`. Reason: preserve the exact Phase 0 replay,
  ten-seed attribution, corrected sensitivity groups, production leakage
  transcript, provenance, LOAFO proposal, feature contract, and hashes.

- Restricted train and validation loss masks to supported labeled rows even
  when an audit policy retains the leaky all-row graph. Reason: the Phase 1.5
  supported-exclusion stage initially sent NaN normalized targets from
  unsupported groups through the loss and produced non-finite metrics.

- Added the Phase 1.5 reproduction and one-change-at-a-time attribution runner,
  including exact Phase 0 code/data replay and production leakage runtime.
  Reason: separate the effects of identity, folds, masking, support policy, and
  graph isolation without conflating them in a single "legacy" setting.

- Froze `protocol.md` before Phase 2. Reason: establish one primary protocol,
  fixed no-tuning model policy, seed set, evaluated set, and paired comparisons
  before any Phase 2 result can influence design choices.

- Removed the sorted accelerator-count tuple from the sensitivity grouping
  key. Reason: counts as identity split the 105 systems into 101 groups, which
  made the sensitivity protocol unable to hold out systems that differed only
  in node count. The key is now
  `submitter|accelerator|processor`.

- Replaced the one-epoch, mocked-weight perturbation checks with full
  production training and early stopping, saved-state comparison, and a
  reintroduced-leak mutation test. Reason: the prior 130-fold run completed in
  18 seconds because its weight checks used `epochs=1, patience=1`, so it did
  not exercise the production training path.

- Imports the existing evaluation scaffold into Git before Phase 1 edits.
  Reason: create an auditable branch because the model code and canonical data
  were previously outside version control.

- Changed `system_id` to
  `suite|submitter|system_name|system_type`.
  Reason: the prior key merged identically named systems from different
  submitters. The resulting count is 105: 96 Datacenter and 9 Edge.

- Added canonical fold construction for five-fold CV, LOSO, LOAFO, and the
  requested sensitivity grouping.
  Reason: every model must reuse identical system-held-out assignments.

- Added the accelerator-family map, including explicit `review` and
  `ambiguous` statuses.
  Reason: LOAFO grouping requires a documented, auditable family definition;
  mixed-accelerator and integrated/CPU-only cases require human review.

- Changed GAT to four heads with 16 output channels each, concatenated to 64.
  Reason: implement the Phase 1 architecture decision and remove the
  previous one-head code/thesis mismatch.

- Added isolation-first target scaling, vocabulary handling, graph fitting, and
  fold evaluation under `evaluation/`.
  Reason: test systems must not affect vocabularies, scalers, graph degrees,
  baselines, early stopping, or training-time weights.

- Added explicit unknown-category masking at probability 0.1 during graph and
  tabular fitting.
  Reason: LOAFO must test a trained unknown representation rather than an
  untrained fallback vector.

- Excluded target groups with fewer than three training rows from primary
  metrics and recorded them separately as unsupported.
  Reason: eliminate the global fallback that caused catastrophic singleton
  extrapolations.

- Added primary median absolute log-ratio metrics, MAPE, normalized RMSE,
  per-group metrics, system-clustered bootstrap B=2,000, and paired bootstrap
  differences.
  Reason: establish one metric contract for later model comparisons.

- Added `run_phase1.py`, generated figures, `phase1_report.md`, and
  `results_manifest.json` output with code commit, seed, data, fold, and output
  hashes.
  Reason: make one script the source of every reported number and figure.

- Added the fold-level leakage suite under `tests/test_leakage.py`.
  Reason: demonstrate that the old all-row graph fails isolation and the new
  train/validation-only graph passes, including exact weight equality under
  scrambled test metadata.

- Added the stamped Phase 1 GraphSAGE run under
  `results/phase1-final-5c9dfde/`.
  Reason: preserve the ten-seed old-versus-isolated comparison, fold files,
  clustered intervals, figures, leakage transcript, and number manifest used by
  the Phase 1 report.

- Added a synthetic regression test for paired system-clustered bootstrap
  differences and made the runner execute the complete test suite.
  Reason: verify that model differences are paired within bootstrap draws and
  that aggregate metrics have an independent numerical check.
