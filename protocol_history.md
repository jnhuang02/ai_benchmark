# Protocol Amendment Log

## Version 2.2, 2026-11-07

Status: post-result documentation addendum. This entry corrects the protocol
record and compute bookkeeping but does not alter the hash-pinned
`protocol.md`, which remains Version 2.1 with SHA-256
`d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e`.

LOAFO addendum:

1. Before the LOAFO run, orchestration specified the LOAFO model set (group
   median, Ridge, random forest, MLP, GraphSAGE, GCN, and GAT), neural seeds
   `7`, `11`, and `16`, omission of the hierarchical mixed model, and
   family-level paired inference. These choices were recorded in the
   orchestration request but were not in the hash-pinned Version 2.1 file.
2. Version 2.1 uses the phrase "LOAFO stress-test estimate" but provides no
   rule for pooling family-level errors. Because the orchestration prohibited
   pooling across headline, small/integrated, and CPU-only families, LOAFO
   results are descriptive family-level results, not a single pooled estimate.

Compute-table erratum:

The Version 2.1 compute table overstated outer-fold counts because it treated
all repeated-CV partition-fold cells as 125 independent outer folds and
described sensitivity as 25 outer folds. The executed designs contain 25
repeated-CV outer folds and 5 sensitivity outer folds. At the documented
`0.36` seconds-per-fit planning rate, the corrected table is:

| Protocol | Outer folds | Neural seeds | Deterministic fits | Neural fits | Total fits | Planning runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5-fold x 5 partitions | 25 | 10 | 125 | 2,250 | 2,375 | 855 seconds, 14.25 minutes |
| LOSO | 105 | 3 | 525 | 2,835 | 3,360 | 1,209.6 seconds, 20.16 minutes |
| Sensitivity | 5 | 10 | 25 | 450 | 475 | 171 seconds, 2.85 minutes |
| Total | 135 | - | 675 | 5,535 | 6,210 | 2,235.6 seconds, 37.26 minutes |

This erratum changes no model, fold assignment, prediction, metric, test, or
reported performance number.

Post-lock analysis deviation:

- `S3A-SHRINK-001`: the sparse-group partial-pooling secondary analysis was
  requested for the full CV5 and LOSO ladder, but Version 2.1 does not define
  a unique shrinkage posterior. It omits the prior, group-size shrinkage
  formula, variance estimator, treatment of center and scale, and the
  post-shrinkage evaluation rule. The analysis was therefore not run, no
  empirical-Bayes convention was invented after inspecting results, and no
  shrinkage performance number is reportable. The primary exclusion analysis
  remains unchanged.

## Version 2.1, 2026-11-07

Status: frozen before any Phase 2 model run.

Supersedes Version 2 from commit `cce2f43`:

- Version 2 SHA-256:
  `9062c13cfe2ae50e9a5da2a0e92f48f8d2d2ef32191cd0d38942ac13b40296f4`

Amendments:

1. Both co-primary protocols must produce the same primary label before a
   claim is made.
2. The interval-to-label mapping is explicit, including intervals that exclude
   zero but cross an equivalence boundary.
3. Repeated-CV metrics are computed per partition and then averaged; LOSO
   metrics use pooled out-of-fold predictions per seed.
4. The compute estimate contains `17,610` fits and `105.66` estimated minutes
   using the documented `0.36` seconds-per-fit planning rate.
5. LOSO neural seeds are reduced to `7`, `11`, and `16`; repeated CV and
   sensitivity keep seeds 7-16.
6. The proposed 15-family LOAFO map labels GB10 and Jetson Thor as integrated
   devices, labels CPU-only as a stress case outside the headline, and keeps
   mixed systems always-train.

## Version 2, 2026-11-07

Status: frozen before any Phase 2 model run.

Pre-run correction: the sensitivity seed is `20265107`, equal to
`20261107 + 4000`. The first V2 draft contained a transcription typo; the
corrected V2 hash is the one recorded in `protocol.sha256`.

Supersedes Version 1:

- commit: `d3bc835`
- protocol SHA-256:
  `eb14e445f34c83ee817f59b35555f086be402d031534d4415815c575579fe811`

Amendments:

1. Grouped five-fold CV is now co-primary with LOSO and is repeated over five
   recorded partitions, seeds `20261107`-`20261111`.
2. The single primary hypothesis is best GNN minus MLP under both co-primary
   protocols, with a `+/-0.05` equivalence margin and four-label decision rule.
3. Best GNN is selected inside each outer fold from validation primary error
   with a fixed tie order. Test labels never participate.
4. Partition and seed variability is averaged inside each system-clustered
   bootstrap draw.
5. The secondary comparisons receive Holm correction across eight
   protocol-comparison tests.
6. Pooled MAPE is appendix-only.
7. Zero-accelerator systems use `log1p(total_accelerators)` plus a CPU-only
   indicator in every model.
8. Median-group Spearman uses at least three evaluated rows per qualifying
   group, reconciling it with original Table 6.4.
9. Sensitivity is labeled modest because 78 of 90 groups are singleton
   systems.
10. LOAFO remains approval-gated; families with fewer than three systems are
    reported separately, and mixed-accelerator systems are always-train.

The V1 and V2 files are both in Git, so the exact diff and hashes are
auditable.
