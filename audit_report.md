# Phase 0 Audit Report

Date: 2026-11-07  
Branch: `audit/phase-0`  
Scope: MLPerf Inference v6.1, Closed division, Datacenter and Edge

This audit makes no changes to the thesis text, result figures, trained-model
outputs, or scientific design. The only added files are this report and
`CHANGELOG.md`.

## Audit inputs

| Input | Observed value |
| --- | --- |
| Canonical long table | `mlperf_gnn/data/mlperf_long.csv` |
| Canonical table SHA-256 | `4b3ffce44b0a1c829c805aa4a2c53e18dc6a0d355f7a47101e2886d1f0e8dbb4` |
| Datacenter source SHA-256 | `8b5ccfe2143281e2ae266de03fbec8e57e69b86aeaa5938de0e578fef47d2d1f` |
| Edge source SHA-256 | `e814a07701936a5453909bd94f999681269734e487a90bd38b86f7808bac9c80` |
| Current training script SHA-256 | `5ad04a65637bdcd36ec77ce58710a9a42fedc8132884e8cceaccdf7b0bb86afc` |
| Current thesis source SHA-256 | `ac25ca5af8619f5eb95d492aa26d7e97eec526fbc82099016390a45836f5d65d` |

The canonical table has 476 measurements, 31 target groups, 108 Public IDs,
and 96 values of the current `system_id`.

The Table 6.4 rerun used:

```bash
cd .
arch -arm64 python3 train_graphsage.py \
  --suite all --folds 5 --model graphsage \
  --out-dir /tmp/phase0-repro/graphsage
arch -arm64 python3 train_graphsage.py \
  --suite all --folds 5 --model gcn \
  --out-dir /tmp/phase0-repro/gcn
arch -arm64 python3 train_graphsage.py \
  --suite all --folds 5 --model gat \
  --out-dir /tmp/phase0-repro/gat
```

The GAT command was rerun independently and produced the same result. The
defaults are seed 7, 150 epochs, early-stopping patience 20, 64 hidden units,
two layers, and dropout 0.2. Median row error was recomputed from the saved
row predictions because the script summary does not contain that metric.

## a. Table 6.4 reproduction

| Method | Pooled MAPE | Median group MAPE | Median row MAPE | Normalized RMSE | Median group Spearman | Reproduces? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GraphSAGE | 460.7% | 38.8% | 27.4% | 1.455 | 0.711 | Yes |
| GAT | 762.2% | 37.2% | 27.4% | 1.444 | 0.661 | No |
| GCN | 450,265.6% | 35.9% | 26.6% | 1.480 | 0.749 | Yes |
| Random forest | 557.0% | 67.8% | 23.5% | 1.502 | 0.732 | Yes |

Fresh values before rounding:

| Method | Pooled MAPE | Median group MAPE | Median row MAPE | Normalized RMSE | Median group Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| GraphSAGE | 460.7259% | 38.7772% | 27.4132% | 1.454748 | 0.711303 |
| GAT | 1,509.0558% | 29.7565% | 23.1571% | 1.422438 | 0.820376 |
| GCN | 450,265.5580% | 35.9127% | 26.5627% | 1.480344 | 0.748883 |
| Random forest | 556.9717% | 67.7673% | 23.5078% | 1.502011 | 0.731984 |

Every displayed GAT number in Table 6.4 fails to reproduce. No other Table
6.4 value differs at the table's displayed precision.

The GAT discrepancy is not attributable to a recorded fold or output file.
The current implementation uses one attention head
(`mlperf_gnn/train_graphsage.py`, `bipartite_conv`), while the thesis says four
attention heads per relation. A separate in-memory four-head audit run also
did not reproduce the Table 6.4 GAT row; it produced 8,329.3% pooled MAPE,
29.4% median-group MAPE, 21.7% median-row MAPE, 1.392 normalized RMSE, and
0.752 Spearman. The Table 6.4 GAT row therefore has no traceable run in the
current code/data tree.

The saved `/runs/datacenter_edge` GraphSAGE output is also from a different
run: it reports 505.4% pooled MAPE and 40.6% median-group MAPE. It should not
be cited as the source of Table 6.4.

## b. Inconsistencies

### Section 6.5 error ranges

The stated ranges are not consistent with Table 6.4.

- Current thesis statement: median group error 29.4%-38.8%, median row error
  21.7%-27.4%.
- Table 6.4 across the three GNNs: median group error 35.9%-38.8%, median row
  error 26.6%-27.4%.
- Current code rerun across the three GNNs: median group error 29.8%-38.8%,
  median row error 23.2%-27.4%.

The 29.4% and 21.7% lower endpoints exactly match the separate four-head GAT
audit run (29.4486% and 21.7373%), not the current one-head implementation or
Table 6.4. The Section 6.5 range is therefore a mixture of result snapshots.

### Submission and system counts

The exported counts are:

| Suite | Unique Public IDs (submissions) | Current `system_id` values | Measurements |
| --- | ---: | ---: | ---: |
| Datacenter | 98 | 87 | 457 |
| Edge | 10 | 9 | 19 |
| Total | 108 | 96 | 476 |

A submission here means one unique MLPerf Public ID; one submission can
contain multiple result cells. The current `system_id` is
`suite|system_name|system_type`. It is a suite-scoped system-description key,
not a hardware serial number or a proof of independent physical ownership.

Eight Datacenter system-description keys and one Edge key have more than one
Public ID. Seven Datacenter name/type descriptions also occur under more than
one organization. Thus 87 + 9 = 96 is correct for the current grouping key,
but "96 independent systems" overstates what the key identifies. The defensible
description is "96 suite-scoped system-description groups."

### Whisper counts

Whisper is 25 measurements in total: 23 Datacenter measurements and 2 Edge
measurements. They form two separate target groups because `suite` is part of
the group key.

Table 6.5 reports only the Datacenter Closed Whisper group, so its `n = 23` is
correct. A combined-workload figure showing 25 is also correct if labeled as
Datacenter plus Edge. The 25 measurements must not be described as one target
group.

On the reproduced GraphSAGE run, the Datacenter Whisper group has median APE
21.53% and Spearman 0.9328. The combined 25-row workload has median APE 33.44%
and Spearman 0.8746.

### Edge mean and median error

On the 14 Edge rows belonging to target groups with at least three
measurements:

- Mean APE: **6,289.5298%**
- Median row APE: **99.9079%**
- Median of group median APEs: **99.9941%**
- Rows within +/-50%: **21.4286%**

The thesis statement of 62.9% mean error is a unit error. The computed mean
absolute ratio is 62.8953; converted to percent it is 6,289.5%. The median
99.9% is correct.

The large mean is caused by right-tailed failures. For example, one YOLO-95
Offline prediction has 86,825.6% APE. A large mean with a 99.9% median is
possible and is evidence of severe outliers, not evidence of average
within-100% accuracy.

### Llama 2 Interactive Spearman

For 9 held-out Llama 2 70B Interactive measurements, the reproduced Spearman
correlation is exactly 1/3 = 0.3333. An approximate two-sided p-value is
0.381, with 13 discordant pairs among 36 comparable pairs.

This is a weak positive rank association with substantial uncertainty at
`n = 9`, not "almost no rank correlation." The thesis should say "weak
positive, not distinguishable from zero in this sample."

The Figure 6.2 value of -0.27 is a different quantity: the exploratory
association between accelerator count and measured performance. It is not the
model's predicted-versus-measured rank correlation.

## c. Unseen categorical values

The GNN does not use learned categorical embedding tables. It uses one-hot
vectors whose categories come from training rows only. Each category list has
a reserved `<unknown>` vector at index zero. An unseen test value is mapped to
that vector.

Specifically:

- A test system is inserted as a system node, but `system_id` has no learned
  identity embedding. Its node features are one-hot organization and
  one-hot system type. An unseen organization or system type receives the
  `<unknown>` one-hot vector; a known organization/type uses its trained
  category vector.
- An unseen accelerator name receives the accelerator `<unknown>` one-hot
  vector.
- An unseen processor name receives the processor `<unknown>` one-hot vector.
- An unseen workload target group receives the workload `<unknown>` one-hot
  vector.
- An unseen precision receives the result-node precision `<unknown>` one-hot
  vector.
- Result-node numeric features use training-only median and scale. Missing
  values after scaling are zero.
- Ridge and random forest use `OneHotEncoder(handle_unknown="ignore")`, so an
  unseen categorical value becomes an all-zero block rather than an explicit
  unknown indicator.

Across the five current folds, test systems introduced unseen accelerator
names on 2, 4, 0, 14, and 14 rows; unseen processor names on 23, 16, 1, 18,
and 35 rows; and unseen workload target groups on 3, 2, 0, 2, and 0 rows.
Unseen organizations also occur in every fold. System type is constant within
each suite and is always known.

The one-hot vocabularies, target scaler, numeric scaler, and tabular baselines
are fitted from training systems only. However, this does not establish strict
test-system isolation for the GNN: see the next section.

## Test-system isolation failure

The current model violates the requirement that test systems never influence
training.

`cross_validate` calls `make_graph(frame, train)`, where `frame` is all rows
including the held-out systems. `make_graph` then creates system, accelerator,
processor, workload, and result nodes and all edges from `all_rows`. During
every training epoch, `train_gnn` calls `model(graph)` on the entire graph and
applies the loss only on the training mask.

Consequently, held-out result-node metadata and graph structure participate in
message passing that produces the embeddings used for the training loss.
Test-system information can therefore influence learned weights even though
test labels are excluded from the loss. This affects the reported GNN results;
the tabular baselines are unaffected by this particular path.

Fold assignments are also generated in memory from seed 7 and are not saved in
a reusable keyed file, contrary to the project requirement.

## d. Singleton-group failure

Yes, the normalization fallback contributes directly.

For the fold containing the single end-to-end question-answering row, that
target group is absent from the training partition. `TargetScaler` therefore
uses the global training median as center and global training standard
deviation as scale:

- Global center: 10.883076
- Global scale: 4.603940
- QA signed log target: -0.767707
- QA normalized target: -2.530611
- GraphSAGE normalized prediction: -1.044341
- Inverse-transformed prediction: 434.845670 tasks/s
- Measured value: 0.464076 tasks/s
- APE: 936.0139%

The fallback alone is already severe: a normalized prediction of zero (the
global median) would invert to 53,267.2 tasks/s. The graph prediction is not
at that center, but it is still on the global rather than a QA-specific scale.
Thus the failure combines the unseen-group normalization fallback with poor
extrapolation by the model.

The claim that five singleton groups account for 85.7% of GraphSAGE's summed
APE also fails to reproduce. In the current Table 6.4 run, those five groups
account for **46.2140%** of summed APE. The QA row alone accounts for 42.6808%.
The 85.7384% value belongs to the different saved run whose overall GraphSAGE
MAPE is 505.4%, not 460.7%.

## e. Throughput versus latency/energy groups

The 31 target groups split as:

- 25 throughput groups: 3 Queries/s, 9 Samples/s, 1 Tasks/s, 12 Tokens/s.
- 6 latency groups: 5 Latency (ms), 1 Latency (s).
- 0 energy or power groups.

At row level, 448 measurements have higher-is-better throughput targets and
28 have lower-is-better latency targets.

## Additional contradiction affecting the current narrative

The thesis describes four GAT attention heads, but the implementation uses one
head. This is both a text/code mismatch and a table-provenance problem.

The current fresh GAT run has lower median group error, lower median row error,
and higher Spearman correlation than Table 6.4 reports, but its pooled MAPE is
higher. The table should not be interpreted until one implementation and one
saved run are selected.

## Phase report

### 1. What was done

- Traced the v6.1 Datacenter and Edge inputs and verified the canonical row,
  submission, system-key, and target-group counts.
- Reran GraphSAGE, GCN, and GAT with the documented five-fold protocol, seed 7,
  and current code, writing only under `/tmp/phase0-repro`.
- Recomputed median row error, Edge diagnostics, singleton-group contribution,
  and rank-correlation diagnostics from saved row predictions.
- Audited target normalization, unseen-category handling, fold construction,
  and test-system participation in graph training.
- Recomputed throughput, latency, and energy target-group counts.
- Created the required Git branch and recorded this audit.

### 2. Files changed

- Added `audit_report.md`.
- Added `CHANGELOG.md`.
- No thesis text, result figure, saved result CSV, source-data file, or model
  code was edited.
- Pre-existing uncommitted changes to `main.tex`, `main.pdf`, and
  `references.bib` were preserved.

### 3. Numbers that changed (old -> corrected)

- GAT pooled MAPE: 762.2% -> 1,509.1% on current one-head code; no saved run
  supports the old value.
- GAT median group MAPE: 37.2% -> 29.8%.
- GAT median row MAPE: 27.4% -> 23.2%.
- GAT normalized RMSE: 1.444 -> 1.422.
- GAT Spearman: 0.661 -> 0.820.
- Section 6.5 GNN group-error range: 29.4%-38.8% -> 29.8%-38.8% on current
  code, or 35.9%-38.8% for Table 6.4's GNN rows.
- Section 6.5 GNN row-error range: 21.7%-27.4% -> 23.2%-27.4% on current
  code, or 26.6%-27.4% for Table 6.4's GNN rows.
- Singleton share of GraphSAGE summed APE: 85.7% -> 46.2%.
- Edge mean APE: 62.9% -> 6,289.5%; the median remains 99.9%.

### 4. Surprising or contradictory findings

- Table 6.4 mixes results from more than one implementation or run. Its GAT
  row cannot be produced with the current one-head code.
- Section 6.5 and the singleton-error figure draw from additional stale runs.
- The current GNN graph lets held-out system metadata and edges participate in
  training message passing.
- The "96 independent systems" phrasing is stronger than the data key supports.
- The Edge mean-error statement is a percentage-scale error.
- A Spearman value of 0.33 is weak, not effectively zero.

### 5. Open questions

- Before Phase 1 fold files are frozen, confirm whether `system_id` should
  remain `suite|system_name|system_type` despite seven Datacenter descriptions
  occurring under multiple organizations.
- Confirm the audited GAT architecture: one attention head as implemented or
  four as currently written in the thesis.
- Phase 1 must establish the compliant inductive procedure: train on a graph
  containing only training and validation rows, then introduce test metadata
  only during inference.

### 6. Acceptance criteria

- Reproduce Table 6.4: **Yes**, with the GAT row explicitly failing
  reproduction.
- List non-reproducing numbers: **Yes**.
- Resolve Section 6.5 ranges: **Yes**.
- Define submission versus system: **Yes**, with a system-identity decision
  still open.
- Reconcile Whisper counts: **Yes**.
- Verify and explain Edge mean/median: **Yes**.
- Evaluate Llama 2 Interactive Spearman language: **Yes**.
- Document unseen categorical handling: **Yes**.
- Diagnose singleton-group failure: **Yes**.
- Count throughput versus latency/energy groups: **Yes**.
- Preserve test-system isolation: **No**; current GNN code violates it.
- Deliver `audit_report.md`: **Yes**.

