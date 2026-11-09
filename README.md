# Predicting MLPerf Inference Performance for Unseen Systems

This repository is the reproducibility package for the thesis study
**Heterogeneous Graph Neural Networks for Predicting MLPerf Inference
Performance from System and Workload Metadata**. It predicts task-specific
MLPerf Inference v6.1 performance on unseen systems from system, accelerator,
processor, and workload metadata using graph and tabular models.

The cautious headline result is copied from the thesis abstract and is not
recomputed by this README: under the frozen protocol, the evaluation did not
establish that graph structure improves prediction; the best-GNN-minus-MLP
differences are `+0.115597` under repeated CV and `+0.104155` under LOSO, and
the frozen label remains **inconclusive**.

These are model-validation results, not official MLPerf submissions. MLPerf
result exports do not provide hardware-profiler counters for a causal
utilization, memory-bandwidth, or roofline claim.

## License status

No project license has been selected yet. The author has been asked to choose
whether code uses a software license such as MIT and whether results use a
license such as CC-BY-4.0. Until that decision is recorded, this repository
contains no `LICENSE` file and does not grant a project-wide license beyond
rights supplied by GitHub's terms and applicable third-party licenses.

The public MLCommons input data described below remains subject to its own
license.

## Repository layout

- `evaluation/`, `prepare_data.py`, `train_graphsage.py`: canonical data,
  feature, graph, model, metric, fold, and evaluation code.
- `run_phase1.py`, `run_phase1_5.py`, `run_phase1_5_gates.py`: Phase 1 and
  Phase 1.5 evaluation, leakage, gate, and sensitivity runs.
- `run_phase2_prerun_checks.py`, `run_phase2_cv.py`,
  `run_phase2_protocol.py`, `run_phase2_cv5_diagnostics.py`,
  `run_phase2_loafo.py`: Phase 2 execution entry points.
- `combine_phase2_cv.py`, `combine_phase2_protocol.py`,
  `combine_phase2_final.py`, `analyze_phase2_5.py`,
  `run_stage3a_corrections.py`: bundle combination and post-run analyses.
- `exploratory_posthoc_b1.py`, `exploratory_posthoc_b2.py`,
  `exploratory_posthoc_b4.py`: explicitly exploratory saved-prediction,
  attrition, and GCN isolation replays.
- `protocol.md`: hash-pinned protocol v2.1.
- `protocol.sha256`: expected protocol digest.
- `protocol_history.md`: versions 1, 2, 2.1, the v2.2 LOAFO addendum, the
  compute erratum, and the unexecuted `S3A-SHRINK-001` deviation.
- `protocol_v1_to_v2.diff`, `protocol_v2_to_v2_1.diff`: auditable protocol
  differences.
- `config/`, `folds/`, `data/`: accelerator-family configuration, saved fold
  assignments, and the canonical normalized table.
- `results/`: frozen result bundles and manifests used by the thesis.
- `generated/`: frozen generated LaTeX tables, generated figures, source
  snapshots, and `generated_asset_manifest.json`.
- `figures/`: the eight thesis figures.
- `phase*_report.md`, `stage3*.md`, `audit_report.md`: phase, stage, leakage,
  replay, and verification reports.
- `phase2_5_number_source_map.csv`,
  `stage3b_number_source_map.csv`: claim-level number-source maps.
- `tests/`, `requirements.txt`, `requirements-lock.txt`: verification suite
  and environment specifications.

The `src/ai_benchmark/`, `scripts/`, and earlier `results/datacenter_edge/`
and `results/architecture_ablation/` material is retained from the repository's
prior public baseline. The root-level `evaluation/` pipeline and the
`results/phase*` bundles are the canonical thesis evidence.

## Data source and scope

The analysis uses 476 published measurements from 105 submitter-aware system
descriptions in the MLPerf Inference v6.1 Closed-division Datacenter and Edge
exports. The scope is Inference Datacenter and Inference Edge; Mobile and Tiny
are excluded.

Source:
[`mlcommons/inference_results_v6.1`](https://github.com/mlcommons/inference_results_v6.1).

The upstream repository was public and identified by GitHub as Apache-2.0 at
audit time. The observed upstream `HEAD` on the access date was
`4bb63cd28eb136be221bc801a9e38c66c8a7553f`. Access date: **2026-11-08**.

The checked-in raw exports and canonical table are:

| Path | Rows | SHA-256 |
| --- | ---: | --- |
| `data/raw/Table - Inference_Full Data_data.csv` | 457 | `8b5ccfe2143281e2ae266de03fbec8e57e69b86aeaa5938de0e578fef47d2d1f` |
| `data/raw/Table - Inference_edge.csv` | 19 | `e814a07701936a5453909bd94f999681269734e487a90bd38b86f7808bac9c80` |
| `data/mlperf_long.csv` | 476 | `a6853e1e2dc777404ddf9f5e0546679fef758df38bc4ebe5bba21e0c0bb735dd` |

The two source exports are UTF-16, tab-separated files. They are public
benchmark metadata and measurements, not accuracy-compliance records or
hardware profiler traces.

## Environment

The production bundles used:

```text
Python   3.14.2
NumPy    2.4.4
pandas   3.0.2
scikit-learn 1.8.0
statsmodels 0.15.0
SciPy    1.17.1
PyTorch  2.11.0
PyTorch Geometric 2.7.0
Matplotlib 3.10.9
```

Create the locked environment from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

On the Apple-silicon machine used for the reported run, the commands below use
`arch -arm64 python3` to select the native interpreter. Use
`python -m unittest discover -s tests -v` for the canonical suite; `pytest -q`
also covers the retained package tests.

The audited source is commit `8725ab8baae8e9fa01fe3b1b9d6dc4cef5bf542f`.
That commit deliberately adds a graph identity-alignment reproduction test
that reports 16 failures. The public package preserves the test and code
unchanged; those failures are reported rather than masked.

## Git LFS

The complete result bundles are tracked with Git LFS. Current Git LFS paths are:

```text
results/phase1_5-final-8fbe960/attribution_predictions_by_seed.csv
results/phase2-cv5-final-aeeb4b4/predictions_by_partition_seed.csv
results/phase2-sensitivity-final-a689ba5/predictions_by_fold_seed.csv
```

Install Git LFS before cloning or run it after cloning:

```bash
git lfs install
git lfs pull
git lfs ls-files
```

No file in the canonical bundles exceeds GitHub's 100 MB limit.

## Protocol and frozen label

`protocol.md` is Version 2.1, frozen before the final Phase 2 model comparison.
Its required digest is:

```text
d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e
```

Verify it without editing it:

```bash
shasum -a 256 protocol.md
shasum -a 256 -c protocol.sha256
```

"Frozen by commit hash" here means the protocol text, hash, model choices, and
fold plan were committed before the final comparison and the result manifests
record that hash and their source commit. It does **not** mean the protocol was
externally registered, time-stamped by a third party, or otherwise independently
immutable.

The v2.2 LOAFO addendum is deliberately outside the hash-pinned v2.1 file. It
documents the LOAFO model set, seeds, omission of the hierarchical mixed model,
and family-level inference. LOAFO is descriptive and unpooled.

Known deviations are:

- `S3A-SHRINK-001` was not executed because v2.1 did not define a unique
  shrinkage posterior, prior, variance estimator, scaling rule, and evaluator.
  No shrinkage performance number is reportable.
- The v2.2 LOAFO addendum is history-only and is not covered by the protocol
  hash.
- The v2.2 compute erratum corrects planning arithmetic from 255 asserted outer
  fold cells to 25 repeated-CV outer folds plus 5 sensitivity outer folds. It
  changes no model, split, prediction, metric, or result value.

## Canonical result bundles

The thesis tables use these bundles:

| Purpose | Path |
| --- | --- |
| Phase 1 evaluation, leakage, figures | `results/phase1-final-5c9dfde/` |
| Phase 1.5 attribution and sensitivity | `results/phase1_5-final-8fbe960/` |
| Phase 1.5 fixed/legacy leakage gates | `results/phase1_5-gates-f82e4d2/` |
| Legacy-versus-fixed pre-run replay | `results/phase2-prerun-d539fc2/` |
| Repeated-CV saved predictions and summaries | `results/phase2-cv5-final-aeeb4b4/` |
| Repeated-CV diagnostics | `results/phase2-cv5-diagnostics-77c4752/` |
| LOSO saved predictions and summaries | `results/phase2-loso-final-a689ba5/` |
| Sensitivity saved predictions and summaries | `results/phase2-sensitivity-final-a689ba5/` |
| LOAFO family results | `results/phase2-loafo-final-1a65664/` |
| Tail summary and LOSO empty-fold detail | `results/phase2-5-analysis-4faac06/` |
| Support, test-suite, and partial-pooling evidence | `results/phase2-5-corrections-e97bb93/` |
| Combined co-primary and Holm results | `results/phase2-final-f51e57a/` |
| Saved-prediction bootstrap replay | `results/exploratory-posthoc-b1-saved-prediction-bootstrap/` |
| Supported-row attrition audit | `results/exploratory-posthoc-b2-supported-row-attrition/` |

The frozen `tail_summary.csv` is
`results/phase2-5-analysis-4faac06/tail_summary.csv`.
The frozen `main_table_by_family.csv` is
`results/phase2-loafo-final-1a65664/main_table_by_family.csv`.

## Reproduce result bundles

Run from the repository root with the locked environment active. Existing
bundles are intentionally not overwritten: when a command's canonical output
path is already populated, use a new output path and compare it with the frozen
bundle. The commands below show the original canonical destinations for a clean
workspace.

First verify the canonical table. It reproduces from the two public raw files:

```bash
arch -arm64 python3 prepare_data.py \
  --source data/raw \
  --out data/mlperf_long.csv
arch -arm64 python3 -m unittest discover -s tests -v
```

Phase 1 and Phase 1.5:

```bash
arch -arm64 python3 run_phase1.py \
  --output results/phase1-final-5c9dfde
arch -arm64 python3 run_phase1_5.py \
  --output results/phase1_5-final-8fbe960
arch -arm64 python3 run_phase1_5_gates.py \
  results/phase1_5-gates-f82e4d2
```

Phase 2 legacy/fixed replay and repeated CV:

```bash
arch -arm64 python3 run_phase2_prerun_checks.py \
  results/phase2-prerun-d539fc2

for partition in 0 1 2 3 4; do
  arch -arm64 python3 run_phase2_cv.py \
    --output "results/phase2-cv5-p${partition}-rerun" \
    --partitions "${partition}"
done

arch -arm64 python3 combine_phase2_cv.py \
  --input \
    results/phase2-cv5-p0-rerun \
    results/phase2-cv5-p1-rerun \
    results/phase2-cv5-p2-rerun \
    results/phase2-cv5-p3-rerun \
    results/phase2-cv5-p4-rerun \
  --output results/phase2-cv5-final-rerun
arch -arm64 python3 run_phase2_cv5_diagnostics.py \
  results/phase2-cv5-diagnostics-rerun
```

The diagnostics script reads the canonical repeated-CV bundle named
`results/phase2-cv5-final-aeeb4b4/`, so reproduce there in a clean workspace or
rename the verified rerun bundle before running diagnostics.

LOSO uses four deterministic shards and seeds `7 11 16`:

```bash
for shard in 0 1 2 3; do
  arch -arm64 python3 run_phase2_protocol.py \
    --protocol loso \
    --output "results/phase2-loso-shard${shard}-rerun" \
    --seeds 7 11 16 \
    --shard-index "${shard}" \
    --shard-count 4
done

arch -arm64 python3 combine_phase2_protocol.py \
  --protocol loso \
  --input \
    results/phase2-loso-shard0-rerun \
    results/phase2-loso-shard1-rerun \
    results/phase2-loso-shard2-rerun \
    results/phase2-loso-shard3-rerun \
  --output results/phase2-loso-final-rerun
```

Sensitivity uses seeds `7` through `16`:

```bash
arch -arm64 python3 run_phase2_protocol.py \
  --protocol sensitivity \
  --output results/phase2-sensitivity-rerun \
  --seeds 7 8 9 10 11 12 13 14 15 16
arch -arm64 python3 combine_phase2_protocol.py \
  --protocol sensitivity \
  --input results/phase2-sensitivity-rerun \
  --output results/phase2-sensitivity-final-rerun
```

LOAFO and the Phase 2.5 analyses:

```bash
arch -arm64 python3 run_phase2_loafo.py \
  --output results/phase2-loafo-final-rerun
arch -arm64 python3 analyze_phase2_5.py \
  --output results/phase2-5-analysis-rerun
arch -arm64 python3 run_stage3a_corrections.py \
  --output results/phase2-5-corrections-rerun
arch -arm64 python3 combine_phase2_final.py \
  results/phase2-final-rerun
```

`analyze_phase2_5.py` also reads the four included canonical LOSO shard
manifests. `run_stage3a_corrections.py` reads the canonical CV and LOAFO
bundles by their frozen paths.

Exploratory saved-prediction and attrition replays:

```bash
arch -arm64 python3 exploratory_posthoc_b1.py \
  --output results/exploratory-posthoc-b1-rerun
arch -arm64 python3 exploratory_posthoc_b2.py \
  --output results/exploratory-posthoc-b2-rerun
arch -arm64 python3 exploratory_posthoc_b4.py \
  --output results/exploratory-posthoc-b4-rerun
```

The B1 and B2 frozen bundles are included. B4's executable GCN isolation and
normalization replay is included, but no frozen B4 result bundle was available
in the audited source tree, so this package does not claim that a frozen B4
bundle exists.

## Reports and generated artifacts

Regenerate the phase reports after their input bundles exist:

```bash
arch -arm64 python3 write_phase2_5_report.py
arch -arm64 python3 write_phase2_cv5_report.py
arch -arm64 python3 write_phase2_final_report.py
arch -arm64 python3 write_stage3a_plan_report.py
arch -arm64 python3 write_stage3b_report.py
```

Regenerate the eight thesis figures from their compact checked-in source data:

```bash
MPLCONFIGDIR=.cache/matplotlib \
  arch -arm64 python3 scripts/generate_thesis_figures.py
```

Generated table and figure map:

| Artifact | Generator | Frozen output |
| --- | --- | --- |
| `tab_data_coverage.tex`, `tab_protocol.tex`, `tab_mlperf_parameters.tex` | `generate_thesis_results.py` | `generated/` |
| `tab_model_ladder.tex`, `tab_primary_decision.tex`, `tab_secondary_holm.tex` | `generate_thesis_results.py` | `generated/` |
| `tab_tail_summary.tex`, `tab_partition_errors.tex`, `tab_gat_diagnostic.tex` | `generate_thesis_results.py` | `generated/` |
| `tab_loafo_headline.tex`, `tab_loafo_pairing.tex`, `tab_loafo_other.tex` | `generate_thesis_results.py` | `generated/` |
| `tab_provenance.tex`, `generated_asset_manifest.json` | `generate_thesis_results.py` | `generated/` |
| `fig_model_ladder.pdf`, `fig_primary_decision.pdf`, `fig_tail_errors.pdf`, `fig_loafo_group_median.pdf` | `generate_thesis_results.py` | `generated/` |
| `fig01_data_scope` through `fig08_evaluation_pipeline` | `scripts/generate_thesis_figures.py` | `figures/` |

To recreate generated tables and figures from the included frozen result
bundles:

```bash
arch -arm64 python3 generate_thesis_results.py \
  --output generated
```

The checked-in snapshots are sufficient to regenerate the tables and figures
without refitting. In this publishing checkout, the generator resolves its
output against a sibling `thesis_overleaf` directory; a standalone artifact
check must therefore use a temporary sibling directory rather than
`--output generated`:

```bash
artifact_root="$(mktemp -d /tmp/ai-benchmark-artifacts.XXXXXX)"
git clone https://github.com/jnhuang02/ai_benchmark.git \
  "${artifact_root}/ai_benchmark"
cd "${artifact_root}/ai_benchmark"
mkdir -p ../thesis_overleaf/generated
cp -a generated/source_snapshots \
  ../thesis_overleaf/generated/source_snapshots
arch -arm64 python3 generate_thesis_results.py \
  --snapshot-only \
  --output ../thesis_overleaf/generated
diff -qr generated ../thesis_overleaf/generated
```

This command does not currently produce a byte-identical tree in every
environment. The published package preserves the thesis-build artifacts
unchanged. In the audited checkout, regenerated PDFs differ at the binary level,
the source manifest changes the repository directory name, and
`tab_primary_decision.tex` reflects a later generator caption revision rather
than the frozen thesis-build caption. These are packaging/reproduction
discrepancies, not changes to result values. The fresh-clone report records the
exact paths rather than claiming byte equality.

## Exact Appendix C workflow

The thesis Appendix C commands are reproduced verbatim below. They assume a
sibling Overleaf-owned thesis checkout named `thesis_overleaf` at thesis
commit `031a83d077be9f9cf6fc14c032bdc70c00e353f1`, and that this repository is
checked out as `mlperf_gnn`. The GitHub package does not publish the thesis
source and does not push to Overleaf.

```bash
cd mlperf_gnn
arch -arm64 python3 -m unittest discover -s tests -v
arch -arm64 python3 write_phase2_5_report.py
arch -arm64 python3 write_stage3a_plan_report.py
arch -arm64 python3 generate_thesis_results.py
cd ../thesis_overleaf
latexmk -pdf main.tex
cd ../mlperf_gnn
arch -arm64 python3 write_stage3b_report.py
```

The appendices also prescribe the snapshot-only fresh-clone check:

```bash
cd mlperf_gnn
arch -arm64 python3 generate_thesis_results.py --snapshot-only
cd ../thesis_overleaf
latexmk -pdf main.tex
```

and the complete integrity check:

```bash
cd mlperf_gnn
arch -arm64 python3 stage3c_final_checks.py --run-tests
```

Because the generated snapshots are included under `generated/`, they can be
regenerated in a temporary sibling directory as shown above. The LaTeX rebuild
still requires an authorized local thesis checkout because thesis text is
intentionally not part of this repository.

## Verification and exclusions

The package excludes caches, `__pycache__`, `.venv`, editor files, scratch run
directories, smoke outputs, noncanonical and superseded bundles, and model
checkpoints that are not required for thesis artifact reproduction. The
canonical saved-prediction files remain in the bundles.

Before publishing, verify:

```bash
shasum -a 256 -c protocol.sha256
arch -arm64 python3 -m unittest discover -s tests -v
arch -arm64 python3 stage3c_final_checks.py --run-tests
! rg -n -I -e '/Users/[A-Za-z0-9._-]+' \
  -e '/home/[A-Za-z0-9._-]+' \
  --glob '!.git/**' \
  --glob '!.venv/**' \
  .
```

The final fresh-clone verification report is recorded in
`stage3c_fresh_clone_check.txt`, with the machine-readable result in
`stage3c_final_checks.json`.
