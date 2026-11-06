# AI Benchmark

This repository contains the reproducibility package for the thesis study
**Heterogeneous Graph Neural Networks for Predicting MLPerf Inference
Performance from System and Workload Metadata**.

The current experiment predicts task-specific MLPerf Inference v6.1
performance using heterogeneous GraphSAGE, GCN, and GAT models over systems,
accelerators, processors, workloads, and result records. It compares those
models with target-group median, Ridge regression, and random forest baselines
under complete-system-held-out validation.

The reported values are model-validation results, not official MLPerf
submissions. The MLPerf exports contain performance measurements but no
general accuracy-compliance field or hardware-profiler counters.

## Repository contents

- `mlperf_graphsage.ipynb`: end-to-end preprocessing, graph model, baselines,
  plots, and per-workload diagnostics.
- `src/ai_benchmark/prepare_data.py`: two-file CSV loader, hash verification,
  target construction, and canonical table generation.
- `src/ai_benchmark/train_graphsage.py`: graph builder, GraphSAGE/GCN/GAT
  models, grouped evaluation, and baseline implementations.
- `data/raw/`: the two MLPerf Inference v6.1 Closed-division CSV exports.
- `data/mlperf_long.csv`: the normalized 476-row analysis table.
- `results/datacenter_edge/`: five-fold summary, per-group metrics, and
  row-level held-out predictions.
- `results/architecture_ablation/`: shared-fold GraphSAGE, GCN, and GAT
  comparison with per-group metrics and row-level predictions.
- `tests/`: data-contract, graph-construction, and model-construction checks.

## Input scope

The experiment reads exactly:

- `Table - Inference_Full Data_data.csv` (Datacenter v6.1, 457 rows)
- `Table - Inference_edge.csv` (Edge v6.1, 19 rows)

Together they contain 476 measurements from 96 independent system
identifiers. Both files are Closed-division data. Mobile and Tiny are excluded
from this experiment. The pipeline rejects byte-identical input files.

See [`data/README.md`](data/README.md) for source links, hashes, and counts.

## Setup

Python 3.10 or newer is required. The reported run was tested with Python
3.14.2, pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, PyTorch 2.11.0,
PyTorch Geometric 2.7.0, and Matplotlib 3.10.9.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
python -m ipykernel install --user --name ai-benchmark \
  --display-name "Python (ai-benchmark)"
```

The loosely bounded dependencies are in `pyproject.toml` and
`requirements.txt`. The tested top-level versions from the reported run are
recorded in `requirements-lock.txt`; operating-system and accelerator
differences may require installing PyTorch through its platform-specific
index.

## Reproduce the result

Prepare the canonical table:

```bash
python -m ai_benchmark.prepare_data
```

Run the five-fold heterogeneous GraphSAGE evaluation:

```bash
python -m ai_benchmark.train_graphsage \
  --suite all --folds 5 --model graphsage
```

Run GraphSAGE, GCN, and GAT on identical folds:

```bash
python -m ai_benchmark.train_graphsage \
  --suite all --folds 5 \
  --models graphsage gcn gat
```

`--models` runs every listed architecture with the same fold assignments and
random seeds. GAT uses four attention heads by default; set
`--attention-heads` to change that value.

Or open `mlperf_graphsage.ipynb` and run it from top to bottom. The notebook
uses the checked-in `data/raw` files by default and falls back to
`~/Downloads/mlcommons_results_latest`.

Available model choices are `graphsage`, `gcn`, and `gat`. Use `--loso` for
the slower leave-one-system-out evaluation.

## Validation

```bash
pytest -q
```

On this Apple-silicon machine, use `arch -arm64 ./.venv/bin/python -m pytest -q`
when the shell reports `x86_64`.

The tests verify input hashes, expected row and system counts, target
construction, heterogeneous graph node/edge types, and construction of the
GraphSAGE and ablation models.

## Results

The committed five-fold run and its interpretation are documented in
[`results/datacenter_edge/README.md`](results/datacenter_edge/README.md).
Pooled MAPE must be interpreted alongside the per-workload table because five
singleton workload groups account for most of the aggregate error.

The completed architecture ablation is documented in
[`results/architecture_ablation/README.md`](results/architecture_ablation/README.md).
GCN has the best median workload-group error and rank correlation, while
GraphSAGE is substantially more stable on pooled error and filtered mean error.
GAT is intermediate and remains sensitive to sparse extrapolation cases.

## Thesis figures

The `figures/` directory contains eight publication-ready PDF/PNG figures
covering the current thesis data scope, exploratory associations,
metadata-only baselines, GraphSAGE/random-forest comparison, workload-level
diagnostics, sparse-group and Edge limitations, and the evaluation pipeline.

The figure generator is `scripts/generate_thesis_figures.py`. Its compact,
reviewable data inputs are in `data/thesis_figures/`, and the provenance notes
distinguish current-thesis reported metrics from the MLPerf v6.1 coverage
counts. Suggested thesis placements and LaTeX captions are in
[`notes/figure_insert_guide.md`](notes/figure_insert_guide.md).

Regenerate the figure set with:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/generate_thesis_figures.py
python -m unittest discover -s tests -v
```

## Scientific limits

The published result metadata is insufficient for a defensible roofline or
causal bottleneck analysis by itself. It lacks consistent peak compute for
the executed precision, memory bandwidth, utilization, interconnect traffic,
batch and sequence configuration, and operator-level timing. GraphSAGE
outputs are performance estimates and do not become measured MLPerf results
without a corresponding benchmark run.
