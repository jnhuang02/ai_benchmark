# Repository Guidelines

## Current Baseline

This repository is the reproducibility package for the thesis *Predicting
Inference Performance for Unseen Systems: A Comparison of Graph and Tabular
Models*. It predicts MLPerf Inference v6.1 Closed-division Datacenter and Edge
performance for held-out systems from system, hardware, and workload metadata,
comparing tabular models and an MLP with heterogeneous GNNs (GraphSAGE, GCN,
GAT). Read `README.md` before running anything.

The thesis reports a **single final evaluation that includes the
accelerator-memory features** (`AI_BENCHMARK_FEATURE_SET=memory`). The older
feature set (`frozen`, the default) reproduces the protocol v2.1 results and
stays available, but the thesis no longer reports it.

## Agent Skill

The MLPerf skill is `.agents/skills/mlperf-benchmarking/`. Read its
`SKILL.md` for MLPerf suite, execution, result, device, GNN, or roofline work.
Invoke it explicitly with `$mlperf-benchmarking` when needed.

## Project Structure & Module Organization

- `evaluation/`: canonical data, features, graph, models, metrics, folds, and
  evaluation code. `evaluation/data.py` holds the feature-set switch;
  `evaluation/bundles.py` holds the result-bundle switch.
- `prepare_data.py`, `prepare_memory_features.py`: build
  `data/mlperf_long.csv` and `data/system_memory.csv`.
- `run_phase2_cv.py`, `run_phase2_protocol.py`, `run_phase2_loafo.py`: model
  runs (repeated CV, LOSO/sensitivity, leave-one-accelerator-family-out).
- `combine_phase2_*.py`, `run_phase2_cv5_diagnostics.py`, `analyze_phase2_5.py`,
  `run_stage3a_corrections.py`: combine shards, diagnostics, tails, and LOAFO
  support.
- `generate_thesis_results.py`, `scripts/generate_eda_figures.py`: thesis
  tables and figures. `exploratory_memory_ablation.py` compares the frozen and
  memory-feature runs.
- `folds/`, `config/`, `data/`: saved fold assignments, accelerator-family map,
  and datasets. `results/` holds result bundles and is git-ignored.
- `protocol.md` is hash-pinned (`protocol.sha256`). Never edit it; record
  changes in `protocol_history.md` and `CHANGELOG.md`.
- `src/ai_benchmark/`, `scripts/`, `results/datacenter_edge/`, and
  `results/architecture_ablation/` are retained from the earlier public
  baseline; the root-level `evaluation/` pipeline is canonical.

## Build, Test, and Development Commands

Create the locked environment (Python 3.14) and run from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m unittest discover -s tests
```

On Apple silicon, prefix Python commands with `arch -arm64`. The suite reports
16 known failures in `tests/test_graph_identity_alignment.py`, documented in the
README; any other failure is new. `pytest -q` also covers the retained package
tests.

To reproduce the thesis run, set both switches and follow the commands in the
thesis reproducibility appendix:

```bash
export AI_BENCHMARK_FEATURE_SET=memory AI_BENCHMARK_BUNDLES=memory
```

`AI_BENCHMARK_THESIS_DIR` overrides where `generate_thesis_results.py` writes
(default: a sibling `thesis_overleaf/` directory).

## Result Bundles

- Never overwrite an existing bundle; write a new output path. The run,
  combine, and analysis scripts refuse to write into an existing directory,
  but `generate_thesis_results.py` overwrites its output directory.
- Run manifests record the feature set, memory-table hash, git commit, and
  whether the tree was dirty. Commit or stash before a run that will be
  reported.
- `analyze_phase2_5.py` and `run_stage3a_corrections.py` end with checks that
  hold only for the frozen run (a Powell-fallback replay, an exact test count,
  and a family count). On the memory run they fail after writing the CSVs the
  thesis needs; verify those outputs by hand instead of weakening the checks.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules,
functions, and variables; `CapWords` for classes; and `UPPER_SNAKE_CASE` for
constants. Avoid wildcard imports, silent exception handling, and hardcoded
secrets. New behavior that changes results goes behind an opt-in environment
switch so existing results still reproduce.

## Testing Guidelines

Name tests `tests/test_<behavior>.py` with functions named
`test_<expected_behavior>`. Keep the suite offline. Every bug fix should
include a regression test when practical; report tests that could not be run.

## Commit & Pull Request Guidelines

Use Conventional Commits, such as `feat: add MLPerf result parser`,
`fix: handle missing benchmark metrics`, or `docs: update setup instructions`.
Keep commits narrowly scoped and add a `CHANGELOG.md` entry with the reason for
any evaluation change. Pull requests should explain the change, list
verification commands and outcomes, and link related issues.

## Thesis Text

The thesis lives in a separate Overleaf project. When editing results prose,
every claim in the abstract, discussion, or conclusion must respect the
decision rule: a comparison receives a label only when repeated CV and LOSO
agree. Trace every number to a result file before writing it.

## Security & Agent Instructions

Load credentials from environment variables. Agents should inspect the tree
before suggesting commands, change only task-related files, preserve unrelated
work, and never commit datasets or generated result bundles without approval.
