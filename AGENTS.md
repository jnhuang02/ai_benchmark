# Repository Guidelines

## Current Baseline

This repository currently contains contributor documentation only. Do not assume
that notebooks, source packages, tests, dependency files, or datasets already
exist. Add those components deliberately and update this guide when the
repository structure changes.

## Project Structure & Module Organization

- Keep exploratory notebooks at the repository root with descriptive
  `snake_case` names, such as `mlperf_analysis.ipynb`.
- Put reusable Python code in `src/ai_benchmark/`.
- Put automated tests in `tests/`, with fixtures in `tests/fixtures/`.
- Store small configuration examples in `config/`; never commit credentials.
- Keep downloaded datasets and generated results outside version control unless
  they are small, non-sensitive, and required for verification.

## Build, Test, and Development Commands

Create and activate an isolated environment before running project code:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pytest
```

Once dependencies are defined, install them with
`python -m pip install -r requirements.txt`. Run `pytest -q` for tests and
`python -m compileall src` for a basic source check. Execute notebooks with
Jupyter or `nbconvert` from a clean kernel before committing changed outputs.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules,
functions, and variables; `CapWords` for classes; and `UPPER_SNAKE_CASE` for
constants. Keep notebook cells focused and precede major steps with Markdown
headings. Avoid wildcard imports, silent exception handling, and hardcoded
secrets. Use `ruff` or `black` if either tool is added to the project.

## Testing Guidelines

Use `pytest` and name tests `tests/test_<behavior>.py` with functions named
`test_<expected_behavior>`. Mock paid model APIs and external services so the
default suite runs offline. Every bug fix should include a regression test when
practical; report any tests or notebooks that could not be executed.

## Commit & Pull Request Guidelines

Use Conventional Commits, such as `feat: add MLPerf result parser`,
`fix: handle missing benchmark metrics`, or `docs: update setup instructions`.
Keep commits narrowly scoped. Pull requests should explain the change, list
verification commands and outcomes, link related issues, and include output or
screenshots for user-visible behavior.

## Security & Agent Instructions

Load credentials from environment variables and provide placeholders only in
`.env.example`. Clear sensitive notebook outputs before committing them. Agents
should inspect the current tree before suggesting commands, change only
task-related files, preserve unrelated work, and never commit datasets or
generated artifacts without explicit approval.
