# Repository Guidelines

## Project Structure & Module Organization

This is a notebook-first AI evaluation project. Keep working notebooks at the repository root with descriptive `snake_case` names, such as `llm_judge.ipynb`. As the project grows, place reusable Python in `src/<module>/`, automated tests in `tests/`, and small fixtures in `tests/fixtures/`. Keep datasets and generated artifacts out of Git unless they are small, non-sensitive, and required by tests.

## Build, Test, and Development Commands

Create an isolated environment before running notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip jupyter pytest
jupyter lab llm_judge.ipynb
```

Execute a notebook from a clean kernel to verify it end to end:

```bash
python -m nbconvert --to notebook --execute llm_judge.ipynb \
  --output /tmp/llm_judge.executed.ipynb
```

If dependencies are added, record pinned versions in `requirements.txt` and install them with `python -m pip install -r requirements.txt`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, `snake_case` functions and variables, `CapWords` classes, and `UPPER_SNAKE_CASE` constants. Keep notebook cells focused and introduce each major step with a Markdown heading. Avoid wildcard imports and silent exception handling. Pin dependency versions and document required Python versions when adding packaging metadata.

## Testing Guidelines

No automated test suite exists yet. Add `pytest` tests as `tests/test_*.py`, with test functions named `test_<behavior>`. Mock external model-provider calls; default tests must not require paid API access. Run the suite with `pytest -q`, then execute changed notebooks from a clean environment before committing.

## Commit & Pull Request Guidelines

This repository has no commit history yet, so use Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, or `chore:`. Keep each commit narrowly scoped. Pull requests should summarize the change, list verification commands and results, link related issues, and include screenshots or output examples for notebook behavior changes.

## Security & Configuration

Load provider credentials from environment variables such as `OPENAI_API_KEY`; never hardcode or commit keys. Ignore `.env` files and provide `.env.example` with placeholders only. Notebook outputs are public once committed, so clear sensitive or oversized outputs before staging.

The untracked `llm_judge.ipynb` currently contains hardcoded credentials. Do not commit it until those keys are removed and rotated. Agents should stage only task-related files and must not include that notebook accidentally.
