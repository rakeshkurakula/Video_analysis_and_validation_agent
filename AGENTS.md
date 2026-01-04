# Repository Guidelines

## 🤖 System Agents

This repository implements a multi-agent validation system.

### 1. Analysis Orchestrator (`analysis_agent.py`)
- **Role**: The central "brain" that orchestrates the entire validation lifecycle.
- **Responsibilities**:
    - Parsing Gherkin feature files and mapping them to "Golden Steps".
    - Ingesting raw logs (Hercules/Chrome) and identifying test execution boundaries.
    - Dispatching evidence to the Vision LLM for verification.
    - Aggregating results into the final Deviation Report.

### 2. Vision LLM Analyzer (`vision_llm_analyzer.py`)
- **Role**: The "eyes" of the system, responsible for semantic visual verification.
- **Key Capabilities**:
    - **Asynchronous Parallel Pipeline**: Uses `asyncio` to process video frames in parallel batches. This results in a ~70% speedup compared to sequential analysis.
    - **Multi-Provider Support**: Seamlessly switches between Groq (Llama-3), OpenAI (GPT-4o), and Google (Gemini) based on configuration.
    - **Contextual Vision**: Evaluates images specifically against the *context* of a test step (e.g., looking for "Size XL" specifically, rather than just reading all text).

### 3. Deviation Classifier (`deviation_classifier.py`)
- **Role**: The "judge" that applies the 6-class taxonomy to every step.
- **Logic**:
    - Distinguishes between **Hallucinated** (Log says yes, Video says no) and **Observed** (Log says yes, Video says yes).
    - Identifies **Deviation-Skipped** steps when a test terminates early, preventing false "failure" noise for subsequent steps.

---


## Project Structure & Module Organization
The repo centers on the analysis agent script `analysis_agent.py`. Hercules artifacts live under `opt/`:
- `opt/log_files/` for planner logs
- `opt/proofs/` for videos and screenshots
- `opt/output/` for JUnit XML and generated deviation reports

If you expand the codebase, keep Python packages under `src/` (e.g., `src/video_agent/`), tests under `tests/` (e.g., `tests/test_validation.py`), and any static artifacts or sample media under `assets/` or `data/` with a short provenance note.

## Build, Test, and Development Commands
Dependency management uses UV via `pyproject.toml`. Common workflows:
- `uv sync` - create/update `.venv` and install dependencies from `pyproject.toml` (writes `uv.lock`).
- `uv add <package>` - add a runtime dependency.
- `uv add --dev <package>` - add a development dependency.
- `uv run python -m <package>` - run an entry point once a package exists.
- `uv run pytest` - run tests after pytest is added.

## Coding Style & Naming Conventions
Use 4 spaces for indentation and avoid tabs. Follow Python naming conventions: `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep filenames lowercase with underscores (e.g., `video_validator.py`). No formatter or linter is configured yet; if you add one (such as Ruff), record the exact command and version here.

## Testing Guidelines
There is no test framework configured and no coverage target. Once tests exist, put them in `tests/` and name files `test_*.py`. Prefer `uv run pytest` as the standard test command and include regression tests for reported bugs.

## Commit & Pull Request Guidelines
Git history currently contains only "Initial commit," so no established commit message convention exists. Until a convention is set, use short, imperative summaries (<= 72 characters) and add a body when context is needed. PRs should include a brief description, verification steps, and links to any related issues; include sample outputs or screenshots when behavior changes.

## Configuration & Secrets
Keep local configuration in `.env` (already ignored by `.gitignore`) and never commit secrets or large binary assets. Commit `uv.lock` once dependencies exist; `.venv` remains ignored. If you add sample media files, document their source and licensing.

## Agent Notes
If you add new directories, scripts, or toolchains, update this guide so contributors can onboard quickly.
