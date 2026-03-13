# ReCode — Reliable Recursive Self-Improvement

Research harness for COLM 2026 paper. Evaluates coding agents (Claude Code, Gemini CLI, Codex) on GDPval professional tasks, scored by a frozen Gemini judge, with a zipper schedule ensuring no data reuse across RSI iterations.

## Quick Reference

```bash
# Activate venv (always do this first)
source .venv/bin/activate

# Run a quick 1-task sanity check
python scripts/run_gdpval_eval.py --agent claude --n 1 --verbose

# Run a full zipper slice (22 tasks)
python scripts/run_gdpval_eval.py --agent claude --slice S1

# Run baseline across all agents and slices
python scripts/run_baseline.py --slices S1 S2 --verbose

# Keyword heuristic scoring (no API calls)
python scripts/run_gdpval_eval.py --agent claude --n 3 --no-judge

# Download datasets
python scripts/download_datasets.py --only gdpval
python scripts/download_datasets.py --gdpval-files-only
```

## Architecture

- **`src/data/`** — Dataset loading, registry, zipper splitter, unified sampler
- **`src/eval/agents/`** — Pluggable agent backends inheriting `BaseAgent` ABC
  - `claude_code.py` — Claude Code via `claude-agent-sdk`
  - `gemini_cli.py` — Gemini CLI via subprocess
  - `codex.py` — Codex CLI via subprocess
- **`src/eval/evaluators/`** — Scoring backends inheriting `BaseEvaluator` ABC
  - `gdpval.py` — Fast keyword heuristic (no API calls)
  - `gdpval_judge.py` — Frozen Gemini judge (accurate, costs API)
- **`src/eval/runner.py`** — `GDPvalRunner` orchestrates agent + evaluator + file extraction
- **`src/eval/visualize.py`** — Paper-quality plots (trajectory, sector, failure taxonomy)
- **`scripts/`** — CLI entry points (`run_gdpval_eval.py`, `run_baseline.py`, `download_datasets.py`)
- **`results/`** — Output dir (gitignored). Each run produces `traces.json` + `eval.json`

## Key Conventions

- Python 3.11+, managed with `uv`
- API keys live in `.env` — **never commit or edit `.env`**
- All agents run tasks in isolated per-task workspaces under `results/workspace_<agent>/<task_id>/`
- New agents must subclass `BaseAgent` (see `src/eval/agents/base.py`) and implement `run()` + `name()`
- New evaluators must subclass `BaseEvaluator` (see `src/eval/evaluators/base.py`)
- The frozen judge model must never be changed mid-experiment (cross-model-family grading consistency)
- Zipper slices are deterministic: S1–S8 (dev), E1–E2 (eval, used once at the end)
- Results are split into `traces.json` (raw output for meta-improvement) and `eval.json` (scores only)

## Design Plan

See `docs/plans/2026-03-07-reliable-rsi-design.md` for the full implementation plan. Currently in **Phase 1** (baseline runs).
