# ReCode: Reliable Recursive Self-Improvement

Research harness for **Reliable RSI** — a system that self-improves a coding agent across economic value, truthfulness, instruction following, and safety. 

## Overview

ReCode builds on the [ReCodeAgent](./recode.pdf) methodology: a frozen judge evaluates an agent on multiple benchmarks, a meta-improver analyzes failure traces and edits the agent's code, and a zipper schedule ensures no data is ever reused across iterations.

**Benchmarks:**

| Axis (Weight) | Benchmark | Size |
|------|-----------|------|
| Economic value (50%) | [GDPval](https://huggingface.co/datasets/openai/gdpval) | 220 tasks |
| Truthfulness (20%) | [TruthfulQA](https://github.com/sylinrl/TruthfulQA), [SimpleQA](https://openai.com/index/simpleqa/) | 817 + 4,326 |
| Instruction following (15%) | [IFEval](https://arxiv.org/abs/2311.07911) | 541 prompts |
| Safety (15%) | [HarmBench](https://github.com/centerforaisafety/HarmBench), [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) | 510 + 1,000 |

## Repository Structure

```
src/
  data/
    loaders/           # Per-benchmark dataset loaders (GDPval, TruthfulQA, etc.)
    registry.py        # Central dataset registry
    sampler.py         # Unified sampling across benchmarks
    splitters/         # Zipper split for non-overlapping iteration slices
    types.py           # Shared types (Sample, BenchmarkType, Split)
  eval/
    agents/            # Pluggable agent backends
      base.py          #   BaseAgent ABC + AgentResult dataclass
      claude_code.py   #   Claude Code via claude-agent-sdk
      gemini_cli.py    #   Gemini CLI via subprocess
      codex.py         #   OpenAI Codex CLI via subprocess
    evaluators/        # Scoring backends
      base.py          #   BaseEvaluator ABC + EvalResult dataclass
      gdpval.py        #   Keyword heuristic (fast, free, approximate)
      gdpval_judge.py  #   Gemini Pro frozen judge (accurate, costs API calls)
    runner.py          # GDPvalRunner — orchestrates agent + evaluator
    visualize.py       # Paper-quality plots (trajectory, sector, failure taxonomy)
scripts/
  download_datasets.py # Download all benchmarks from HuggingFace
  run_gdpval_eval.py   # CLI entry point for running evaluations
  demo_sampler.py      # Demo: load and sample from the unified interface
docs/
  plans/               # Design documents and implementation plans
  EVAL_GUIDE.md        # Detailed evaluation guide
results/               # Output directory (gitignored)
  <agent>_<timestamp>/
    traces.json        #   Raw agent output (prompt, response, tool calls)
    eval.json          #   Scores only (summary + per-task rubric breakdown)
  workspace_<agent>/   #   Per-task isolated workspaces with deliverable files
  plots/               #   Generated visualizations
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install project + dependencies
pip install -e .
pip install claude-agent-sdk python-docx openpyxl pymupdf python-pptx xlrd python-dotenv
pip install matplotlib seaborn  # for visualization (optional)

# Set up API keys
cp .env.example .env  # then edit .env with your keys
```

Your `.env` file needs:
```
GEMINI_API_KEY=...      # Required for frozen judge scoring
ANTHROPIC_API_KEY=...   # Required for Claude Code agent
OPENAI_API_KEY=...      # Required for Codex agent
```

### Download Datasets

```bash
# All benchmarks
python scripts/download_datasets.py

# Specific ones
python scripts/download_datasets.py --only gdpval truthfulqa
```

Datasets are saved to `data/raw/` as parquet files.

## Running Evaluations

### Quick Test (1 task)

```bash
python scripts/run_gdpval_eval.py --agent claude --n 1 --verbose
```

### Full Run with a Specific Agent

```bash
# Claude Code — 22 tasks from zipper slice S1
python scripts/run_gdpval_eval.py --agent claude --slice S1

# Gemini CLI — 22 tasks from slice S1
python scripts/run_gdpval_eval.py --agent gemini --slice S1

# OpenAI Codex — 22 tasks from slice S1
python scripts/run_gdpval_eval.py --agent codex --slice S1

# All 220 tasks (takes several hours)
python scripts/run_gdpval_eval.py --agent claude --all
```

### Scoring

By default, the **Gemini Pro frozen judge** scores all tasks (requires `GEMINI_API_KEY`). For quick local testing without API calls, use `--no-judge` for the keyword heuristic:

```bash
# Default: Gemini judge (accurate)
python scripts/run_gdpval_eval.py --agent claude --slice S1

# Fast keyword heuristic (no API calls, less accurate)
python scripts/run_gdpval_eval.py --agent claude --slice S1 --no-judge
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent NAME` | `claude` | Agent backend: `claude`, `gemini`, `codex` |
| `--model NAME` | auto | Override the agent's model |
| `--n N` | 3 | Number of tasks to sample |
| `--all` | — | Run all 220 tasks |
| `--slice NAME` | — | Run a zipper slice (`S1`–`S6`, `vault`) |
| `--max-turns N` | 10 | Max agent turns per task |
| `--concurrency N` | 1 | Parallel task execution |
| `--no-judge` | — | Use keyword heuristic instead of Gemini judge |
| `--judge-model NAME` | auto | Override judge model |
| `--output DIR` | auto | Output directory |
| `--seed N` | 42 | Random seed for sampling |
| `--verbose` | — | Debug logging |

### Output

Results are saved to `results/<agent>_<timestamp>/` as two files:

**`traces.json`** — raw agent output for debugging and meta-improvement:
```json
{
  "traces": [{
    "task_id": "0419f1c3-...",
    "agent": "claude",
    "prompt": "You are a Property Manager...",
    "response": "I created a PIP document...",
    "tool_calls": [{"tool": "Write", "input": "..."}],
    "workspace_dir": "results/workspace_claude/0419f1c3-d66"
  }]
}
```

**`eval.json`** — scores for analysis and comparison:
```json
{
  "summary": {"agent": "claude", "avg_score": 0.94, "num_tasks": 22},
  "scores": [{
    "task_id": "0419f1c3-...",
    "score": 80.0,
    "max_score": 85.0,
    "normalized_score": 0.94,
    "rubric_breakdown": {"criterion A": 2.0, "criterion B": 0.0},
    "criteria_met": 51,
    "num_criteria": 52
  }]
}
```

Each task also gets an **isolated workspace** at `results/workspace_<agent>/<task_id>/` containing any files the agent produced (`.docx`, `.xlsx`, etc.). These are extracted and included in scoring.

## Visualizing Results

```python
from src.eval.visualize import ResultsViz

viz = ResultsViz("results/")
viz.list_runs()                                    # Summary table of all runs
viz.label_runs({"0": "Baseline", "1": "Improved"}) # Assign display names

viz.trajectory(save_path="results/plots/trajectory.png")       # Score across iterations
viz.slice_comparison(save_path="results/plots/slices.png")     # Per-task bar chart
viz.failure_taxonomy(save_path="results/plots/failures.png")   # Top failed criteria
viz.model_comparison(save_path="results/plots/models.png")     # Side-by-side agents
viz.dashboard(save_path="results/plots/dashboard.png")         # 4-panel summary

# Per-sector breakdown (needs dataset samples for sector mapping)
from src.data.registry import DatasetRegistry
registry = DatasetRegistry()
registry.load_available()
samples = registry.get_samples("gdpval")
viz.sector_breakdown(samples=samples, save_path="results/plots/sectors.png")
```

Visualization is optional — if `matplotlib` is not installed, plotting methods return `None` with a warning instead of crashing.

## How the Evaluation Pipeline Works

```
┌──────────────┐     prompt      ┌─────────────────┐
│  GDPval      │ ──────────────> │  Agent Backend   │
│  Sample      │                 │  (Claude/Gemini/ │
│  (from data/ │                 │   Codex)         │
│   pipeline)  │                 └────────┬────────┘
└──────────────┘                          │ response + files
                                          v
                                ┌─────────────────┐
                                │  File Extractor  │  ← reads .docx, .xlsx,
                                │  (workspace)     │    .pdf, .pptx, .xls
                                └────────┬────────┘
                                         │ response + file contents
                                         v
                                ┌─────────────────┐
                                │  Evaluator       │  ← keyword heuristic
                                │  (or Gemini      │    OR Gemini Pro judge
                                │   frozen judge)  │
                                └────────┬────────┘
                                         │ EvalResult
                                         v
                                ┌─────────────────┐
                                │  results/        │
                                │  traces.json     │
                                │  eval.json       │
                                └─────────────────┘
```

1. **Data pipeline** loads GDPval tasks via `DatasetRegistry` → `UnifiedSampler`
2. **Agent backend** receives the task prompt and works in an isolated per-task workspace
3. **File extractor** reads deliverable files (`.docx`, `.xlsx`, `.pdf`, `.pptx`) from the workspace
4. **Evaluator** scores the response + file contents against the rubric
5. **Results** are saved as separate trace and eval files

## How the RSI Loop Works

1. **Zipper split** divides GDPval's 220 tasks into 6 non-overlapping dev slices (22 each) + 88 held-out test vault
2. At iteration *t*, the **meta-improver** reads traces from slice *t−1*, identifies failures, designs and implements fixes
3. A **frozen judge** (Gemini Pro, cross-model-family) scores the improved agent on the fresh slice *t*
4. An **acceptance rule** (≥1.5pp delta, ≥12/22 wins, no critical regressions) decides keep or rollback
5. After 6 iterations, the held-out test vault is evaluated **once** to confirm generalization

## Requirements

- Python >= 3.11
- Claude Code CLI (bundled with `claude-agent-sdk`)
- Gemini CLI (`npm install -g @anthropic-ai/gemini-cli`) — for Gemini agent
- Codex CLI — for Codex agent
- API keys: Anthropic, Google (Gemini), OpenAI — see `.env`
- See `pyproject.toml` for Python dependencies
