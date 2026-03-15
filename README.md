# ReCode: Reliable Recursive Self-Improvement

Research harness for **Reliable RSI** — a system that evolves a coding agent's source code to improve task completion while monitoring safety. Built for the COLM 2026 paper.

## Overview

ReCode uses SkyDiscover (evolutionary search) to mutate an agent's Python source code across deterministic zipper slices of GDPval tasks. A frozen LLM judge scores each variant. The best code carries forward to the next slice, forcing generalization across unseen tasks.

**Two evolution tracks** test whether RSI degrades safety:
- **Track A**: Evolve on GDPval only (capability-blind to safety)
- **Track B**: Evolve on 50% GDPval + 50% safety (AgentHarm + ToolEmu)

**Benchmarks:**

| Axis | Benchmark | Size | Used In |
|------|-----------|------|---------|
| Economic value | [GDPval](https://huggingface.co/datasets/openai/gdpval) | 220 tasks | Evolution fitness + baseline |
| Safety (harmful actions) | [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | 208 behaviors | Track B fitness + post-hoc eval |
| Safety (unsafe tools) | [ToolEmu](https://github.com/ryoungj/ToolEmu) | 144 cases | Track B fitness + post-hoc eval |

**Baseline agents:**

| Agent | Model | Avg Score (S1-S8) |
|-------|-------|-------------------|
| Codex CLI | gpt-5.4 (xhigh) | 75.3% |
| Claude Code | claude-opus-4-6 | 69.1% |

## Repository Structure

```
src/
  data/
    loaders/           # Per-benchmark dataset loaders
      gdpval.py        #   220 professional tasks (economic value)
      agentharm.py     #   208 harmful tool-use behaviors (safety)
      toolemu.py        #   144 unsafe tool-use test cases (safety)
      asb.py           #   51 adversarial attack scenarios (security)
    registry.py        # Central dataset registry
    splitters/
      zipper.py        #   GDPval zipper split (S1-S8, E1-E2)
      safety_zipper.py #   Safety benchmarks zipper split
    types.py           # Shared types (Sample, BenchmarkType)
  eval/
    agents/            # Pluggable agent backends
      base.py          #   BaseAgent ABC + rate limiter
      claude_code.py   #   Claude Code (claude-opus-4-6)
      codex.py         #   Codex CLI (gpt-5.4, xhigh)
      gemini_cli.py    #   Gemini CLI (gemini-2.5-pro)
      custom.py        #   Custom evolvable agent (Gemini SDK)
    evaluators/
      gdpval.py        #   Keyword heuristic (fast, free)
      gdpval_judge.py  #   Frozen LLM judge (accurate, costs API)
    runner.py          # GDPvalRunner — orchestrates agent + evaluator
  evolve/
    run_evolve.py      # Main evolution orchestrator
    cli.py             # CLI arguments + logging
    config.py          # SkyDiscover config builder
    evaluator.py       # Fitness function (GDPval + safety composite)
    slices.py          # Zipper slice utilities
    seeds/             # Starting agent code per provider
      openai.py        #   gpt-5.4 seed
      anthropic.py     #   claude-opus-4-6 seed
      gemini.py        #   gemini-2.5-pro seed
scripts/
  run_baseline.py      # Baseline eval: all agents x all slices
  run_gdpval_eval.py   # Single agent evaluation
  download_datasets.py # Download benchmarks from HuggingFace
results/               # Output directory (gitignored)
docs/
  safety_benchmarks.md # Safety benchmark reference + examples
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# API keys
cp .env.example .env  # then edit with your keys
```

`.env` needs:
```
GEMINI_API_KEY=...      # Required: frozen judge + Gemini agent
ANTHROPIC_API_KEY=...   # Required: Claude Code agent
OPENAI_API_KEY=...      # Required: Codex agent + evolved agent
```

### Install Agent CLIs

```bash
npm install -g @google/gemini-cli    # Gemini CLI
npm install -g @openai/codex         # Codex CLI
npm install -g @anthropic-ai/claude-code  # Claude Code
```

### Download Datasets

```bash
# All benchmarks
python scripts/download_datasets.py

# GDPval reference files (509 files)
python scripts/download_datasets.py --gdpval-files-only

# ToolEmu (manual — not on HuggingFace)
mkdir -p data/raw/toolemu
curl -sL https://raw.githubusercontent.com/ryoungj/ToolEmu/main/assets/all_cases.json \
  -o data/raw/toolemu/all_cases.json
```

## Running Baselines

```bash
# Full baseline: claude + codex across S1-S8 + E1-E2
python scripts/run_baseline.py --agents claude codex

# Specific slices
python scripts/run_baseline.py --agents claude codex --slices S1 S2 S3

# Resume after interruption
python scripts/run_baseline.py --resume results/baseline_main

# Quick test (keyword heuristic, no API cost)
python scripts/run_baseline.py --slices S1 --no-judge --max-turns 5
```

## Running Evolution

```bash
# Track A: GDPval only (no safety signal)
python -m src.evolve.run_evolve \
  --seed openai --tier slow \
  --slices S1 S2 S3 S4 S5 S6 S7 S8 \
  --iterations 5 --sample-size 3 --verbose

# Track B: 50% GDPval + 50% safety (AgentHarm + ToolEmu)
python -m src.evolve.run_evolve \
  --seed openai --tier slow \
  --slices S1 S2 S3 S4 S5 S6 S7 S8 \
  --safety-weight 0.5 --safety-samples 3 \
  --iterations 5 --sample-size 3 --verbose

# Quick test (1 slice, 1 iteration)
python -m src.evolve.run_evolve --seed openai --slices S1 --iterations 1 --sample-size 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `openai` | Agent provider: `openai`, `anthropic`, `gemini` |
| `--tier` | `fast` | Model tier: `fast` (cheap) or `slow` (frontier) |
| `--slices` | `S1 S2 S3` | Zipper slices to evolve across |
| `--iterations` | `5` | Mutations per slice |
| `--sample-size` | `3` | GDPval tasks per evaluation |
| `--safety-weight` | `0.0` | Safety weight: 0.0 (Track A) to 0.5 (Track B) |
| `--safety-samples` | `3` | Safety tasks per evaluation |
| `--mutation-model` | `gemini/gemini-2.5-flash` | LLM for code mutations |
| `--search` | `adaevolve` | Search strategy |
| `--full-eval` | off | Full 22-task eval after each slice |

See [`src/evolve/README.md`](src/evolve/README.md) for detailed documentation on how evolution works, the fitness signal, and safety integration.

## How the RSI Loop Works

```
Seed agent (seeds/openai.py)
      │
      ├─ Eval on S1 (3 GDPval + 3 safety tasks) → baseline score
      │
      │  SkyDiscover: mutate code → eval → select (5 iterations)
      │
      │  Best code carries to S2 (completely different tasks)
      │
      ├─ Eval on S2 → score → mutate → select
      │
      ├─ ... S3 through S8 ...
      │
      └─ Final eval on E1, E2 (held-out, run once)
```

Each slice has unseen tasks. Improvements must generalize. The fitness is:
- **Track A**: `combined_score = gdpval_score`
- **Track B**: `combined_score = 0.50 * gdpval_score + 0.50 * safety_score`

## Key Question

> Does evolving an agent to be better at professional tasks cause it to become less safe?

- Track A measures capability evolution with no safety guardrail
- Track B adds safety to the fitness signal
- Comparing A vs B reveals the capability cost of maintaining safety during RSI

## Requirements

- Python >= 3.11, managed with `uv`
- Agent CLIs: Claude Code, Gemini CLI, Codex CLI
- API keys: Anthropic, Google (Gemini), OpenAI
- See `pyproject.toml` for Python dependencies
