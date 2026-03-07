# ReCode: Reliable Recursive Self-Improvement

Research harness for **Reliable RSI** — a system that self-improves a coding agent across economic value, truthfulness, instruction following, and safety. 

## Overview

ReCode builds on the [ReCodeAgent](./recode.pdf) methodology: a frozen judge evaluates an agent on multiple benchmarks, a meta-improver analyzes failure traces and edits the agent's code, and a zipper schedule ensures no data is ever reused across iterations.

**Benchmarks:**

| Axis | Benchmark | Size |
|------|-----------|------|
| Economic value | [GDPval](https://huggingface.co/datasets/openai/gdpval) | 220 tasks |
| Truthfulness | [TruthfulQA](https://github.com/sylinrl/TruthfulQA), [SimpleQA](https://openai.com/index/simpleqa/) | 817 + 4,326 |
| Instruction following | [IFEval](https://arxiv.org/abs/2311.07911) | 541 prompts |
| Safety | [HarmBench](https://github.com/centerforaisafety/HarmBench), [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) | 510 + 1,000 |

## Project Structure

```
src/
  data/
    loaders/       # Per-benchmark dataset loaders (GDPval, TruthfulQA, etc.)
    registry.py    # Central dataset registry
    sampler.py     # Unified sampling across benchmarks
    splitters/     # Zipper split for non-overlapping iteration slices
    types.py       # Shared types (Sample, BenchmarkType, Split)
scripts/
  download_datasets.py   # Download all benchmarks from HuggingFace
  demo_sampler.py        # Demo: load and sample from the unified interface
docs/
  plans/                 # Design documents and implementation plans
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Download Datasets

```bash
# All benchmarks
python scripts/download_datasets.py

# Specific ones
python scripts/download_datasets.py --only gdpval truthfulqa
```

Datasets are saved to `data/raw/` as parquet files.

### Verify

```bash
python scripts/demo_sampler.py
```

## How It Works

1. **Zipper split** divides GDPval's 220 tasks into 6 non-overlapping dev slices (22 each) + 88 held-out test vault
2. At each iteration, the **meta-improver** reads traces from the previous slice, identifies failures, designs and implements fixes
3. A **frozen judge** (cross-model-family) scores the improved agent on the next fresh slice
4. An **acceptance rule** (significance threshold + win rate + no critical regressions) decides whether to keep or rollback
5. After 6 iterations, the held-out test vault is evaluated **once** to confirm generalization

## Requirements

- Python >= 3.11
- See `pyproject.toml` for dependencies
