# ReCode — Reliable Recursive Self-Improvement

Research harness for COLM 2026 paper. Evolves coding agents via SkyDiscover across GDPval professional tasks + safety benchmarks (AgentHarm, ToolEmu), scored by a frozen LLM judge, with zipper schedules ensuring no data reuse.

## Quick Reference

```bash
source .venv/bin/activate

# Baselines
python scripts/run_baseline.py --agents claude codex --slices S1 S2 S3 S4 S5 S6 S7 S8

# Evolution — Track A (GDPval only)
python -m src.evolve.run_evolve --seed openai --tier slow --slices S1 S2 S3 S4 S5 S6 S7 S8

# Evolution — Track B (50% GDPval + 50% safety)
python -m src.evolve.run_evolve --seed openai --tier slow --slices S1 S2 S3 S4 S5 S6 S7 S8 --safety-weight 0.5

# Quick sanity check
python scripts/run_gdpval_eval.py --agent claude --n 1 --verbose
```

## Architecture

- **`src/data/`** — Dataset loading, registry, zipper splitters
  - `loaders/` — GDPval, AgentHarm, ToolEmu, ASB (+ TruthfulQA, SimpleQA, IFEval, HarmBench, OR-Bench)
  - `splitters/zipper.py` — GDPval S1-S8/E1-E2 split
  - `splitters/safety_zipper.py` — AgentHarm + ToolEmu S1-S8/E1-E2 split
- **`src/eval/agents/`** — Pluggable agent backends inheriting `BaseAgent` ABC
  - `claude_code.py` — Claude Code (claude-opus-4-6)
  - `codex.py` — Codex CLI (gpt-5.4, xhigh)
  - `gemini_cli.py` — Gemini CLI (gemini-2.5-pro)
  - `custom.py` — Custom evolvable agent (Gemini SDK)
- **`src/eval/evaluators/`** — Scoring backends
  - `gdpval.py` — Fast keyword heuristic (no API calls)
  - `gdpval_judge.py` — Frozen LLM judge (accurate, costs API)
- **`src/eval/runner.py`** — `GDPvalRunner` orchestrates agent + evaluator
- **`src/evolve/`** — Evolution pipeline (see `src/evolve/README.md`)
  - `evaluator.py` — Fitness function (GDPval + safety composite)
  - `seeds/` — Starting agent code per provider (openai, anthropic, gemini)
- **`results/`** — Output dir (gitignored)

## Key Conventions

- Python 3.11+, managed with `uv`
- API keys live in `.env` — **never commit or edit `.env`**
- All agents run tasks in isolated per-task workspaces
- The frozen judge model must never be changed mid-experiment
- Zipper slices are deterministic: S1-S8 (dev), E1-E2 (eval, used once at the end)
- Safety zipper split aligns ToolEmu to the same S1-S8/E1-E2 structure
- Evolution fitness: `combined = (1-w) * gdpval + w * toolemu_safety` where w = `--safety-weight`

## Benchmarks

| Benchmark | Type | Size | Purpose |
|-----------|------|------|---------|
| GDPval | Economic | 220 | Professional task completion (fitness signal) |
| ToolEmu | Safety | 144 | Cautious tool-use under ambiguity (ICLR 2024) |

AgentHarm (208 tasks) is loaded but not used in the fitness signal — gpt-5.4's
RLHF refuses all jailbreak prompts at 100%, giving no evolution signal. ToolEmu
measures what evolution actually changes: cautious vs reckless tool-use behavior.

## Design Plan

See `docs/plans/2026-03-07-reliable-rsi-design.md` for the full implementation plan.
See `docs/safety_benchmarks.md` for safety benchmark details and examples.
