# GDPval Evaluation with Claude Code Agent SDK

How to run GDPval benchmark evaluations using the `claude-agent-sdk` to programmatically send tasks to Claude Code and score responses.

## Prerequisites

```bash
# 1. Python 3.10+ with venv
python3 --version  # must be >= 3.10

# 2. Activate the project venv
source .venv/bin/activate

# 3. Install dependencies (claude-agent-sdk bundles the Claude Code CLI)
pip install claude-agent-sdk

# 4. Ensure you have a valid Anthropic API key / Claude Code session
#    The SDK uses the bundled Claude Code CLI, which requires authentication.
#    If you haven't already, run:
claude  # interactive login, then Ctrl+C after authenticated

# 5. Download GDPval data (if not already done)
python scripts/download_datasets.py --only gdpval
```

## Quick Start — Run 3 Tasks

```bash
source .venv/bin/activate
python scripts/run_gdpval_eval.py --n 3 --verbose
```

This will:
1. Load 3 random GDPval tasks (seeded for reproducibility)
2. Send each task prompt to Claude Code via the Agent SDK
3. Collect the full response (text + tool calls)
4. Score against the rubric using keyword matching
5. Save results to `results/gdpval_<timestamp>.json`

Expected output:
```
Loading GDPval dataset...
  Loaded 220 GDPval tasks
  Sampled 3 tasks

Tasks to evaluate:
  1. [Property Managers] 0419f1c3-d66... — You are a Property Manager...
  2. [Compliance Officers] dfb4e0cd-a0b... — You are a grants management...
  3. [Admin Services Mgrs] a328feea-47d... — You are the Administrative...

Starting evaluation (max_turns=10, concurrency=1)...
============================================================
[1/3] Done.
[2/3] Done.
[3/3] Done.

============================================================
EVALUATION COMPLETE
============================================================
  Tasks run:      3
  Completed:      3
  Errors:         0
  Avg score:      45%
  Total time:     127.3s

Results saved to: results/gdpval_20260310_143022.json
```

## Run a Zipper Slice

Each zipper slice contains 22 non-overlapping tasks used for one iteration of the RSI loop:

```bash
# Run slice S1 (22 tasks, used for iteration 1)
python scripts/run_gdpval_eval.py --slice S1

# Run slice S2 (22 tasks, used for iteration 2)
python scripts/run_gdpval_eval.py --slice S2
```

Available slices: `S1`, `S2`, `S3`, `S4`, `S5`, `S6`, `vault` (88 tasks, use once).

## Full Evaluation

```bash
# Run all 220 GDPval tasks (takes several hours)
python scripts/run_gdpval_eval.py --all --output results/baseline_full.json
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n N` | 3 | Number of tasks to sample |
| `--all` | — | Run all 220 tasks |
| `--slice NAME` | — | Run a specific zipper slice (S1-S6, vault) |
| `--max-turns N` | 10 | Max Claude Code agent turns per task |
| `--concurrency N` | 1 | Parallel task execution (use with caution) |
| `--output PATH` | auto | Output JSON path |
| `--seed N` | 42 | Random seed for sampling |
| `--verbose` | — | Enable debug logging |

## Architecture

```
scripts/run_gdpval_eval.py     # CLI entry point
src/eval/
  runner.py                    # GDPvalRunner — sends tasks via Agent SDK
  evaluators/
    base.py                    # EvalResult + BaseEvaluator ABC
    gdpval.py                  # GDPvalEvaluator — rubric keyword scoring
```

### How it works

```
┌─────────────┐    prompt     ┌──────────────────┐
│  GDPval     │ ────────────> │  Claude Code      │
│  Sample     │               │  (via Agent SDK)  │
└─────────────┘               └────────┬─────────┘
                                       │ response
                                       v
                              ┌──────────────────┐
                              │  GDPvalEvaluator  │
                              │  (rubric scoring) │
                              └────────┬─────────┘
                                       │ EvalResult
                                       v
                              ┌──────────────────┐
                              │  results/*.json   │
                              └──────────────────┘
```

1. **GDPvalRunner** creates a `ClaudeAgentOptions` with:
   - System prompt tuned for professional task completion
   - Tools: Read, Write, Edit, Bash, Glob, Grep
   - `permission_mode='acceptEdits'` (auto-approve file operations)
   - `max_turns=10` (configurable)

2. Each task prompt is sent via `query()` from the SDK

3. The runner collects all `AssistantMessage` text blocks and `ToolUseBlock` calls

4. **GDPvalEvaluator** parses the `[+N]` rubric format and checks keyword presence in the response

### Key SDK usage (from `src/eval/runner.py`):

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock, ToolUseBlock

options = ClaudeAgentOptions(
    system_prompt="You are a professional assistant...",
    max_turns=10,
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    permission_mode="acceptEdits",
)

async for message in query(prompt=sample.prompt, options=options):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                response_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({"tool": block.name, "input": block.input})
```

## Output Format

Results are saved as JSON:

```json
{
  "summary": {
    "avg_score": 0.45,
    "num_tasks": 3,
    "num_completed": 3,
    "num_errors": 0,
    "total_duration_s": 127.3
  },
  "traces": [
    {
      "task_id": "0419f1c3-d66...",
      "benchmark": "gdpval",
      "duration_s": 42.1,
      "error": null,
      "response_length": 2847,
      "num_tool_calls": 5,
      "tool_calls": [...],
      "eval": {
        "score": 8.0,
        "max_score": 20.0,
        "normalized_score": 0.4,
        "rubric_breakdown": {...},
        "metadata": {"num_criteria": 10, "criteria_met": 4}
      }
    }
  ]
}
```

## Scoring Notes

The current evaluator uses **keyword-based rubric matching** — a lightweight heuristic that checks whether the agent's response text contains evidence of meeting each `[+N]` rubric criterion. This is a baseline.

For accurate GDPval scoring, the design plan calls for a **frozen judge model** (Gemini 3 Pro Preview) that:
- Reads the full rubric
- Evaluates the agent's deliverable files (not just text)
- Assigns per-criterion scores

The keyword evaluator will systematically undercount because:
- Many criteria require file deliverables (Excel, Word) that the text response may not fully capture
- Structural requirements (e.g., "workbook contains worksheet named X") need file inspection
- Quantitative checks (e.g., "sample size is 22") need arithmetic verification

## Next Steps

1. **Run baseline**: `python scripts/run_gdpval_eval.py --n 10` to establish initial scores
2. **Add judge model**: Implement Gemini-based evaluator for accurate rubric scoring
3. **Add other benchmarks**: IFEval (programmatic), SimpleQA (exact match), etc.
4. **Run full baseline**: All 220 tasks with judge scoring before starting RSI iterations
