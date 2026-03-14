# Recursive Self-Improvement: Evolving the Agent Code

An AI agent that rewrites its own source code to get better at tasks.

---

## Quick Start

```bash
source .venv/bin/activate

# Evolve Gemini agent across S1→S3 (~30 min)
python -m src.evolve.run_evolve --seed gemini --slices S1 S2 S3

# Evolve OpenAI agent with top-k search
python -m src.evolve.run_evolve --seed openai --search topk --iterations 20

# Full S1→S8 with full eval after each slice
python -m src.evolve.run_evolve --slices S1 S2 S3 S4 S5 S6 S7 S8 --full-eval
```

---

## How the Evolution Actually Works

The slices aren't training data. There's no gradient descent or weight updates.
The LLM's weights never change. What evolves is the **source code** of the
agent — the Python file that defines how it thinks, acts, and recovers from errors.

### The Loop

```
       SLICE S1 (22 tasks)              SLICE S2 (22 tasks)              SLICE S3
       ──────────────────              ──────────────────              ──────────

       seed_agent.py (v0)
            │
            ├─ eval on 3 random S1 tasks → 85%  (baseline)
            │
            │  SkyDiscover iteration 1:
            │    LLM reads v0, writes v1 (mutated code)
            │    eval v1 on 3 random S1 tasks → 78%  (worse, keep v0)
            │
            │  SkyDiscover iteration 2:
            │    LLM reads v0, writes v2
            │    eval v2 on 3 random S1 tasks → 91%  (better!)
            │
            │  SkyDiscover iteration 3:
            │    LLM reads v2, writes v3
            │    eval v3 on 3 random S1 tasks → 88%  (worse than v2)
            │
            │  ...5 iterations total...
            │
            │  best = v2 (91%)
            │
            └─────────────────────────────┐
                                          │
                                    v2 carries over
                                          │
                                          ├─ eval on 3 random S2 tasks → 80%
                                          │
                                          │  SkyDiscover iterations 1-5
                                          │  (mutate v2, score on S2 tasks)
                                          │
                                          │  best = v5 (93%)
                                          │
                                          └──────────────────────┐
                                                                 │
                                                           v5 carries over
                                                                 │
                                                                 ├─ eval on S3...
                                                                 │
                                                                ...
```

### What Happens at Each Step

**"eval on 3 random S1 tasks"** means:
1. Dynamically load the evolved Python file as a module
2. Instantiate the agent class
3. Give it 3 real task prompts (e.g. "Create a budget spreadsheet...")
4. The agent runs — calls Gemini/OpenAI/Claude, uses its tools, creates files
5. LLM judge reads the task + rubric + agent output, grades it
6. Average score across 3 tasks = fitness

**"LLM reads v0, writes v1"** means:
- SkyDiscover sends the full agent source code to the evolution LLM
- The LLM sees the code + context from other high-scoring variants in the population
- It outputs a modified version (diff or full rewrite)
- That's the mutation — no crossover, no bit-flipping, just "LLM, make this better"

**"91% better, keep it"** means:
- The variant enters SkyDiscover's population
- If it's the highest score, it becomes the new best
- Lower-scoring variants still survive (diversity matters for exploration)
- At the end of the slice, the single best is extracted and carried forward

### Why Slices Matter

S1 tasks and S2 tasks are **completely different**. If the evolution overfits the
code to S1's specific tasks, it'll score poorly on S2. The slice schedule forces
generalization:

```
S1: "Create a marketing budget"    → agent learns to make spreadsheets well
S2: "Draft a legal contract"       → does that improvement transfer?
S3: "Analyze survey data"          → does the agent generalize?
```

A variant that hardcodes "always use openpyxl" might ace S1 but fail S2. A
variant that genuinely improves how the agent reads tasks, plans deliverables,
and verifies work will score well across all slices.

### What's NOT Happening

- **No gradient updates** — the agent's LLM weights never change
- **No fine-tuning** — the model API is called as-is
- **No memorization** — each slice has unseen tasks
- **No training set** — the code itself is what evolves, not model parameters

The "training signal" is: did this code change make the agent complete tasks
better? If yes, keep it. If no, discard. Pure selection pressure on source code.

---

## File Structure

```
src/evolve/
├── run_evolve.py      # Orchestrator — evolve loop across slices
├── cli.py             # Argument parsing + logging setup
├── config.py          # SkyDiscover config builder + system message
├── slices.py          # Zipper slice loading + full-slice evaluation
├── evaluator.py       # Scores evolved code on tasks
├── seeds/             # Starting agent code per provider
│   ├── gemini.py      #   Google Gemini (google-genai SDK)
│   ├── openai.py      #   OpenAI GPT (openai SDK)
│   └── anthropic.py   #   Anthropic Claude (anthropic SDK)
└── README.md
```

| Module | Responsibility |
|--------|----------------|
| `run_evolve.py` | Loop over slices, call evolve, save artifacts |
| `cli.py` | Parse `--flags`, configure console + file logging |
| `config.py` | Build SkyDiscover `Config` for any search strategy |
| `slices.py` | Load zipper slices, run full evaluations |
| `evaluator.py` | Compile, load, instantiate, score evolved agents |

---

## Seed Agents

Three provider variants sharing the same 8-tool standard. Each is fully
self-contained — every definition inlined so the evolving LLM can see and
modify everything.

| Seed | SDK | Default Model | API Key |
|------|-----|---------------|---------|
| `gemini` | `google-genai` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| `openai` | `openai` | `gpt-5-mini` | `OPENAI_API_KEY` |
| `anthropic` | `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |

### 8 Standard Tools

| Tool | Purpose |
|------|---------|
| `bash` | Run shell commands, scripts, pip install |
| `read_file` | Read any file (.pdf, .docx, .xlsx, .pptx, text) |
| `write_file` | Create or overwrite text files |
| `edit_file` | Surgical find-and-replace in existing files |
| `list_dir` | List directory contents with sizes |
| `grep` | Search file contents by regex |
| `glob` | Find files by name pattern |
| `done` | Signal task completion with summary |

---

## The Evaluator

How each code variant gets scored:

```
Evolved code (temp file)
      │
      ▼
compile() ─── syntax error? → score = 0, eliminated
      │
      ▼
importlib load ─── import error? → score = 0, eliminated
      │
      ▼
Find agent class (duck-typing: any non-abstract class with run + name)
      │                        ─── not found? → score = 0, eliminated
      ▼
Instantiate agent, run on task sample
      │                        ─── crashes? → score = 0, eliminated
      ▼
LLM judge grades output → combined_score
```

Broken code dies. Working code that scores low gets pushed down in the
population. Only code that actually completes tasks well survives.

---

## Search Strategies

`--search` selects the algorithm. See `docs/search_strategies.md` for details.

| Algorithm | Flag | Description |
|-----------|------|-------------|
| AdaEvolve | `adaevolve` | Multi-island adaptive. UCB + migration. Default. |
| Top-K | `topk` | Keep best K, mutate from top. Simple, fast. |
| EvoX | `evox` | Co-evolves code AND search strategy. |
| Beam Search | `beam_search` | Fixed-width frontier, depth-first. |
| Best-of-N | `best_of_n` | Generate N variants, keep best. Baseline. |

---

## CLI Reference

```
python -m src.evolve.run_evolve [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `gemini` | Starting agent: `gemini`, `openai`, `anthropic` |
| `--slices` | `S1 S2 S3` | Zipper slices to evolve across |
| `--iterations` | `5` | SkyDiscover iterations per slice |
| `--sample-size` | `3` | Tasks per evaluation |
| `--model` | `gemini/gemini-2.5-flash` | LLM for code mutation |
| `--search` | `adaevolve` | Search strategy |
| `--full-eval` | off | Full 22-task eval after each slice |
| `--judge-model` | auto | LLM judge model override |
| `--output-dir` | auto | Results directory |
| `--verbose` | off | Debug logging |

---

## Output

```
results/evolve_<timestamp>/
├── evolve.log                   # Full run log
├── trajectory.json              # Scores per slice + code size
├── harness_initial.py           # Starting agent code
├── harness_final.py             # Final evolved agent code
├── harness_S1_input.py          # Code entering each slice
├── harness_S1_evolved.py        # Best code per slice
└── skydiscover_S1/              # SkyDiscover artifacts
    ├── best/best_program.py
    ├── checkpoints/
    └── logs/
```

### Trajectory format

```json
{
  "slice": "S2",
  "initial_score": 0.915,
  "evolved_score": 0.959,
  "code_lines": 851,
  "duration_s": 552
}
```

---

## Trial Results

Code evolution, S1→S3, AdaEvolve, 5 iters, 3 tasks/eval:

```
Slice   Initial   Evolved   Lines
S1       97.1%     97.1%     864
S2       91.5%     95.9%     851
S3       92.2%     93.2%     853
```
