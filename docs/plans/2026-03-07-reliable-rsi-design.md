# Reliable Recursive Self-Improvement: Implementation Plan

> **Project:** Reliable RSI for COLM 2026
> **Date:** 2026-03-07
> **Status:** Phase 0 complete, Phase 1 in progress — baseline S1 run for Claude (32.3%) and Gemini (3.6%) with GPT-5.4 judge
> **Target Venue:** COLM 2026 (San Francisco, Oct 6-9, 2026)

---

## 1. Vision

Build a **Reliable RSI system** that self-improves a coding agent harness across three axes simultaneously:
1. **Economic value** (GDPval — real-world professional tasks)
2. **Truthfulness** (hallucination benchmarks — stop fabricating, say "I don't know")
3. **Instruction following** (stop being verbose/descriptive, follow constraints precisely)

The system should demonstrably **beat Claude Code Opus** on GDPval after 8 batches of self-improvement, while also improving on safety/hallucination metrics. Uses SkyDiscover's evolutionary harness as the search backbone.

---

## 2. Key Resources

| Resource | What it provides |
|----------|-----------------|
| [SkyDiscover](https://github.com/skydiscover-ai/skydiscover) | Modular evaluator + evolutionary search (AdaEvolve/EvoX), ~200 optimization tasks |
| [GDPval](https://huggingface.co/datasets/openai/gdpval) | 220 real-world professional tasks, 44 occupations, 9 GDP sectors |
| [ReCodeAgent paper](./recode.pdf) | Zipper schedule, 7-step meta-improvement, frozen judge, acceptance rules |
| [Nicolas Schneider blog](https://nicolas2912.github.io/2025/09/24/recursive-self-improvement.html) | Validation bottleneck insight: g(C)/D(C) ratio |
| Gemini / Codex | Worker models for task execution (cheaper, available) |

---

## 3. Architecture Overview

```
+------------------------------------------------------------------+
|                    RELIABLE RSI HARNESS                            |
|                                                                    |
|  +------------------+   +------------------+   +----------------+  |
|  |   BENCHMARK      |   |    ARCHIVE       |   |   IMPROVE      |  |
|  |   SUITE          |   |    (per-iter     |   |   (7-step      |  |
|  |                  |   |     snapshots)   |   |    meta-loop)  |  |
|  |  - GDPval (220)  |   |  - code snapshot |   |  1. ID fails   |  |
|  |  - Hallucination |   |  - scores/traces |   |  2. ID wins    |  |
|  |  - IFEval        |   |  - CI heuristic  |   |  3. Design fix |  |
|  |  - Safety        |   |                  |   |  4. Implement  |  |
|  +--------+---------+   +--------+---------+   |  5. Test       |  |
|           |                      |              |  6. Update desc|  |
|           v                      v              |  7. Changelog  |  |
|  +------------------+   +------------------+   +--------+-------+  |
|  |  FROZEN JUDGE    |   |    SELECT        |            |          |
|  |  (Gemini 3 Pro)  |   |  (CI heuristic,  |            |          |
|  |  Never modified  |   |   rollback rule) |<-----------+          |
|  +------------------+   +------------------+                       |
|                                                                    |
|  WORKER MODELS: Gemini / Codex (task execution)                    |
|  META-IMPROVER: Claude Opus 4.6 (self mode)                       |
+------------------------------------------------------------------+
```

---

## 4. Data Splits (GDPval)

220 tasks split into 10 deterministic slices of 22 tasks each via round-robin over sorted occupations:

```
220 GDPval Tasks (44 occupations x 5 tasks)
|
+-- 176 Dev Pool (8 slices, 22 tasks each)
|   |
|   +-- S1 (22 tasks) -- Iter 1: Baseline run
|   +-- S2 (22 tasks) -- Iter 2: Evolve harness from S1 traces
|   +-- S3 (22 tasks) -- Iter 3: Evolve from S2 traces
|   +-- S4 (22 tasks) -- Iter 4: Evolve from S3 traces
|   +-- S5 (22 tasks) -- Iter 5: Evolve from S4 traces
|   +-- S6 (22 tasks) -- Iter 6: Evolve from S5 traces
|   +-- S7 (22 tasks) -- Iter 7: Evolve from S6 traces
|   +-- S8 (22 tasks) -- Iter 8: Evolve from S7 traces
|
+-- 44 Eval Pool (2 slices, 22 tasks each) -- Used ONCE at the end
    |
    +-- E1 (22 tasks) -- Final eval slice 1
    +-- E2 (22 tasks) -- Final eval slice 2
```

### RSI Loop

```
S1 → score → evolve harness → S2 → score → evolve → ... → S8 → score → E1, E2 (final)
```

1. **Baseline (S1):** Run all agents (Claude, Gemini, Codex, custom) on S1. Record scores.
2. **Evolve (S1→S2):** Meta-improver reads S1 traces, identifies failures, edits the custom harness.
3. **Evaluate (S2):** Run evolved harness on S2 (unseen tasks). Record scores.
4. **Repeat** through S3–S8, evolving after each slice.
5. **Final eval (E1, E2):** Run the final evolved harness on both eval slices. This is the paper number.

**Rules:**
- At iteration t, meta-improver reads traces from S(t-1), evaluates on fresh S(t). No task reuse.
- Baseline agents (Claude, Gemini, Codex) run unchanged on all slices for comparison.
- Score trajectory across S1→S8 shows whether the harness improves over time.
- E1+E2 confirm that improvements generalize to unseen tasks.

---

## 5. Benchmark Suite

### 5.1 GDPval (Primary — Economic Value)
- 220 real-world professional tasks
- Rubric-based scoring (50+ criteria per task)
- Split as above

### 5.2 Hallucination / Truthfulness Benchmark (TBD — needs selection)

**Candidates to evaluate:**

| Benchmark | What it tests | Size | Notes |
|-----------|--------------|------|-------|
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Factual accuracy, resisting common misconceptions | 817 Qs | Well-established, MC + generation |
| [HaluEval](https://github.com/RUCAIBox/HaluEval) | Hallucination detection in QA/summarization/dialogue | 35K samples | Can filter for harness-relevant subsets |
| [FELM](https://github.com/hkust-nlp/FELM) | Faithfulness of LLM-generated text | 847 samples | Fine-grained annotation |
| [FactScore](https://github.com/shmsw25/FActScore) | Factual precision of long-form generation | Flexible | Per-claim verification |
| [SimpleQA](https://openai.com/index/simpleqa/) | Short factual QA where "I don't know" is correct | 4,326 Qs | OpenAI benchmark, tests calibration |

**Recommendation:** SimpleQA + TruthfulQA — they directly test the "cheating" behavior (fabricating answers vs. admitting uncertainty).

### 5.3 Instruction Following Benchmark

| Benchmark | What it tests | Size | Notes |
|-----------|--------------|------|-------|
| [IFEval](https://arxiv.org/abs/2311.07911) | Verifiable instruction constraints (format, length, keywords) | 541 prompts | Google, programmatic verification |
| [FollowBench](https://github.com/YJiangcm/FollowBench) | Multi-level constraint following | 820 prompts | Fine-grained difficulty levels |
| [InFoBench](https://github.com/qinyiwei/InFoBench) | Decomposed instruction following | 500 prompts | Decomposition-based eval |

**Recommendation:** IFEval — programmatic verification (no judge needed), directly tests "descriptive" vs "precise" behavior.

### 5.4 Safety Benchmark

| Benchmark | What it tests | Size | Notes |
|-----------|--------------|------|-------|
| [SafeRLHF / BeaverTails](https://github.com/PKU-Alignment/safe-rlhf) | Safety vs. helpfulness tradeoff | 330K annotations | PKU, multilabel |
| [WildGuard](https://github.com/allenai/wildguard) | Prompt harmfulness + response safety | 92K items | Allen AI |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Red-teaming robustness | 510 behaviors | CAIS, standardized |
| [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) | Over-refusal detection | 1,000+ prompts | Tests if model refuses safe requests |

**Recommendation:** HarmBench + OR-Bench — tests both under-refusal (unsafe) and over-refusal (unhelpful).

---

## 6. Combined Harness Design

### 6.1 Scoring Function

```python
# Weighted composite score
def composite_score(gdpval_score, hallucination_score, ifeval_score, safety_score):
    return (
        0.50 * gdpval_score +        # Economic value (primary)
        0.20 * hallucination_score +  # Truthfulness
        0.15 * ifeval_score +         # Instruction following
        0.15 * safety_score           # Safety
    )
```

Weights are tunable. GDPval dominates because it's the paper's primary contribution.

### 6.2 Harness Structure (as built)

```
src/eval/                        # ← IMPLEMENTED
  agents/
    base.py                      # BaseAgent ABC + AgentResult dataclass
    claude_code.py               # Claude Code via claude-agent-sdk
    gemini_cli.py                # Gemini CLI via subprocess
    codex.py                     # OpenAI Codex CLI via subprocess
  evaluators/
    base.py                      # BaseEvaluator ABC + EvalResult dataclass
    gdpval.py                    # Keyword heuristic (fast, free)
    gdpval_judge.py              # Gemini Pro frozen judge (accurate)
  runner.py                      # GDPvalRunner — orchestrates agent + evaluator
  visualize.py                   # Paper-quality plots
scripts/
  run_gdpval_eval.py             # CLI entry point (--agent, --judge, --slice, etc.)
results/
  <agent>_<timestamp>/
    traces.json                  # Raw agent output
    eval.json                    # Scores only
  workspace_<agent>/<task_id>/   # Per-task isolated workspaces with deliverables

src/eval/ (TODO)                 # ← NOT YET IMPLEMENTED
  evaluators/
    hallucination_evaluator.py   # TruthfulQA + SimpleQA scoring
    ifeval_evaluator.py          # IFEval programmatic verification
    safety_evaluator.py          # HarmBench + OR-Bench scoring
  search/
    evolutionary.py              # AdaEvolve/EvoX from SkyDiscover
    meta_improver.py             # 7-step meta-improvement (from ReCodeAgent)
    selector.py                  # CI-based selection + rollback
  archive/
    iteration_0/                 # Baseline snapshot
    iteration_1/                 # After first improvement
```

---

## 7. Implementation Phases

### Phase 0: Setup & Download (Week 1) — COMPLETE ✅
- [x] Clone SkyDiscover repo, study harness structure
- [x] Download GDPval from HuggingFace
- [x] Download hallucination benchmarks (TruthfulQA, SimpleQA)
- [x] Download IFEval
- [x] Download safety benchmarks (HarmBench, OR-Bench)
- [x] Set up project structure as above
- [x] Implement data_split.py (zipper schedule from ReCodeAgent Algorithm 1)

### Phase 1: Sanity Check — Baseline (Week 2) — IN PROGRESS 🔧
- [x] Implement GDPval keyword evaluator (baseline heuristic)
- [x] Implement GDPval frozen Gemini judge evaluator (`src/eval/evaluators/gdpval_judge.py`)
- [x] Implement pluggable agent backends: Claude Code, Gemini CLI, Codex (`src/eval/agents/`)
- [x] Implement `GDPvalRunner` — sends tasks to agents via SDK, collects traces, scores responses
- [x] Implement per-task isolated workspaces (parallel-safe)
- [x] Implement file content extraction from agent deliverables (.docx, .xlsx, .pdf, .pptx, .xls)
- [x] Implement split output: `traces.json` (raw) + `eval.json` (scores)
- [x] Implement paper-quality visualization (`src/eval/visualize.py`) — trajectory, sector, failure taxonomy, model comparison, dashboard
- [x] Run Claude Code n=1 sanity check on GDPval — **94% on keyword heuristic** (51/52 criteria)
- [x] Implement `run_baseline.py` — orchestrates all agents across all slices with checkpointing and resume
- [x] Run Claude Code baseline on S1 — **32.3%** (GPT-5.4 judge, 22/22 completed)
- [x] Run Gemini CLI baseline on S1 — **3.6%** (GPT-5.4 judge, 22/22 completed)
- [ ] Run Codex baseline on S1 (not yet started)
- [ ] Run Claude Code Opus baseline on remaining dev slices (S2–S8)
- [ ] Run Gemini CLI baseline on remaining dev slices (S2–S8)
- [ ] Run Codex baseline on all 8 dev slices
- [ ] Run baselines on hallucination/IFEval/safety benchmarks
- [ ] Record baseline scores per slice, per benchmark, per agent
- [ ] Verify zipper schedule produces disjoint slices
- [ ] **Milestone:** Baseline numbers for all benchmarks and all agents

### Phase 2: Custom Harness (Week 3-4)
- [x] Implement `task_runner.py` — runs agent, collects traces (done as `runner.py`)
- [x] Batch runner with concurrency support (done in `GDPvalRunner.run_batch()`)
- [ ] Implement composite scoring function (aggregate across 4 benchmark axes)
- [ ] Implement evaluators for other benchmarks (IFEval programmatic, SimpleQA exact-match, HarmBench, OR-Bench)
- [ ] Implement archive system (code snapshots + results per iteration)
- [ ] Implement pre-flight gate (syntax + unit tests)
- [ ] Implement acceptance rule (>=1.5pp delta, >=12/22 wins, no critical regressions)
- [ ] **Milestone:** Harness can run a full iteration (benchmark -> archive -> select)

### Phase 3: Meta-Improvement Loop (Week 5-6)
- [ ] Implement 7-step meta-improvement process
- [ ] Implement self mode (agent reads own traces, edits own code)
- [ ] Integrate SkyDiscover's evolutionary search (AdaEvolve) as alternative search strategy
- [ ] Implement rollback safety
- [ ] Run 8 iterations on dev slices (S1→S8), evolving after each
- [ ] Plot score trajectory across S1→S8
- [ ] **Milestone:** Full RSI loop runs end-to-end

### Phase 4: SkyDiscover Comparison (Week 7)
- [ ] Run SkyDiscover's own harness on its native tasks as sanity check
- [ ] Adapt SkyDiscover to run on our custom harness
- [ ] Compare: SkyDiscover search vs. ReCodeAgent 7-step meta-improvement
- [ ] Train 6 batches with SkyDiscover's AdaEvolve on our benchmark suite
- [ ] **Milestone:** SkyDiscover RSI results on combined harness

### Phase 5: Beat Claude Code (Week 8)
- [ ] Compare final improved agent vs. Claude Code Opus baseline on GDPval
- [ ] Compare on hallucination benchmarks
- [ ] Compare on IFEval
- [ ] Compare on safety benchmarks
- [ ] Run 44-task held-out eval (E1 + E2) evaluation (ONE TIME ONLY)
- [ ] **Milestone:** Improved agent > Claude Code on GDPval + safer + less hallucination

### Phase 6: Paper & Analysis (Week 9-10)
- [ ] Ablation: which benchmarks contribute most to improvement?
- [ ] Ablation: SkyDiscover search vs. 7-step meta-improvement
- [ ] Per-sector analysis on GDPval
- [ ] Failure taxonomy (format-specific, hallucination, safety)
- [ ] Write COLM 2026 paper
- [ ] **Milestone:** Paper draft ready

---

## 8. Key Design Decisions (Need Input)

### 8.1 Hallucination/Safety Benchmark Selection

**Question:** Which direction for the hallucination/safety component?

| Option | Benchmarks | Pros | Cons |
|--------|-----------|------|------|
| **(A)** TruthfulQA only | TruthfulQA (817 Qs) | Simple, well-established | Narrow scope |
| **(B)** Instruction-following focus | IFEval (541 prompts) | Directly tests "descriptive" problem | Doesn't test hallucination |
| **(C)** Safety focus | HarmBench + OR-Bench | Tests both under- and over-refusal | Less relevant to RSI improvement |
| **(D)** Full combination | SimpleQA + IFEval + HarmBench + OR-Bench | Comprehensive, strongest paper story | More engineering, slower iterations |

**Recommendation:** **(D)** — the full combination makes the strongest COLM paper. The thesis becomes "RSI improves agents across ALL axes, not just task performance." The extra benchmarks add maybe 2 weeks of work but make the contribution much broader.

### 8.2 Worker Models

- **Task execution:** Gemini 2.5 Pro / Codex (cheaper per iteration)
- **Meta-improvement:** Claude Opus 4.6 (self mode, strongest reasoning)
- **Frozen judge:** Gemini 3 Pro Preview (cross-model-family grading)

### 8.3 Search Strategy

| Strategy | Source | How it works |
|----------|--------|-------------|
| 7-step meta-improvement | ReCodeAgent | Analyze failures -> design fix -> implement -> test |
| AdaEvolve | SkyDiscover | Multi-island evolutionary search with UCB + migration |
| EvoX | SkyDiscover | Self-evolving paradigm, co-adapts solution + experience |

**Plan:** Start with 7-step meta-improvement (proven in ReCodeAgent), then compare against AdaEvolve as ablation.

---

## 9. Acceptance Criteria for the Paper

The paper succeeds if we demonstrate:

1. **GDPval improvement:** Improved agent scores higher than Claude Code Opus baseline on GDPval (target: baseline ~94.7% -> improved 96%+)
2. **Hallucination reduction:** Measurable improvement on TruthfulQA/SimpleQA after RSI
3. **Instruction following:** Measurable improvement on IFEval after RSI
4. **Safety maintenance:** Safety scores don't degrade (and ideally improve)
5. **Reliability:** Monotonic progress across iterations (no catastrophic regressions)
6. **Generalization:** Held-out 88-task test vault confirms dev improvements transfer

---

## 10. Repos to Download

```bash
# SkyDiscover — evolutionary harness
git clone https://github.com/skydiscover-ai/skydiscover.git

# Hallucination benchmarks
git clone https://github.com/sylinrl/TruthfulQA.git
# SimpleQA — download from OpenAI (check HuggingFace)

# Instruction following
# IFEval — available via google-research or HuggingFace

# Safety
git clone https://github.com/centerforaisafety/HarmBench.git
# OR-Bench — available on HuggingFace

# GDPval
# huggingface-cli download openai/gdpval
```

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GDPval baseline already ~95%, ceiling effect | High | High | Focus improvement on format-specific + modality gap failures |
| Hallucination benchmarks not sensitive to scaffolding changes | Medium | Medium | Test early in Phase 1; drop if no signal |
| SkyDiscover harness too different to adapt | Low | Medium | We can copy the evaluator pattern without the full framework |
| 6 iterations insufficient for meaningful improvement | Medium | High | Each iteration targets a different failure mode; accept smaller gains |
| Cost of running 220 tasks x 6 iterations x 2 models | High | Medium | Use cheaper Gemini for task execution; parallelize |

---

*This plan is a living document. Update as decisions are made and phases complete.*
