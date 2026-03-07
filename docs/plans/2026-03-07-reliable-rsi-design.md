# Reliable Recursive Self-Improvement: Implementation Plan

> **Project:** Reliable RSI for COLM 2026
> **Date:** 2026-03-07
> **Status:** DRAFT — Awaiting design approval
> **Target Venue:** COLM 2026 (San Francisco, Oct 6-9, 2026)

---

## 1. Vision

Build a **Reliable RSI system** that self-improves a coding agent harness across three axes simultaneously:
1. **Economic value** (GDPval — real-world professional tasks)
2. **Truthfulness** (hallucination benchmarks — stop fabricating, say "I don't know")
3. **Instruction following** (stop being verbose/descriptive, follow constraints precisely)

The system should demonstrably **beat Claude Code Opus** on GDPval after 6 batches of self-improvement, while also improving on safety/hallucination metrics. Uses SkyDiscover's evolutionary harness as the search backbone.

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

Directly from ReCodeAgent's zipper schedule:

```
220 GDPval Tasks (44 occupations x 5 tasks)
|
+-- 132 Dev Pool (3 per occupation)
|   |
|   +-- S1: Odd-a  (22 tasks) -- Iter 1
|   +-- S2: Even-a (22 tasks) -- Iter 2
|   +-- S3: Odd-b  (22 tasks) -- Iter 3
|   +-- S4: Even-b (22 tasks) -- Iter 4
|   +-- S5: Odd-c  (22 tasks) -- Iter 5
|   +-- S6: Even-c (22 tasks) -- Iter 6
|
+-- 88 Test Vault (2 per occupation) -- Used ONCE at the end
```

**Rule:** At iteration t, meta-improver reads traces from S(t-1), evaluates on fresh S(t). No task reuse.

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

### 6.2 Harness Structure (adapted from SkyDiscover)

```
harness/
  evaluators/
    gdpval_evaluator.py       # GDPval rubric scoring (frozen judge)
    hallucination_evaluator.py # TruthfulQA + SimpleQA scoring
    ifeval_evaluator.py        # IFEval programmatic verification
    safety_evaluator.py        # HarmBench + OR-Bench scoring
  runners/
    task_runner.py             # Runs agent on tasks, collects traces
    batch_runner.py            # Runs a full slice (22 tasks)
  search/
    evolutionary.py            # AdaEvolve/EvoX from SkyDiscover
    meta_improver.py           # 7-step meta-improvement (from ReCodeAgent)
    selector.py                # CI-based selection + rollback
  data/
    gdpval/                    # Downloaded GDPval dataset
    splits/
      data_split.json          # Zipper schedule assignment
    benchmarks/
      truthfulqa/
      simpleqa/
      ifeval/
      harmbench/
      or_bench/
  agents/
    agent.py                   # The ONLY file that changes (task-solving agent)
    judge_agent.py             # FROZEN — Gemini 3 Pro Preview
    meta_agent.py              # FROZEN — Claude Opus 4.6 code architect
  archive/
    iteration_0/               # Baseline snapshot
    iteration_1/               # After first improvement
    ...
  config/
    harness_config.yaml        # Model selection, weights, thresholds
  tests/
    test_evaluators.py
    test_splits.py
    test_runner.py
```

---

## 7. Implementation Phases

### Phase 0: Setup & Download (Week 1)
- [ ] Clone SkyDiscover repo, study harness structure
- [ ] Download GDPval from HuggingFace
- [ ] Download hallucination benchmarks (TruthfulQA, SimpleQA)
- [ ] Download IFEval
- [ ] Download safety benchmarks (HarmBench, OR-Bench)
- [ ] Set up project structure as above
- [ ] Implement data_split.py (zipper schedule from ReCodeAgent Algorithm 1)

### Phase 1: Sanity Check — Baseline (Week 2)
- [ ] Implement GDPval evaluator with frozen Gemini judge
- [ ] Run Claude Code Opus baseline on all 220 GDPval tasks
- [ ] Run Claude Code Opus baseline on hallucination/IFEval/safety benchmarks
- [ ] Record baseline scores per slice, per benchmark
- [ ] Verify zipper schedule produces disjoint slices
- [ ] **Milestone:** Baseline numbers for all benchmarks

### Phase 2: Custom Harness (Week 3-4)
- [ ] Adapt SkyDiscover's evaluator pattern for our benchmark suite
- [ ] Implement `task_runner.py` — runs agent, collects traces
- [ ] Implement `batch_runner.py` — runs full slice
- [ ] Implement composite scoring function
- [ ] Implement archive system (code snapshots + results per iteration)
- [ ] Implement pre-flight gate (syntax + unit tests)
- [ ] Implement acceptance rule (>=1.5pp delta, >=12/22 wins, no critical regressions)
- [ ] **Milestone:** Harness can run a full iteration (benchmark -> archive -> select)

### Phase 3: Meta-Improvement Loop (Week 5-6)
- [ ] Implement 7-step meta-improvement process
- [ ] Implement self mode (agent reads own traces, edits own code)
- [ ] Integrate SkyDiscover's evolutionary search (AdaEvolve) as alternative search strategy
- [ ] Implement rollback safety
- [ ] Run 6 iterations on dev slices
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
- [ ] Run 88-task held-out test vault evaluation (ONE TIME ONLY)
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
