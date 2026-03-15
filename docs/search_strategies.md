# Search Strategies

How each SkyDiscover algorithm selects parents, mutates code, and decides what survives.

---

## AdaEvolve (default)

Multi-island adaptive evolution. The smart default.

```
                 ┌─────────────────┐         ┌─────────────────┐
                 │   Island 0      │ migrate  │   Island 1      │
                 │   (exploiting)  │ ──────>  │   (exploring)   │
                 │                 │ <──────  │                 │
                 │  [97] [95] [93] │  top     │  [90] [72] [88] │
                 │  [91] [88] [85] │ agents   │  [65] [78] [55] │
                 └────────┬────────┘         └────────┬────────┘
                          │                           │
                          └─────────┬─────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   UCB Bandit       │
                          │   picks island     │
                          │   with recent      │
                          │   breakthroughs    │
                          └─────────┬─────────┘
                                    │
          ┌──────────┬──────────┬───┴────┬──────────┬──────────┐
          ▼          ▼          ▼        ▼          ▼          ▼
     ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
     │ Pick   │→│ Sample │→│  LLM   │→│Compile │→│  Run   │→│Score → │
     │ island │ │ parent │ │mutates │ │& load  │ │ tasks  │ │  pop   │
     └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

**How selection works:**

1. **UCB bandit** picks which island gets the next evaluation
   - `UCB = avg_reward / visits + C × √(ln(total) / visits)`
   - Islands with recent breakthroughs → more evaluations
   - Islands that stalled → less attention, but never zero (exploration term)

2. **Adaptive intensity** per island based on accumulated improvement signal (G):
   - `intensity = I_min + (I_max - I_min) / (1 + √G)`
   - High G (productive island) → low intensity → **exploit** (mutate from top scorers)
   - Low G (stagnating island) → high intensity → **explore** (sample random parents, try wild changes)

3. **Migration** every N iterations: top agents from island 0 copy to island 1 and vice versa
   - Ring topology: 0 → 1 → 2 → 0
   - Prevents islands from diverging too far
   - Good mutations spread across the population

**When to use:** Most cases. Balanced exploration/exploitation. Resists getting stuck.

---

## Top-K

Keep the K best programs. Mutate from them. That's it.

```
     Gen 0           Gen 1           Gen 2           Gen 3
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ 1. [72%]│     │ 1. [78%]│     │ 1. [84%]│     │ 1. [93%]│  ← best
  │ 2. [65%]│ ──> │ 2. [72%]│ ──> │ 2. [78%]│ ──> │ 2. [89%]│
  │ 3. [58%]│     │ 3. [65%]│     │ 3. [72%]│     │ 3. [84%]│
  │ 4. [51%]│     │ 4. [58%]│     │ 4. [65%]│     │ 4. [78%]│
  │ 5. [45%]│     │ 5. [51%]│     │ 5. [58%]│     │ 5. [72%]│
  └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

**How selection works:**

1. Pick a parent from the top K (usually the best, sometimes random from top K)
2. LLM generates a mutated version
3. Evaluate the mutant
4. If it scores in the top K → insert, push out the worst
5. If it doesn't → discard

No islands. No adaptation. No migration. Greedy hill-climbing with a small memory buffer.

**When to use:** Quick experiments, baselines. Fewer API calls. Risk: premature convergence.

---

## EvoX

Co-evolution. Two populations evolve simultaneously.

```
  ┌─────────────────────────┐         ┌─────────────────────────┐
  │   Agent Code Population │         │  Search Strategy Pop    │
  │                         │         │                         │
  │  ┌─────┐ ┌─────┐       │ strats  │  ┌───────────┐          │
  │  │ v1  │ │ v2  │       │ produce │  │ "focus on │          │
  │  │ 95% │ │ 88% │       │ agents  │  │  prompt"  │          │
  │  └─────┘ └─────┘       │ ──────> │  └───────────┘          │
  │  ┌─────┐ ┌─────┐       │         │  ┌───────────┐          │
  │  │ v3  │ │ v4  │       │ agent   │  │ "rewrite  │          │
  │  │ 91% │ │ 72% │       │ scores  │  │  tools"   │          │
  │  └─────┘ └─────┘       │ select  │  └───────────┘          │
  │                         │ strats  │  ┌───────────┐          │
  │                         │ <────── │  │ "tune     │          │
  │                         │         │  │  config"  │          │
  └─────────────────────────┘         └─────────────────────────┘
```

**How selection works:**

1. **Strategy selection:** Pick a search strategy from the strategy population
2. **Strategy applies:** The chosen strategy tells the LLM *how* to mutate
   - "Focus on improving the system prompt"
   - "Try rewriting the tool implementations"
   - "Experiment with different config values"
3. **Agent evaluation:** The mutated agent runs on tasks, gets a score
4. **Dual feedback:**
   - Good agent score → the agent enters the agent population
   - Good agent score → the strategy that produced it gets credit
   - Bad agent score → the strategy gets penalized
5. **Strategy evolution:** Strategies themselves get mutated and selected

The system learns *how to search better* while searching. A strategy that says "focus on tool descriptions" survives if that approach actually produces winning agents.

**When to use:** Long runs with budget. Most expensive. Finds novel solutions others miss.

---

## Beam Search

Fixed-width frontier. Depth-first refinement with pruning.

```
                         [80%]              ← depth 0 (root)
                       /   |   \
                    /      |      \
                [85%]    [77%]    [82%]     ← depth 1 (expand all, keep beam=3)
               /  \      /  \     /  \
           [89%] [86%] [83%] ... [87%] ..   ← depth 2 (expand, prune to beam=3)
            / \    |
        [93%] [90%] [88%]                  ← depth 3
         ↑
        best
```

**How selection works:**

1. Start with beam of width W (e.g., 3 candidates)
2. For each candidate in the beam, generate children via LLM mutation
3. Evaluate all children
4. Keep only the top W across all children → new beam
5. Repeat, going deeper

Unlike Top-K which always mutates from the global best, beam search follows **lineages** — if v1 produced a good child, that child gets expanded next round even if another lineage has a higher score. This means it explores depth more than breadth.

**When to use:** When you want to deeply refine a few promising directions rather than broadly explore.

---

## Best-of-N

The simplest possible strategy. No memory. No population.

```
                    ┌────────┐
                    │Current │
                    │ Best   │
                    │  80%   │
                    └───┬────┘
                        │
          ┌─────┬───────┼───────┬─────┐
          ▼     ▼       ▼       ▼     ▼
       [88%] [72%]   [91%]   [85%] [79%]    ← generate N=5 variants
                       ↑
                    ★ BEST
                       │
                    ┌──▼─────┐
                    │  New   │
                    │ Best   │
                    │  91%   │               ← becomes parent for next round
                    └────────┘
```

**How selection works:**

1. Take the current best agent code
2. Generate N variants by asking the LLM to improve it N times
3. Evaluate all N variants
4. The single highest scorer becomes the new "current best"
5. Repeat

No population history. No adaptation. No islands. Each round starts fresh from the single best. Pure parallel sampling — relies entirely on the LLM generating at least one good variant per batch.

**When to use:** Sanity check baseline. "Can the LLM improve this code at all?"

---

## Comparison

| | AdaEvolve | Top-K | EvoX | Beam Search | Best-of-N |
|---|---|---|---|---|---|
| **Population** | Multi-island | Single list | Dual (code + strategy) | Beam frontier | None (single best) |
| **Adaptation** | Per-island intensity | None | Strategy evolution | None | None |
| **Diversity** | High (islands + migration) | Low | High (dual pop) | Medium | Low |
| **API cost/iter** | Medium | Low | High (2x populations) | Medium | High (N evals/iter) |
| **Convergence risk** | Low | High | Low | Medium | High |
| **Best for** | Default choice | Quick tests | Long deep runs | Refining lineages | Baseline sanity |

## CLI Usage

```bash
python -m src.evolve.run_evolve --search adaevolve   # default
python -m src.evolve.run_evolve --search topk
python -m src.evolve.run_evolve --search evox
python -m src.evolve.run_evolve --search beam_search
python -m src.evolve.run_evolve --search best_of_n
```
