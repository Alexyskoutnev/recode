"""Evolve agent code across GDPval slices.

This is the main orchestrator for recursive self-improvement. It takes a seed
agent (a self-contained Python file that calls an LLM API with tools) and
iteratively improves it using SkyDiscover — an evolutionary search framework
that treats source code as the "genome" and task completion scores as fitness.

The loop works like this:
  1. Load the seed agent code (e.g. seeds/openai.py)
  2. For each zipper slice (S1, S2, ...):
     a. Score the current code on a sample of tasks from that slice
     b. Run N iterations of mutation+evaluation:
        - A mutation LLM (e.g. gemini-2.5-flash) reads the code and rewrites it
        - The mutant is scored on a fresh sample of tasks
        - SkyDiscover's population keeps high-scoring variants
     c. Extract the best code and carry it to the next slice
  3. Save the final evolved code + trajectory of scores

Each slice has completely different tasks, so improvements must generalize —
code that overfits to one slice's tasks will score poorly on the next.

Usage:
    python -m src.evolve.run_evolve --slices S1 S2 S3 --iterations 5
    python -m src.evolve.run_evolve --seed openai --search topk --iterations 20
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from skydiscover.api import run_discovery

from src.evolve.cli import parse_args, setup_logging
from src.evolve.config import build_config
from src.evolve.slices import run_full_slice_eval

log = logging.getLogger("evolve")


# ── Logging helpers ───────────────────────────────────────────────────────

def _code_stats(code: str) -> str:
    return f"{len(code):,} chars, {code.count(chr(10)) + 1} lines"


def _log_banner(title: str, items: dict[str, str]) -> None:
    """Log a boxed banner with key-value pairs as a single log entry."""
    sep = "=" * 60
    lines = [sep, title]
    for k, v in items.items():
        lines.append(f"  {k + ':':<18}{v}")
    lines.append(sep)
    log.info("\n".join(lines))


def _log_table(header: list[str], rows: list[list[str]], footer: dict[str, str] | None = None) -> None:
    """Log a formatted results table as a single log entry."""
    fmt = "  %-6s %9s %9s %6s"
    lines = ["=" * 60, "DONE", fmt % tuple(header)]
    for row in rows:
        lines.append(fmt % tuple(row))
    if footer:
        for k, v in footer.items():
            lines.append(f"  {k}  {v}")
    lines.append("=" * 60)
    log.info("\n".join(lines))


# ── Core logic ────────────────────────────────────────────────────────────

def evolve_slice(
    current_code: str,
    slice_name: str,
    model: str,
    search: str,
    iterations: int,
    sample_size: int,
    evaluator_path: str,
    output_dir: Path,
) -> tuple[str, float, float]:
    """Run SkyDiscover on one slice. Returns (best_code, initial_score, best_score).

    This is where the actual evolution happens for a single slice:

    1. Set EVOLVE_SLICE env var so the evaluator knows which tasks to sample from
    2. Build a SkyDiscover config (search strategy, LLM settings, population params)
    3. Call run_discovery() which:
       a. Evaluates the initial program to get a baseline score
       b. Seeds the population (e.g. 2 islands for adaevolve)
       c. For each iteration:
          - UCB bandit selects which island to explore
          - Picks a parent program from that island's archive
          - Sends the code to the mutation LLM (e.g. gemini-2.5-flash)
          - The LLM returns a modified version (diff or full rewrite)
          - The mutant is scored by calling evaluator.evaluate(temp_file)
          - The mutant enters the population if it passes selection
          - Periodically migrates best programs between islands
       d. Returns the best program found across all iterations
    4. Compare best vs initial — only carry forward if strictly improved
    """
    # The evaluator reads this env var to know which slice's tasks to sample
    os.environ["EVOLVE_SLICE"] = slice_name

    log.info("Evolving %s  |  %s  |  %s  |  %d iters, %d tasks",
             slice_name, search, model, iterations, sample_size)

    # Build the SkyDiscover config: search strategy, mutation LLM, population
    # params, checkpoint intervals, etc. See config.py for details.
    config = build_config(model=model, iterations=iterations, search=search)
    t0 = time.monotonic()

    # run_discovery is SkyDiscover's main entry point. It orchestrates the
    # entire evolutionary loop: eval initial → seed population → iterate
    # (mutate+eval+select) → return best.
    #
    # Args:
    #   evaluator     — path to evaluator.py; SkyDiscover imports it and calls
    #                   evaluate(program_path) to score each code variant
    #   initial_program — the seed agent source code (string)
    #   config        — search strategy, LLM, population, checkpointing settings
    #   iterations    — number of mutation+eval cycles to run
    #   output_dir    — where to write best program, checkpoints, logs, stats
    #   cleanup       — False = keep all artifacts for analysis
    result = run_discovery(
        evaluator=evaluator_path,
        initial_program=current_code,
        config=config,
        iterations=iterations,
        output_dir=str(output_dir / f"skydiscover_{slice_name}"),
        cleanup=False,
    )

    elapsed = time.monotonic() - t0

    # Extract results — initial_score is what the seed scored, best_score is
    # the highest fitness found across all iterations
    initial = result.initial_score or 0.0
    best = result.best_score
    best_code = result.best_solution or current_code
    improved = best > initial

    log.info("%s done (%.0fs): %.1f%% -> %.1f%% %s",
             slice_name, elapsed, initial * 100, best * 100,
             "(IMPROVED)" if improved else "(no change)")

    # Only carry forward improved code — if no mutation beat the seed,
    # we keep the original to avoid regression
    return (best_code if improved else current_code), initial, best


def main():
    args = parse_args()

    # Create timestamped output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/evolve_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dual logging: console (human-readable) + file (detailed with module names)
    setup_logging(args.verbose, output_dir / "evolve.log")

    # Load the seed agent — a self-contained Python file with tools + agent loop.
    # This is the starting "genome" that evolution will mutate.
    seed_dir = Path(__file__).resolve().parent / "seeds"
    seed_path = seed_dir / f"{args.seed}.py"
    if not seed_path.exists():
        log.error("Seed not found: %s", seed_path)
        sys.exit(1)
    current_code = seed_path.read_text()

    # Configure the evaluator via env vars. The evaluator runs in a separate
    # process (SkyDiscover imports evaluator.py independently), so we can't
    # pass these as function args — env vars are the bridge.
    os.environ["EVOLVE_SAMPLE_SIZE"] = str(args.sample_size)
    os.environ["EVOLVE_WORKING_DIR"] = str(output_dir / "evolve_workspace")
    os.environ["EVOLVE_AGENT_MODEL"] = args.agent_model
    if args.judge_model:
        os.environ["EVOLVE_JUDGE_MODEL"] = args.judge_model
    evaluator_path = str(Path(__file__).resolve().parent / "evaluator.py")

    _log_banner("RSI EVOLVE", {
        "Seed":           f"{args.seed} ({_code_stats(current_code)})",
        "Agent model":    f"{args.agent_model} ({args.tier} tier)",
        "Mutation model": args.mutation_model,
        "Slices":         " -> ".join(args.slices),
        "Search":         f"{args.search}  |  {args.iterations} iters/slice, {args.sample_size} tasks/eval",
        "Output":         str(output_dir),
    })

    # Save the starting code for reproducibility
    (output_dir / "harness_initial.py").write_text(current_code)
    trajectory: list[dict] = []

    # ── Main evolution loop ──
    # Process slices sequentially. Each slice uses completely different tasks,
    # so the code must generalize — it can't just memorize S1's tasks because
    # S2 will have entirely different ones. The best code from each slice
    # becomes the starting point for the next.
    for i, s in enumerate(args.slices):
        t0 = time.monotonic()
        log.info("── SLICE %s (%d/%d) ──", s, i + 1, len(args.slices))

        # Snapshot the code entering this slice (for debugging regressions)
        (output_dir / f"harness_{s}_input.py").write_text(current_code)

        try:
            current_code, initial, best = evolve_slice(
                current_code, s, args.mutation_model, args.search,
                args.iterations, args.sample_size, evaluator_path, output_dir,
            )
        except Exception:
            log.exception("Evolution failed on %s", s)
            initial, best = 0.0, 0.0

        # Save the best code found for this slice
        (output_dir / f"harness_{s}_evolved.py").write_text(current_code)

        # Optional: run the best code on ALL 22 tasks for a reliable score.
        # The evolution only samples 3 tasks per eval (fast but noisy), so
        # this gives a ground-truth score. Roughly doubles runtime per slice.
        full_score = None
        if args.full_eval:
            try:
                full_score = run_full_slice_eval(
                    current_code, s, True, args.judge_model, output_dir,
                )["avg_score"]
            except Exception:
                log.exception("Full eval failed on %s", s)

        duration = time.monotonic() - t0
        trajectory.append({
            "slice": s, "iteration": i,
            "initial_score": initial, "evolved_score": best,
            "full_eval_score": full_score,
            "code_lines": current_code.count("\n") + 1,
            "duration_s": duration,
        })
        # Write trajectory after each slice so progress survives crashes
        (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    # Save the final evolved code — this is the end product of the run
    (output_dir / "harness_final.py").write_text(current_code)

    _log_table(
        header=["Slice", "Initial", "Evolved", "Lines"],
        rows=[[t["slice"], f'{t["initial_score"]*100:.1f}%', f'{t["evolved_score"]*100:.1f}%', str(t["code_lines"])]
              for t in trajectory],
        footer={"Output:": str(output_dir), "Harness:": f"{output_dir}/harness_final.py"},
    )


if __name__ == "__main__":
    main()
