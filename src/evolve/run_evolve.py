#!/usr/bin/env python3
"""Evolve agent code across GDPval slices.

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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from skydiscover.api import run_discovery

from src.evolve.cli import parse_args, setup_logging
from src.evolve.config import build_config
from src.evolve.slices import run_full_slice_eval

log = logging.getLogger("evolve")


def _code_stats(code: str) -> str:
    return f"{len(code):,} chars, {code.count(chr(10)) + 1} lines"


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
    """Run SkyDiscover on one slice. Returns (best_code, initial_score, best_score)."""

    os.environ["EVOLVE_SLICE"] = slice_name

    log.info("")
    log.info("  Evolving %s", slice_name)
    log.info("  Strategy: %s  |  Model: %s", search, model)
    log.info("  Iterations: %d  |  Tasks/eval: %d", iterations, sample_size)
    log.info("  Input code: %s", _code_stats(current_code))
    log.info("")

    config = build_config(model=model, iterations=iterations, search=search)
    t0 = time.monotonic()

    result = run_discovery(
        evaluator=evaluator_path,
        initial_program=current_code,
        config=config,
        iterations=iterations,
        output_dir=str(output_dir / f"skydiscover_{slice_name}"),
        cleanup=False,
    )

    elapsed = time.monotonic() - t0
    initial = result.initial_score or 0.0
    best = result.best_score
    best_code = result.best_solution or current_code
    improved = best > initial

    log.info("")
    log.info("  Evolution complete for %s (%.0fs)", slice_name, elapsed)
    log.info("  Score: %.1f%% -> %.1f%% %s",
             initial * 100, best * 100,
             "(IMPROVED)" if improved else "(no change)")
    if improved and best_code != current_code:
        log.info("  New code: %s", _code_stats(best_code))
    log.info("")

    return (best_code if improved else current_code), initial, best


def main():
    args = parse_args()

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/evolve_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.verbose, output_dir / "evolve.log")

    # Seed
    seed_path = _PROJECT_ROOT / "src" / "evolve" / "seeds" / f"{args.seed}.py"
    if not seed_path.exists():
        log.error("Seed not found: %s", seed_path); sys.exit(1)
    current_code = seed_path.read_text()

    # Evaluator env
    os.environ["EVOLVE_SAMPLE_SIZE"] = str(args.sample_size)
    os.environ["EVOLVE_WORKING_DIR"] = str(output_dir / "evolve_workspace")
    os.environ["EVOLVE_AGENT_MODEL"] = args.agent_model
    if args.judge_model:
        os.environ["EVOLVE_JUDGE_MODEL"] = args.judge_model
    evaluator_path = str(Path(__file__).resolve().parent / "evaluator.py")

    # Banner
    log.info("=" * 60)
    log.info("RSI EVOLVE")
    log.info("  Seed:   %s (%s)", args.seed, _code_stats(current_code))
    log.info("  Agent model:    %s (%s tier)", args.agent_model, args.tier)
    log.info("  Mutation model: %s", args.mutation_model)
    log.info("  Slices: %s", " -> ".join(args.slices))
    log.info("  Iters:  %d/slice, %d tasks/eval", args.iterations, args.sample_size)
    log.info("  Search: %s", args.search)
    log.info("  Output: %s", output_dir)
    log.info("=" * 60)

    (output_dir / "harness_initial.py").write_text(current_code)
    trajectory: list[dict] = []

    # ── Evolve loop ──
    for i, s in enumerate(args.slices):
        t0 = time.monotonic()
        log.info("-" * 60)
        log.info("SLICE %s (%d/%d)", s, i + 1, len(args.slices))
        log.info("-" * 60)

        (output_dir / f"harness_{s}_input.py").write_text(current_code)

        try:
            current_code, initial, best = evolve_slice(
                current_code, s, args.mutation_model, args.search,
                args.iterations, args.sample_size, evaluator_path, output_dir,
            )
        except Exception as e:
            log.exception("Evolution failed on %s", s)
            initial, best = 0.0, 0.0

        (output_dir / f"harness_{s}_evolved.py").write_text(current_code)

        # Optional full eval
        full_score = None
        if args.full_eval:
            try:
                full_score = run_full_slice_eval(
                    current_code, s, True, args.judge_model, output_dir,
                )["avg_score"]
            except Exception as e:
                log.exception("Full eval failed on %s", s)

        duration = time.monotonic() - t0
        trajectory.append({
            "slice": s, "iteration": i,
            "initial_score": initial, "evolved_score": best,
            "full_eval_score": full_score,
            "code_lines": current_code.count("\n") + 1,
            "duration_s": duration,
        })
        (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        log.info("Slice %s: %.0fs", s, duration)

    # ── Summary ──
    (output_dir / "harness_final.py").write_text(current_code)

    log.info("=" * 60)
    log.info("DONE")
    log.info("  %-6s %9s %9s %6s", "Slice", "Initial", "Evolved", "Lines")
    for t in trajectory:
        log.info("  %-6s %8.1f%% %8.1f%% %6d", t["slice"],
                 t["initial_score"] * 100, t["evolved_score"] * 100, t["code_lines"])
    log.info("  Output:  %s", output_dir)
    log.info("  Harness: %s/harness_final.py", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
