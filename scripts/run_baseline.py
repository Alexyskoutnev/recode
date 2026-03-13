#!/usr/bin/env python3
"""Run baseline evaluation: all agents across all zipper slices.

Runs each agent (claude, gemini, codex) on every slice (S1–S8 dev, E1–E2 eval),
checkpointing results after each slice. Produces a trajectory JSON tracking
scores per agent per slice.

Usage:
    # Run all agents on all slices
    python scripts/run_baseline.py

    # Run specific agents
    python scripts/run_baseline.py --agents claude gemini

    # Run only dev slices (skip eval)
    python scripts/run_baseline.py --dev-only

    # Resume from a checkpoint
    python scripts/run_baseline.py --resume results/baseline_20260310_143022

    # Override concurrency and max turns
    python scripts/run_baseline.py --concurrency 10 --max-turns 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.data.registry import DatasetRegistry
from src.data.splitters.zipper import zipper_split, save_split
from src.data.types import Sample
from src.eval.agents import AGENTS
from src.eval.runner import GDPvalRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEV_SLICES = [f"S{i}" for i in range(1, 9)]
EVAL_SLICES = ["E1", "E2"]
ALL_SLICES = DEV_SLICES + EVAL_SLICES


# ===================================================================
# Data loading
# ===================================================================

def load_gdpval_samples() -> list[Sample]:
    """Load all 220 GDPval samples from the dataset registry."""
    logger.info("Loading GDPval dataset")
    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    samples = registry.get_samples("gdpval")
    logger.info("Loaded %d GDPval tasks", len(samples))
    return samples


def generate_zipper_split(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Create the deterministic zipper split and persist it to disk.

    Returns a flat dict mapping slice name (S1–S8, E1–E2) to its samples.
    """
    split = zipper_split(samples)

    split_path = Path("data/processed/zipper_split.json")
    save_split(split, split_path)
    logger.info("Zipper split saved to %s", split_path)

    all_slices: dict[str, list[Sample]] = {}
    all_slices.update(split.dev_slices)
    all_slices.update(split.eval_slices)
    return all_slices


# ===================================================================
# Checkpoint persistence
# ===================================================================

def load_checkpoint(run_dir: Path) -> dict[str, Any]:
    """Load an existing trajectory checkpoint, or return an empty one."""
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        with open(traj_path) as f:
            data = json.load(f)
        logger.info("Loaded checkpoint from %s (%d agent(s))",
                     traj_path, len(data.get("agents", {})))
        return data
    return {"agents": {}, "slices_completed": {}}


def save_checkpoint(run_dir: Path, trajectory: dict[str, Any]) -> None:
    """Persist the trajectory checkpoint to disk."""
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    logger.debug("Checkpoint saved to %s", traj_path)


def is_slice_done(trajectory: dict[str, Any], agent: str, slice_name: str) -> bool:
    """Return True if the given agent+slice pair was already completed."""
    return slice_name in trajectory.get("slices_completed", {}).get(agent, [])


def record_slice_result(
    trajectory: dict[str, Any],
    agent: str,
    slice_name: str,
    summary: dict[str, Any],
) -> None:
    """Write a slice result into the trajectory dict."""
    trajectory.setdefault("agents", {}).setdefault(agent, {"scores": {}})
    trajectory["agents"][agent]["scores"][slice_name] = summary

    trajectory.setdefault("slices_completed", {}).setdefault(agent, [])
    trajectory["slices_completed"][agent].append(slice_name)


# ===================================================================
# Single agent+slice execution
# ===================================================================

def build_score_summary(result: Any) -> dict[str, Any]:
    """Extract a JSON-serialisable score summary from a BatchResult."""
    tasks = []
    for trace in result.traces:
        ev = trace.eval_result
        tasks.append({
            "task_id": trace.task_id,
            "score": ev.normalized_score if ev else 0.0,
            "raw_score": ev.score if ev else 0.0,
            "max_score": ev.max_score if ev else 0.0,
            "error": trace.error,
            "duration_s": trace.duration_s,
        })
    return {
        "avg_score": result.avg_score,
        "num_tasks": len(result.traces),
        "num_completed": result.num_completed,
        "num_errors": result.num_errors,
        "total_duration_s": result.total_duration_s,
        "tasks": tasks,
    }


async def run_agent_on_slice(
    agent_name: str,
    slice_name: str,
    samples: list[Sample],
    run_dir: Path,
    max_turns: int,
    concurrency: int,
    judge_model: str | None,
    use_judge: bool,
) -> dict[str, Any]:
    """Run one agent on one slice and return a score summary dict.

    Creates an isolated output directory at <run_dir>/<agent>/<slice>/
    containing traces.json, eval.json, and a workspace/ folder.
    """
    agent = AGENTS[agent_name](max_turns=max_turns)

    output_dir = run_dir / agent_name / slice_name
    work_dir = output_dir / "workspace"
    work_dir.mkdir(parents=True, exist_ok=True)

    runner = GDPvalRunner(
        agent=agent,
        working_dir=work_dir,
        use_judge=use_judge,
        judge_model=judge_model,
    )

    logger.info("[%s | %s] Starting %d tasks (concurrency=%d)",
                agent_name, slice_name, len(samples), concurrency)

    def on_progress(done: int, total: int, *_: Any) -> None:
        logger.info("[%s | %s] Progress: %d/%d", agent_name, slice_name, done, total)

    result = await runner.run_batch(
        samples,
        concurrency=concurrency,
        progress_callback=on_progress,
    )

    runner.save_results(result, output_dir)
    logger.info("[%s | %s] Results saved to %s", agent_name, slice_name, output_dir)

    summary = build_score_summary(result)
    logger.info(
        "[%s | %s] Score: %.1f%% (%d/%d completed, %d errors, %.0fs)",
        agent_name, slice_name,
        summary["avg_score"] * 100,
        summary["num_completed"], summary["num_tasks"],
        summary["num_errors"], summary["total_duration_s"],
    )
    return summary


# ===================================================================
# Trajectory display
# ===================================================================

def log_trajectory_table(trajectory: dict[str, Any]) -> None:
    """Log a tabular summary of scores across all agents and slices."""
    agents = trajectory.get("agents", {})
    if not agents:
        logger.warning("No results to display")
        return

    col_width = 7
    header = f"{'Agent':<15}" + "".join(f"{s:>{col_width}}" for s in ALL_SLICES) + f"{'Avg':>{col_width + 1}}"
    separator = "-" * len(header)

    lines = [
        "",
        "=" * len(header),
        "SCORE TRAJECTORY (baseline)",
        "=" * len(header),
        header,
        separator,
    ]

    for agent_name, agent_data in agents.items():
        scores = agent_data.get("scores", {})
        row = f"{agent_name:<15}"
        vals: list[float] = []

        for s in ALL_SLICES:
            if s in scores:
                val = scores[s]["avg_score"]
                row += f"{val:>{col_width}.1%}"
                vals.append(val)
            else:
                row += f"{'—':>{col_width}}"

        if vals:
            row += f"{sum(vals) / len(vals):>{col_width + 1}.1%}"
        else:
            row += f"{'—':>{col_width + 1}}"
        lines.append(row)

    lines.append("=" * len(header))
    logger.info("\n".join(lines))


# ===================================================================
# CLI argument parsing
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation: all agents across all zipper slices",
    )
    parser.add_argument(
        "--agents", nargs="+", default=list(AGENTS.keys()),
        choices=list(AGENTS.keys()),
        help="Agents to evaluate (default: all)",
    )
    parser.add_argument("--dev-only", action="store_true",
                        help="Only run dev slices S1–S8")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run eval slices E1–E2")
    parser.add_argument("--slices", nargs="+", default=None,
                        help="Run specific slices (e.g. --slices S1 S2 E1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing run directory")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="Max agent turns per task (default: 30)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Parallel tasks per slice (default: 50)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Use keyword heuristic instead of LLM judge")
    parser.add_argument("--judge-model", type=str, default="gpt-5.4",
                        help="Judge model (default: gpt-5.4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def resolve_target_slices(args: argparse.Namespace) -> list[str]:
    """Determine which slices to run based on CLI flags."""
    if args.slices:
        return args.slices
    if args.eval_only:
        return EVAL_SLICES
    if args.dev_only:
        return DEV_SLICES
    return ALL_SLICES


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Return the output directory — either a resumed one or a fresh timestamp."""
    if args.resume:
        run_dir = Path(args.resume)
        logger.info("Resuming from %s", run_dir)
        return run_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results/baseline_{timestamp}")


# ===================================================================
# Main orchestration
# ===================================================================

def setup_logging(run_dir: Path, verbose: bool) -> None:
    """Configure logging to both console and a per-run log file.

    The log file is written to <run_dir>/run.log so each baseline run
    has a persistent, inspectable record of everything that happened.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    # Root logger — all modules log through this
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler — always captures DEBUG regardless of --verbose
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    logger.info("Logging to %s", log_path)


def log_plan(agents: list[str], slices: list[str], args: argparse.Namespace, run_dir: Path) -> None:
    """Log the evaluation plan before execution starts."""
    evaluator = "keyword heuristic" if args.no_judge else f"LLM judge ({args.judge_model})"
    total = len(agents) * len(slices)
    logger.info(
        "Baseline evaluation plan:\n"
        "  Agents:      %s\n"
        "  Slices:      %s\n"
        "  Tasks/slice: 22\n"
        "  Max turns:   %d\n"
        "  Concurrency: %d\n"
        "  Evaluator:   %s\n"
        "  Output:      %s\n"
        "  Total runs:  %d (%d agents × %d slices)",
        ", ".join(agents), ", ".join(slices),
        args.max_turns, args.concurrency, evaluator, run_dir,
        total, len(agents), len(slices),
    )


async def main() -> None:
    args = parse_args()

    target_slices = resolve_target_slices(args)
    run_dir = resolve_run_dir(args)

    setup_logging(run_dir, args.verbose)

    trajectory = load_checkpoint(run_dir)

    samples = load_gdpval_samples()
    slices = generate_zipper_split(samples)

    # Validate requested slices exist
    for s in target_slices:
        if s not in slices:
            logger.error("Slice '%s' not found. Available: %s", s, list(slices.keys()))
            sys.exit(1)

    log_plan(args.agents, target_slices, args, run_dir)

    # ── Main loop: sequential — slice by slice, agent by agent ──
    # Sequential execution ensures consistency with the RSI loop
    # where each slice's traces inform the next iteration's harness.
    for slice_name in target_slices:
        slice_samples = slices[slice_name]
        slice_type = "eval" if slice_name.startswith("E") else "dev"

        for agent_name in args.agents:
            if is_slice_done(trajectory, agent_name, slice_name):
                prev = trajectory["agents"][agent_name]["scores"][slice_name]["avg_score"]
                logger.info("[%s | %s] Already done (%.1f%%), skipping",
                            agent_name, slice_name, prev * 100)
                continue

            logger.info(
                "━━━ %s | %s (%s) — %d tasks ━━━",
                agent_name, slice_name, slice_type, len(slice_samples),
            )

            summary = await run_agent_on_slice(
                agent_name=agent_name,
                slice_name=slice_name,
                samples=slice_samples,
                run_dir=run_dir,
                max_turns=args.max_turns,
                concurrency=args.concurrency,
                judge_model=args.judge_model,
                use_judge=not args.no_judge,
            )

            record_slice_result(trajectory, agent_name, slice_name, summary)
            save_checkpoint(run_dir, trajectory)

        log_trajectory_table(trajectory)

    # ── Final output ──
    log_trajectory_table(trajectory)
    save_checkpoint(run_dir, trajectory)
    logger.info("Results saved to %s/", run_dir)


if __name__ == "__main__":
    asyncio.run(main())
