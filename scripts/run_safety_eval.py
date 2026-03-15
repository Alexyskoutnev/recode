#!/usr/bin/env python3
"""Post-hoc safety evaluation of evolved agent code.

Takes a harness file (e.g. harness_final.py from a Track A run) and evaluates
it on the FULL AgentHarm (208) + ToolEmu (144) benchmarks — not sampled, all
352 tasks. This gives a definitive answer to "did evolution degrade safety?"

Compare the seed agent's safety score vs the evolved agent's safety score.

Usage:
    # Evaluate an evolved harness
    python scripts/run_safety_eval.py results/evolve_20260315_145821/harness_final.py

    # Evaluate the original seed for comparison
    python scripts/run_safety_eval.py src/evolve/seeds/openai.py

    # With model override and custom output
    python scripts/run_safety_eval.py harness_final.py --model gpt-5.4 --output results/safety_eval

    # Specific benchmarks only
    python scripts/run_safety_eval.py harness_final.py --benchmarks agentharm
    python scripts/run_safety_eval.py harness_final.py --benchmarks toolemu

    # Limit concurrency
    python scripts/run_safety_eval.py harness_final.py --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.data.registry import DatasetRegistry
from src.eval.runner import GDPvalRunner
from src.evolve.evaluator import _load_agent_from_code

logger = logging.getLogger("safety_eval")


def setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

    for name in ("httpx", "httpcore", "urllib3", "google", "openai", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_all_safety_samples(benchmarks: list[str]) -> list:
    """Load all samples from the specified safety benchmarks."""
    registry = DatasetRegistry()
    all_samples = []
    for name in benchmarks:
        try:
            registry.load_dataset(name)
            samples = registry.get_samples(name)
            all_samples.extend(samples)
            logger.info("Loaded %s: %d samples", name, len(samples))
        except (FileNotFoundError, KeyError) as e:
            logger.error("Failed to load %s: %s", name, e)
    return all_samples


def run_safety_eval(
    code: str,
    samples: list,
    model: str | None,
    judge_model: str | None,
    output_dir: Path,
    concurrency: int,
) -> dict:
    """Run agent on all safety samples and return results."""
    agent = _load_agent_from_code(code, model=model)
    if agent is None:
        logger.error("Failed to load agent from code")
        return {"error": "load failed", "avg_score": 0.0}

    logger.info("Agent loaded: %s", agent.name())
    logger.info("Running %d safety tasks (concurrency=%d)...", len(samples), concurrency)

    workspace = output_dir / "workspace"

    completed = 0
    def _progress(done: int, total: int, trace) -> None:
        nonlocal completed
        completed = done
        score_str = ""
        if trace.eval_result and trace.eval_result.max_score > 0:
            score_str = f" {trace.eval_result.normalized_score:.0%}"
        status = "ok" if trace.error is None else "ERR"
        if done % 10 == 0 or done == total:
            logger.info("Progress: %d/%d %s [%s]%s",
                        done, total, trace.task_id[:8], status, score_str)

    runner = GDPvalRunner(
        agent=agent,
        working_dir=workspace,
        use_judge=True,
        judge_model=judge_model,
    )

    t0 = time.monotonic()
    result = asyncio.run(
        runner.run_batch(samples, concurrency=concurrency, progress_callback=_progress)
    )
    elapsed = time.monotonic() - t0

    # Break down by benchmark
    by_benchmark: dict[str, list[float]] = {}
    for trace in result.traces:
        bench = "unknown"
        # Find the sample to get benchmark name
        for s in samples:
            if s.id == trace.task_id:
                bench = s.benchmark
                break
        if trace.eval_result and trace.eval_result.max_score > 0:
            by_benchmark.setdefault(bench, []).append(trace.eval_result.normalized_score)

    summary = {
        "overall": {
            "avg_score": result.avg_score,
            "num_tasks": len(samples),
            "num_completed": result.num_completed,
            "num_errors": result.num_errors,
            "duration_s": elapsed,
        },
        "by_benchmark": {},
    }

    for bench, scores in sorted(by_benchmark.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        summary["by_benchmark"][bench] = {
            "avg_score": avg,
            "num_scored": len(scores),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc safety evaluation of evolved agent code"
    )
    parser.add_argument("harness", type=str,
                        help="Path to the agent harness file (e.g. harness_final.py)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model override for the agent")
    parser.add_argument("--judge-model", type=str, default="gpt-5.4",
                        help="Judge model (default: gpt-5.4)")
    parser.add_argument("--benchmarks", nargs="+", default=["agentharm", "toolemu"],
                        choices=["agentharm", "toolemu", "asb"],
                        help="Safety benchmarks to evaluate (default: agentharm toolemu)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Parallel tasks (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    harness_path = Path(args.harness)
    if not harness_path.exists():
        print(f"Error: {harness_path} not found")
        sys.exit(1)

    code = harness_path.read_text()
    lines = code.count("\n") + 1

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/safety_eval_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.verbose, output_dir / "safety_eval.log")

    logger.info("=" * 60)
    logger.info("SAFETY EVALUATION")
    logger.info("  Harness:    %s (%d lines)", harness_path, lines)
    logger.info("  Model:      %s", args.model or "default from harness")
    logger.info("  Judge:      %s", args.judge_model)
    logger.info("  Benchmarks: %s", ", ".join(args.benchmarks))
    logger.info("  Output:     %s", output_dir)
    logger.info("=" * 60)

    # Load safety samples
    samples = load_all_safety_samples(args.benchmarks)
    if not samples:
        logger.error("No safety samples loaded — check data/raw/")
        sys.exit(1)

    logger.info("Total safety tasks: %d", len(samples))

    # Run evaluation
    summary = run_safety_eval(
        code=code,
        samples=samples,
        model=args.model,
        judge_model=args.judge_model,
        output_dir=output_dir,
        concurrency=args.concurrency,
    )

    # Save results
    results_path = output_dir / "safety_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("  Overall: %.1f%% (%d/%d completed, %d errors)",
                summary["overall"]["avg_score"] * 100,
                summary["overall"]["num_completed"],
                summary["overall"]["num_tasks"],
                summary["overall"]["num_errors"])
    for bench, data in summary.get("by_benchmark", {}).items():
        logger.info("  %s: %.1f%% (%d scored)",
                    bench, data["avg_score"] * 100, data["num_scored"])
    logger.info("  Duration: %.0fs", summary["overall"]["duration_s"])
    logger.info("  Results:  %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
