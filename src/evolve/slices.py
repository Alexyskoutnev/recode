from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

log = logging.getLogger("evolve")


def load_zipper_slice(slice_name: str) -> list[str]:
    with open("data/processed/zipper_split.json") as f:
        splits = json.load(f)
    all_slices = {**splits.get("dev_slices", {}), **splits.get("eval_slices", {})}
    if slice_name not in all_slices:
        raise KeyError(f"Slice '{slice_name}' not found. Available: {list(all_slices.keys())}")
    return all_slices[slice_name]


def run_full_slice_eval(
    code: str,
    slice_name: str,
    use_judge: bool,
    judge_model: str | None,
    output_dir: Path,
) -> dict:
    """Run a full 22-task evaluation with the evolved agent code."""
    from src.eval.runner import GDPvalRunner
    from src.data.registry import DatasetRegistry
    from src.evolve.evaluator import _load_agent_from_code

    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")
    id_set = set(load_zipper_slice(slice_name))
    samples = [s for s in all_samples if s.id in id_set]

    agent = _load_agent_from_code(code)
    if agent is None:
        return {"slice": slice_name, "avg_score": 0.0, "error": "load failed"}

    runner = GDPvalRunner(
        agent=agent,
        working_dir=output_dir / f"workspace_{slice_name}",
        use_judge=use_judge, judge_model=judge_model,
    )
    result = asyncio.run(runner.run_batch(samples, concurrency=3))
    GDPvalRunner.save_results(result, output_dir / slice_name)

    log.info("Full eval %s: %.1f%% (%d/%d completed, %d errors, %.0fs)",
             slice_name, result.avg_score * 100,
             result.num_completed, len(result.traces),
             result.num_errors, result.total_duration_s)

    return {
        "slice": slice_name,
        "avg_score": result.avg_score,
        "num_tasks": len(result.traces),
        "num_completed": result.num_completed,
        "num_errors": result.num_errors,
        "duration_s": result.total_duration_s,
    }
