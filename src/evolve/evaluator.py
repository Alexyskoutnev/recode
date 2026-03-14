"""SkyDiscover evaluator bridge — scores an evolved HARNESS CODE on GDPval tasks.

SkyDiscover calls ``evaluate(program_path)`` where *program_path* is a temp file
containing the full evolved agent code (the entire custom harness — config,
prompt, tools, loop, agent class). We:

1. Read the evolved code from the file.
2. Dynamically load it as a Python module.
3. Find the agent class (subclass of BaseAgent).
4. Run a sample of GDPval tasks with that agent.
5. Return ``{"combined_score": avg_score}`` for SkyDiscover's fitness signal.

Configuration via environment variables (SkyDiscover imports this independently):
    EVOLVE_SLICE        — zipper slice name (default: S1)
    EVOLVE_SAMPLE_SIZE  — tasks per evaluation (default: 5)
    EVOLVE_WORKING_DIR  — workspace directory (default: results/evolve_workspace)
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

logger = logging.getLogger("evolve.eval")

# Cache loaded samples to avoid re-loading on every evaluation
_cached_samples: dict[str, list] = {}
_load_counter = 0
_eval_counter = 0  # tracks how many evaluations we've run


def _get_config():
    """Read configuration from environment variables."""
    return {
        "slice": os.environ.get("EVOLVE_SLICE", "S1"),
        "sample_size": int(os.environ.get("EVOLVE_SAMPLE_SIZE", "5")),
        "working_dir": os.environ.get("EVOLVE_WORKING_DIR", "results/evolve_workspace"),
        "agent_model": os.environ.get("EVOLVE_AGENT_MODEL", None),
        "judge_model": os.environ.get("EVOLVE_JUDGE_MODEL", None),
    }


def _load_zipper_slice(slice_name: str) -> list[str]:
    """Load task IDs for a specific zipper slice from the split file."""
    import json
    split_path = Path("data/processed/zipper_split.json")
    with open(split_path) as f:
        splits = json.load(f)
    all_slices = {**splits.get("dev_slices", {}), **splits.get("eval_slices", {})}
    if slice_name not in all_slices:
        raise KeyError(f"Slice '{slice_name}' not found. Available: {list(all_slices.keys())}")
    return all_slices[slice_name]


def _load_samples(slice_name: str):
    """Load and cache GDPval samples for a slice."""
    if slice_name in _cached_samples:
        return _cached_samples[slice_name]

    from src.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")

    task_ids = _load_zipper_slice(slice_name)
    id_set = set(task_ids)
    samples = [s for s in all_samples if s.id in id_set]
    _cached_samples[slice_name] = samples
    return samples


def _load_agent_from_code(code: str, model: str | None = None):
    """Dynamically load an agent from evolved code.

    Uses duck-typing: finds any class with ``run`` and ``name`` methods.
    The evolved code is fully self-contained — it defines its own BaseAgent
    and AgentResult, so we don't check inheritance from our project's classes.

    If model is provided, it's passed to the agent constructor to override
    the default model in the seed.

    Returns an instance of the agent, or None on failure.
    """
    global _load_counter
    _load_counter += 1
    module_name = f"evolved_agent_{_load_counter}"

    try:
        # Write code to a temp file so importlib can load it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="evolved_", delete=False,
            dir=str(_PROJECT_ROOT),
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            if spec is None or spec.loader is None:
                logger.error("[evolve-eval] Cannot create module spec from evolved code")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find the concrete agent class by duck-typing:
            # any non-abstract class with run() and name() methods
            import inspect
            agent_cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "run")
                    and hasattr(attr, "name")
                    and attr.__module__ == module_name
                    and not inspect.isabstract(attr)  # skip ABCs
                ):
                    agent_cls = attr
                    break

            if agent_cls is None:
                logger.error("[evolve-eval] No agent class (with run + name) found in evolved code")
                return None

            # Pass model override if provided
            try:
                return agent_cls(model=model) if model else agent_cls()
            except TypeError:
                return agent_cls()

        finally:
            os.unlink(temp_path)
            sys.modules.pop(module_name, None)

    except Exception as e:
        logger.error("[evolve-eval] Failed to load agent from code: %s\n%s",
                     e, traceback.format_exc())
        return None


def _run_eval(code: str) -> dict[str, float]:
    """Run GDPval tasks with the evolved agent code and return metrics."""
    import random
    import time as _time
    from src.eval.runner import GDPvalRunner

    global _eval_counter
    _eval_counter += 1
    eval_id = _eval_counter

    cfg = _get_config()
    slice_name = cfg["slice"]
    sample_size = cfg["sample_size"]
    working_dir = cfg["working_dir"]

    # Step 1: Compile check
    try:
        compile(code, "<evolved_agent>", "exec")
    except SyntaxError as e:
        logger.warning("[eval #%d] SYNTAX ERROR: %s → score=0", eval_id, e)
        return {"combined_score": 0.0, "syntax_error": 1.0}

    # Step 2: Load agent
    agent_model = cfg.get("agent_model")
    logger.info("[eval #%d] Loading evolved agent (%d lines, model=%s)...",
                eval_id, code.count("\n") + 1, agent_model or "default")
    agent = _load_agent_from_code(code, model=agent_model)
    if agent is None:
        logger.warning("[eval #%d] LOAD FAILED → score=0", eval_id)
        return {"combined_score": 0.0, "load_error": 1.0}
    logger.info("[eval #%d] Agent loaded: %s", eval_id, agent.name())

    # Step 3: Load task samples
    samples = _load_samples(slice_name)
    if len(samples) > sample_size:
        eval_samples = random.sample(samples, sample_size)
    else:
        eval_samples = samples

    task_ids = [s.id[:8] for s in eval_samples]
    logger.info("[eval #%d] Running %d tasks on %s: %s",
                eval_id, len(eval_samples), slice_name, ", ".join(task_ids))

    # Step 4: Run with per-task progress
    t0 = _time.monotonic()
    try:
        workspace = Path(working_dir) / f"eval_{slice_name}"

        def _progress(done: int, total: int, trace) -> None:
            score_str = ""
            if trace.eval_result and trace.eval_result.max_score > 0:
                score_str = f" {trace.eval_result.normalized_score:.0%}"
            status = "ok" if trace.error is None else "ERR"
            logger.info("[eval #%d] Task %d/%d %s [%s]%s",
                        eval_id, done, total, trace.task_id[:8], status, score_str)

        judge_model = cfg.get("judge_model")
        runner = GDPvalRunner(
            agent=agent,
            working_dir=workspace,
            use_judge=True,
            judge_model=judge_model,
        )

        result = asyncio.run(
            runner.run_batch(eval_samples, concurrency=3, progress_callback=_progress)
        )

        elapsed = _time.monotonic() - t0
        avg_score = result.avg_score
        num_completed = result.num_completed
        num_errors = result.num_errors

        logger.info(
            "[eval #%d] DONE — %s avg=%.1f%% (%d/%d ok, %d err) %.0fs",
            eval_id, slice_name, avg_score * 100,
            num_completed, len(eval_samples), num_errors, elapsed,
        )

        return {
            "combined_score": avg_score,
            "avg_score": avg_score,
            "num_completed": float(num_completed),
            "num_errors": float(num_errors),
            "num_tasks": float(len(eval_samples)),
        }

    except Exception as e:
        logger.error("[eval #%d] RUNTIME ERROR: %s", eval_id, e, exc_info=True)
        return {"combined_score": 0.0, "runtime_error": 1.0}


def evaluate(program_path: str) -> dict[str, float]:
    """SkyDiscover entry point — evaluate evolved harness code.

    Args:
        program_path: Path to temp file containing the full evolved agent code.

    Returns:
        Dict with ``combined_score`` and additional metrics.
    """
    try:
        with open(program_path, "r") as f:
            code = f.read()

        if not code.strip():
            return {"combined_score": 0.0, "error": 1.0}

        return _run_eval(code)

    except Exception as e:
        logger.error("[evolve-eval] Evaluation failed: %s", e, exc_info=True)
        return {"combined_score": 0.0, "error": 1.0}
