"""SkyDiscover evaluator bridge — the fitness function for code evolution.

This file is the critical bridge between SkyDiscover (the evolutionary search
framework) and our GDPval evaluation pipeline. SkyDiscover doesn't know anything
about coding agents or task scoring — it just knows how to evolve text (code)
and needs a fitness function to tell it how good each variant is.

That fitness function is ``evaluate(program_path)``. SkyDiscover calls it for
every code variant it generates. The flow looks like this:

    SkyDiscover                          This file (evaluator.py)
    ──────────                           ────────────────────────
    1. Generate mutated code        →
    2. Write to temp file           →
    3. Call evaluate(temp_file)      →    4. Read the code from disk
                                         5. compile() check — catch syntax errors
                                         6. importlib load — find the agent class
                                         7. Instantiate the agent
                                         8. Run it on N sampled GDPval tasks
                                         9. LLM judge scores each task output
    10. Receive {"combined_score"}   ←   10. Return avg score as fitness

WHY THIS FILE EXISTS:
    SkyDiscover imports this file independently (not as part of our package).
    It looks for a module-level ``evaluate(program_path)`` function. That's
    the only interface contract. Everything else is internal.

WHY ENV VARS:
    Because SkyDiscover imports this file in its own process, we can't pass
    config as function arguments. The parent process (run_evolve.py) sets
    env vars before launching SkyDiscover, and we read them here.

WHY sys.path MANIPULATION:
    SkyDiscover imports this file standalone, not via ``python -m src.evolve``.
    Without the sys.path hack, ``from src.data.registry import ...`` would fail
    because the project root isn't on the path.

RETURN VALUE CONTRACT:
    Must return a dict with at least ``combined_score`` (float, 0.0-1.0).
    SkyDiscover uses this as the fitness signal. Additional keys are stored
    in the iteration stats but don't affect selection.

    On any failure (syntax error, import crash, runtime error), we return
    combined_score=0.0 so the variant is naturally eliminated by selection
    pressure — no special error handling needed in SkyDiscover.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

# SkyDiscover imports this file standalone, so we need the project root
# on sys.path to resolve ``from src.data import ...`` etc.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("evolve.eval")

# Cache loaded GDPval samples so we don't re-parse the dataset on every
# evaluation call. SkyDiscover calls evaluate() many times per run.
_cached_samples: dict[str, list] = {}
_load_counter = 0   # unique module names for importlib
_eval_counter = 0   # tracks evaluations for log prefixes


# ── Configuration ─────────────────────────────────────────────────────────

def _get_config() -> dict:
    """Read evaluation config from environment variables.

    These are set by run_evolve.py before launching SkyDiscover:
      EVOLVE_SLICE        — which zipper slice to sample tasks from (e.g. "S1")
      EVOLVE_SAMPLE_SIZE  — how many tasks to run per evaluation (e.g. 3)
      EVOLVE_WORKING_DIR  — where agents write their output files
      EVOLVE_AGENT_MODEL  — model override for the agent (e.g. "gpt-5.4")
      EVOLVE_JUDGE_MODEL  — model override for the LLM judge
    """
    return {
        "slice": os.environ.get("EVOLVE_SLICE", "S1"),
        "sample_size": int(os.environ.get("EVOLVE_SAMPLE_SIZE", "5")),
        "working_dir": os.environ.get("EVOLVE_WORKING_DIR", "results/evolve_workspace"),
        "agent_model": os.environ.get("EVOLVE_AGENT_MODEL"),
        "judge_model": os.environ.get("EVOLVE_JUDGE_MODEL"),
    }


# ── Data loading ──────────────────────────────────────────────────────────

def _load_zipper_slice(slice_name: str) -> list[str]:
    """Load task IDs for a zipper slice from the pre-computed split file."""
    split_path = _PROJECT_ROOT / "data" / "processed" / "zipper_split.json"
    with open(split_path) as f:
        splits = json.load(f)
    all_slices = {**splits.get("dev_slices", {}), **splits.get("eval_slices", {})}
    if slice_name not in all_slices:
        raise KeyError(f"Slice '{slice_name}' not found. Available: {list(all_slices.keys())}")
    return all_slices[slice_name]


def _load_samples(slice_name: str):
    """Load and cache GDPval samples for a slice.

    First call loads the full dataset and filters to the slice's task IDs.
    Subsequent calls return the cached result (samples don't change mid-run).
    """
    if slice_name in _cached_samples:
        return _cached_samples[slice_name]

    from src.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    registry.load_dataset("gdpval")
    all_samples = registry.get_samples("gdpval")

    id_set = set(_load_zipper_slice(slice_name))
    samples = [s for s in all_samples if s.id in id_set]
    _cached_samples[slice_name] = samples
    return samples


# ── Agent loading ─────────────────────────────────────────────────────────

def _load_agent_from_code(code: str, model: str | None = None):
    """Dynamically load an agent class from evolved source code.

    The evolved code is a self-contained Python file — it defines its own
    imports, data classes, tools, and agent class. We can't rely on inheritance
    from our project's BaseAgent because the evolution might change the class
    hierarchy. Instead we use duck-typing:

    1. Write the code to a temp file (importlib needs a real file)
    2. Load it as a Python module via importlib
    3. Scan all classes in the module for one that has both ``run()`` and
       ``name()`` methods and isn't abstract
    4. Instantiate it (with optional model override)
    5. Clean up the temp file and module registration

    Returns an agent instance, or None on any failure.
    """
    global _load_counter
    _load_counter += 1
    module_name = f"evolved_agent_{_load_counter}"

    try:
        # importlib.util.spec_from_file_location needs a real file on disk
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="evolved_", delete=False,
            dir=str(_PROJECT_ROOT),
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            if spec is None or spec.loader is None:
                logger.error("Cannot create module spec from evolved code")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Duck-typing scan: find any concrete class with run() + name()
            # that was defined in this module (not imported from elsewhere)
            agent_cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and hasattr(attr, "run") and hasattr(attr, "name")
                        and attr.__module__ == module_name
                        and not inspect.isabstract(attr)):
                    agent_cls = attr
                    break

            if agent_cls is None:
                logger.error("No agent class found in evolved code")
                return None

            # Try passing model override; fall back to no-arg if the evolved
            # code changed the constructor signature
            try:
                return agent_cls(model=model) if model else agent_cls()
            except TypeError:
                return agent_cls()

        finally:
            # Always clean up — don't leave temp files or pollute sys.modules
            os.unlink(temp_path)
            sys.modules.pop(module_name, None)

    except Exception as e:
        logger.error("Failed to load agent: %s\n%s", e, traceback.format_exc())
        return None


# ── Evaluation pipeline ──────────────────────────────────────────────────

def _run_eval(code: str) -> dict[str, float]:
    """Score an evolved agent on a sample of GDPval tasks.

    This is the core evaluation pipeline:

    Step 1 — COMPILE CHECK:
        Fast-fail on syntax errors. Broken code scores 0 and gets eliminated
        by SkyDiscover's selection pressure. No need for special handling.

    Step 2 — LOAD AGENT:
        Dynamically import the code and find the agent class via duck-typing.
        If loading fails (import error, missing class), score 0.

    Step 3 — SAMPLE TASKS:
        Pick ``sample_size`` random tasks from the current slice. Using a
        random sample (vs. fixed) means different evaluations test different
        tasks, reducing overfitting to specific tasks.

    Step 4 — RUN + JUDGE:
        Execute the agent on each task (it calls an LLM API, uses tools,
        creates files). Then an LLM judge scores each output against a
        rubric. The average score across tasks is the fitness.

    Returns dict with combined_score (fitness) + diagnostic metrics.
    """
    from src.eval.runner import GDPvalRunner

    global _eval_counter
    _eval_counter += 1
    eval_id = _eval_counter

    cfg = _get_config()
    slice_name = cfg["slice"]
    sample_size = cfg["sample_size"]

    # Step 1: Compile check — catch syntax errors before attempting import
    try:
        compile(code, "<evolved_agent>", "exec")
    except SyntaxError as e:
        logger.warning("[eval #%d] SYNTAX ERROR: %s", eval_id, e)
        return {"combined_score": 0.0, "syntax_error": 1.0}

    # Step 2: Load the agent class from the evolved code
    agent_model = cfg.get("agent_model")
    logger.info("[eval #%d] Loading evolved agent (%d lines, model=%s)...",
                eval_id, code.count("\n") + 1, agent_model or "default")
    agent = _load_agent_from_code(code, model=agent_model)
    if agent is None:
        logger.warning("[eval #%d] LOAD FAILED", eval_id)
        return {"combined_score": 0.0, "load_error": 1.0}
    logger.info("[eval #%d] Agent loaded: %s", eval_id, agent.name())

    # Step 3: Sample tasks from the current slice
    samples = _load_samples(slice_name)
    eval_samples = random.sample(samples, min(sample_size, len(samples)))
    task_ids = [s.id[:8] for s in eval_samples]
    logger.info("[eval #%d] Running %d tasks on %s: %s",
                eval_id, len(eval_samples), slice_name, ", ".join(task_ids))

    # Step 4: Run agent on tasks, score with LLM judge
    t0 = time.monotonic()
    try:
        workspace = Path(cfg["working_dir"]) / f"eval_{slice_name}"

        def _progress(done: int, total: int, trace) -> None:
            score_str = ""
            if trace.eval_result and trace.eval_result.max_score > 0:
                score_str = f" {trace.eval_result.normalized_score:.0%}"
            status = "ok" if trace.error is None else "ERR"
            logger.info("[eval #%d] Task %d/%d %s [%s]%s",
                        eval_id, done, total, trace.task_id[:8], status, score_str)

        runner = GDPvalRunner(
            agent=agent,
            working_dir=workspace,
            use_judge=True,
            judge_model=cfg.get("judge_model"),
        )
        # Run all sampled tasks in parallel — concurrency matches sample count
        # since each eval is small (typically 3-5 tasks)
        result = asyncio.run(
            runner.run_batch(eval_samples, concurrency=len(eval_samples), progress_callback=_progress)
        )

        elapsed = time.monotonic() - t0
        logger.info("[eval #%d] DONE — %s avg=%.1f%% (%d/%d ok, %d err) %.0fs",
                    eval_id, slice_name, result.avg_score * 100,
                    result.num_completed, len(eval_samples), result.num_errors, elapsed)

        return {
            "combined_score": result.avg_score,
            "avg_score": result.avg_score,
            "num_completed": float(result.num_completed),
            "num_errors": float(result.num_errors),
            "num_tasks": float(len(eval_samples)),
        }

    except Exception as e:
        logger.error("[eval #%d] RUNTIME ERROR: %s", eval_id, e, exc_info=True)
        return {"combined_score": 0.0, "runtime_error": 1.0}


# ── SkyDiscover interface ────────────────────────────────────────────────

def evaluate(program_path: str) -> dict[str, float]:
    """SkyDiscover entry point — the fitness function for code evolution.

    SkyDiscover calls this function for every code variant it generates.
    This is the ONLY function SkyDiscover knows about — it's the interface
    contract between the evolutionary search and our evaluation pipeline.

    How SkyDiscover uses this:
        1. SkyDiscover's mutation LLM generates a modified version of the agent
        2. SkyDiscover writes the code to a temp file on disk
        3. SkyDiscover calls evaluate(temp_file_path)
        4. We read the code, load it as a Python module, run tasks, return score
        5. SkyDiscover uses combined_score to decide if this variant survives

    Args:
        program_path: Path to a temp file containing the full evolved agent
                      source code. SkyDiscover manages this file's lifecycle.

    Returns:
        Dict with at least ``combined_score`` (float, 0.0 to 1.0).
        This is the fitness value SkyDiscover uses for selection.
        Additional keys (avg_score, num_completed, etc.) are stored in
        SkyDiscover's iteration stats for analysis but don't affect selection.

        On ANY failure — empty code, syntax error, import crash, runtime
        error — we return combined_score=0.0. This means broken mutations
        are naturally eliminated by selection pressure. No variant gets a
        free pass.
    """
    try:
        code = Path(program_path).read_text()
        if not code.strip():
            return {"combined_score": 0.0, "error": 1.0}
        return _run_eval(code)
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return {"combined_score": 0.0, "error": 1.0}
