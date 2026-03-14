"""CLI argument parsing and logging setup."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# ── Model presets per provider ──────────────────────────────────────────────
# fast = cheap, good enough for quick trials and evolution iterations
# slow = frontier, best quality for serious runs and final evals
#
# These are the models the SEED AGENT uses to run tasks.
# The --mutation-model flag controls the separate LLM that writes code mutations.

MODELS = {
    "gemini": {
        "fast": "gemini-2.5-flash",        # fast, cheap, pro-grade reasoning
        "slow": "gemini-2.5-pro",          # frontier, best quality
    },
    "openai": {
        "fast": "gpt-5-mini",             # fast, cost-efficient
        "slow": "gpt-5.4",               # frontier, 1M context
    },
    "anthropic": {
        "fast": "claude-haiku-4-5",       # fast, near-frontier quality
        "slow": "claude-opus-4-6",        # frontier, best reasoning
    },
}

# Default mutation LLMs (the model that rewrites agent code, not the agent itself)
# SkyDiscover uses LiteLLM — format is "provider/model" (e.g. gemini/, openai/)
EVOLVE_MODELS = {
    "fast": "gemini/gemini-2.5-flash",     # cheap mutations, more iterations
    "slow": "gemini/gemini-2.5-pro",       # smarter mutations, fewer needed
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolve agent code across GDPval slices")

    # Which task groups to evolve across. GDPval's 220 tasks are split into
    # 8 dev slices (S1-S8, 22 tasks each). The loop processes them in order —
    # evolve on S1, carry best code to S2, evolve again, etc. Each slice uses
    # fresh unseen tasks so improvements must generalize.
    p.add_argument("--slices", nargs="+", default=["S1", "S2", "S3"],
                    help="Zipper slices (default: S1 S2 S3)")

    # How hard SkyDiscover searches per slice. Each iteration = one LLM call
    # to generate a code mutation + one evaluation (running tasks). More
    # iterations = more candidates explored = higher chance of improvement.
    # 5 is quick (~5 min/slice), 10-20 for serious runs.
    p.add_argument("--iterations", type=int, default=5,
                    help="SkyDiscover iterations per slice (default: 5)")

    # Tasks per evaluation. Each code variant is scored on this many randomly
    # sampled tasks from the current slice (22 available). Lower = faster but
    # noisier (bad sample can make good code look bad). Higher = reliable
    # signal but slower. 3 is the sweet spot for trials; 5+ for real runs.
    p.add_argument("--sample-size", type=int, default=3,
                    help="Tasks per evaluation (default: 3)")

    # Starting agent code. Each seed is a self-contained Python file with
    # 8 tools and a ReAct loop using a specific LLM provider SDK. The seed
    # determines which API the evolved agent calls — a Gemini seed stays
    # a Gemini agent. Needs the corresponding API key in .env.
    p.add_argument("--seed", default="openai", choices=["gemini", "openai", "anthropic"],
                    help="Starting agent (default: openai)")

    # Which tier of model the seed agent uses to run tasks.
    #   fast = cheap, good enough (gemini-3-flash / gpt-5-mini / claude-haiku-4-5)
    #   slow = frontier, best quality (gemini-3.1-pro / gpt-5.4 / claude-opus-4-6)
    # This is the model INSIDE the agent — the one that actually does the work.
    p.add_argument("--tier", default="fast", choices=["fast", "slow"],
                    help="Agent model tier: fast (cheap) or slow (frontier) (default: fast)")

    # The mutation model — the LLM that rewrites/improves the agent's source
    # code. NOT the agent model that runs tasks (that's --seed + --tier).
    # SkyDiscover sends the current agent code to this model and asks for
    # improved versions. Default: gemini-2.5-flash.
    p.add_argument("--mutation-model", default=None,
                    help="Mutation model (rewrites agent code): 'fast', 'slow', or specific model (default: fast)")

    # How SkyDiscover explores the space of code variants.
    #   adaevolve   — multi-island adaptive, UCB bandit, migration (default)
    #   topk        — keep best K, mutate from top (simple, fast)
    #   evox        — co-evolves code AND search strategy (expensive)
    #   beam_search — depth-first refinement with fixed-width beam
    #   best_of_n   — generate N variants, keep single best (baseline)
    # See docs/search_strategies.md for detailed visuals.
    p.add_argument("--search", default="adaevolve",
                    choices=["adaevolve", "topk", "evox", "beam_search", "best_of_n"],
                    help="Search strategy (default: adaevolve)")

    # After each slice's evolution, run the best code on ALL 22 tasks for a
    # reliable score. Without this, scores come only from SkyDiscover's
    # sample-based evaluations (fast but noisy). Roughly doubles runtime.
    p.add_argument("--full-eval", action="store_true",
                    help="Full 22-task eval after each slice")

    # Override the LLM judge model used for scoring. Evolution always uses the
    # LLM judge (not keyword heuristic) so the fitness signal is accurate.
    p.add_argument("--judge-model", default=None, help="Judge model override")

    # Where results go. Default: results/evolve_<timestamp>. Contains
    # trajectory.json, evolved code per slice, SkyDiscover checkpoints,
    # and evolve.log.
    p.add_argument("--output-dir", default=None, help="Output directory")

    # Shows DEBUG-level messages from all modules: SkyDiscover internals
    # (island selection, UCB scores, population stats), API calls, tool
    # execution. Useful for debugging; noisy for normal runs.
    p.add_argument("--verbose", action="store_true", help="Debug logging")

    args = p.parse_args()

    # Resolve --mutation-model shorthand
    if args.mutation_model is None:
        args.mutation_model = EVOLVE_MODELS["fast"]
    elif args.mutation_model in ("fast", "slow"):
        args.mutation_model = EVOLVE_MODELS[args.mutation_model]

    # Resolve agent model from --seed + --tier
    args.agent_model = MODELS[args.seed][args.tier]

    return args


def setup_logging(verbose: bool, log_file: Path) -> None:
    """Configure dual logging: console (clean) + file (detailed)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console: timestamp + level + message
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # File: full detail with module names for post-mortem
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Quiet noisy libraries
    for name in ("httpx", "httpcore", "urllib3", "google", "openai", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
