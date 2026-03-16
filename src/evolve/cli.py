"""CLI argument parsing and logging setup."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Agent model presets per provider (--seed + --tier)
AGENT_MODELS = {
    "gemini": {"fast": "gemini-2.5-flash", "slow": "gemini-2.5-pro"},
    "openai": {"fast": "gpt-5-mini", "slow": "gpt-5.4"},
    "anthropic": {"fast": "claude-haiku-4-5", "slow": "claude-opus-4-6"},
}

# Mutation model presets (the LLM that rewrites agent code)
MUTATION_MODELS = {
    "fast": "gemini/gemini-2.5-flash",
    "slow": "gemini/gemini-2.5-pro",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolve agent code across GDPval slices")

    # Which zipper slices to evolve across. GDPval has 8 dev slices (S1-S8, 22
    # tasks each) and 2 eval slices (E1-E2). The loop processes them in order:
    # evolve on S1, carry best code to S2, evolve again, etc. Each slice uses
    # fresh unseen tasks so improvements must generalize.
    p.add_argument("--slices", nargs="+", default=["S1", "S2", "S3"],
                    help="Zipper slices (default: S1 S2 S3)")

    # How many code mutations SkyDiscover tries per slice. Each iteration =
    # one LLM call to generate a mutation + one evaluation round. More
    # iterations = more candidates explored. 5 is quick, 10-20 for serious runs.
    p.add_argument("--iterations", type=int, default=5,
                    help="Iterations per slice (default: 5)")

    # How many tasks each code variant is scored on. Sampled randomly from the
    # slice's 22 tasks. Lower = faster but noisier, higher = reliable but slow.
    p.add_argument("--sample-size", type=int, default=3,
                    help="Tasks per evaluation (default: 3)")

    # Which provider SDK the seed agent uses. Determines the API the evolved
    # agent will call — an OpenAI seed stays an OpenAI agent throughout.
    p.add_argument("--seed", default="openai", choices=["gemini", "openai", "anthropic"],
                    help="Starting agent provider (default: openai)")

    # Which model tier the agent uses to run tasks (fast = cheap, slow = frontier).
    # This is the model INSIDE the agent, not the mutation model.
    p.add_argument("--tier", default="fast", choices=["fast", "slow"],
                    help="Agent model tier (default: fast)")

    # The LLM that rewrites the agent's source code. Separate from the agent
    # model (--seed + --tier). SkyDiscover sends current code to this model
    # and asks for improved versions. Must be gemini/ prefixed (SkyDiscover
    # only supports Gemini for mutations).
    p.add_argument("--mutation-model", default=None,
                    help="Mutation model: 'fast', 'slow', or provider/model (default: fast)")

    # How SkyDiscover explores the space of code variants.
    p.add_argument("--search", default="adaevolve",
                    choices=["adaevolve", "topk", "evox", "beam_search", "best_of_n"],
                    help="Search strategy (default: adaevolve)")

    # Run the best code on ALL 22 tasks after each slice for a reliable score.
    # Without this, scores come only from the sampled evaluations. ~2x runtime.
    p.add_argument("--full-eval", action="store_true",
                    help="Full 22-task eval after each slice")

    # Override the LLM judge that scores agent outputs against task rubrics.
    p.add_argument("--judge-model", default=None,
                    help="Judge model override")

    # Safety weight in the composite fitness function. Controls the
    # balance between task completion (GDPval) and safety (AgentHarm + ToolEmu).
    #   0.0 = Track A: GDPval only, evolution is blind to safety
    #   0.5 = Track B: 50% GDPval + 50% safety, evolution maintains safety
    p.add_argument("--safety-weight", type=float, default=0.0,
                    help="Safety weight in fitness: 0.0 (GDPval only) to 1.0 (safety only) (default: 0.0)")

    # How many safety tasks (AgentHarm + ToolEmu) to sample per evaluation.
    # Only used when --safety-weight > 0.
    p.add_argument("--safety-samples", type=int, default=10,
                    help="Safety tasks per evaluation (default: 10)")

    p.add_argument("--output-dir", default=None,
                    help="Output directory")
    p.add_argument("--verbose", action="store_true",
                    help="Debug logging")

    args = p.parse_args()

    if args.mutation_model is None:
        args.mutation_model = MUTATION_MODELS["fast"]
    elif args.mutation_model in ("fast", "slow"):
        args.mutation_model = MUTATION_MODELS[args.mutation_model]

    args.agent_model = AGENT_MODELS[args.seed][args.tier]
    return args


def setup_logging(verbose: bool, log_file: Path) -> None:
    """Configure dual logging: console + file."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

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
