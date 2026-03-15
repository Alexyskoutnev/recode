"""Zipper schedule splitter for safety benchmarks (AgentHarm + ToolEmu).

Creates deterministic, non-overlapping slices that align with the GDPval
zipper schedule (S1-S8 dev, E1-E2 eval). Each slice gets a balanced mix
of samples from both benchmarks.

AgentHarm: 208 samples across 8 harm categories
ToolEmu:   144 samples across 36 toolkits
Total:     352 safety samples → ~35 per slice (10 slices)

The split is deterministic (sorted by ID, round-robin) so every evolution
run evaluates on the exact same safety tasks per slice.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.types import Sample

NUM_DEV_SLICES = 8
NUM_EVAL_SLICES = 2
NUM_SLICES = NUM_DEV_SLICES + NUM_EVAL_SLICES


def safety_zipper_split(
    agentharm_samples: list[Sample],
    toolemu_samples: list[Sample],
) -> dict[str, dict[str, list[Sample]]]:
    """Split safety samples into 10 slices aligned with GDPval's S1-S8, E1-E2.

    Algorithm:
    1. Sort each benchmark's samples by ID for determinism.
    2. Round-robin assign each benchmark's samples across 10 slices independently.
    3. Merge: each slice gets its share of AgentHarm + ToolEmu samples.

    This ensures:
    - Every slice has samples from both benchmarks
    - Categories are distributed (not clustered in one slice)
    - The split is fully reproducible

    Returns:
        {"dev_slices": {"S1": [...], ...}, "eval_slices": {"E1": [...], ...}}
    """
    def _round_robin(samples: list[Sample]) -> list[list[Sample]]:
        sorted_samples = sorted(samples, key=lambda s: s.id)
        slices: list[list[Sample]] = [[] for _ in range(NUM_SLICES)]
        for i, s in enumerate(sorted_samples):
            slices[i % NUM_SLICES].append(s)
        return slices

    ah_slices = _round_robin(agentharm_samples)
    te_slices = _round_robin(toolemu_samples)

    # Merge: each slice gets AgentHarm + ToolEmu samples
    merged: list[list[Sample]] = []
    for i in range(NUM_SLICES):
        merged.append(ah_slices[i] + te_slices[i])

    dev_slices = {f"S{i+1}": merged[i] for i in range(NUM_DEV_SLICES)}
    eval_slices = {f"E{i+1}": merged[NUM_DEV_SLICES + i] for i in range(NUM_EVAL_SLICES)}

    return {"dev_slices": dev_slices, "eval_slices": eval_slices}


def save_safety_split(
    split: dict[str, dict[str, list[Sample]]],
    output_path: Path,
) -> None:
    """Save the safety zipper split to JSON for reproducibility."""
    data = {
        kind: {
            name: [s.id for s in samples]
            for name, samples in slices.items()
        }
        for kind, slices in split.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
