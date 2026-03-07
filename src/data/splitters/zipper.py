"""Zipper schedule splitter for GDPval (Algorithm 1 from ReCodeAgent)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.data.types import Sample


@dataclass(frozen=True)
class ZipperSplit:
    """Result of the zipper split assignment."""

    dev_slices: dict[str, list[Sample]]  # S1..S6, 22 tasks each.
    test_vault: list[Sample]             # 88 tasks, used once.
    occupation_groups: dict[str, list[str]]  # "odd" / "even" -> occupations.


def zipper_split(samples: list[Sample], seed: int = 42) -> ZipperSplit:
    """Split GDPval into 6 disjoint dev slices + 88 test vault.

    Algorithm (from ReCodeAgent paper):
    1. Group samples by occupation (44 occupations, 5 tasks each).
    2. Sort occupations alphabetically.
    3. Assign occupations to odd (indices 0,2,4,...) and even (1,3,5,...) groups.
    4. For each occupation, sort tasks deterministically.
    5. Reserve 2 tasks per occupation for the test vault.
    6. Assign remaining 3 tasks per occupation to slices using labels A, B, C:
       - Odd group:  A->S1, B->S3, C->S5
       - Even group: A->S2, B->S4, C->S6

    Args:
        samples: List of GDPval samples (must be exactly 220).
        seed: Random seed (unused — split is deterministic by occupation sort).

    Returns:
        ZipperSplit with dev_slices (S1..S6) and test_vault.
    """
    # Group by occupation.
    by_occupation: dict[str, list[Sample]] = {}
    for s in samples:
        occ = s.metadata.get("occupation", "unknown")
        by_occupation.setdefault(occ, []).append(s)

    occupations = sorted(by_occupation.keys())
    if len(occupations) != 44:
        raise ValueError(
            f"Expected 44 occupations, got {len(occupations)}. "
            "Is this the full GDPval dataset?"
        )

    # Sort tasks within each occupation by task_id for determinism.
    for occ in occupations:
        by_occupation[occ].sort(key=lambda s: s.id)

    # Odd/even groups.
    odd_occs = [occupations[i] for i in range(0, len(occupations), 2)]
    even_occs = [occupations[i] for i in range(1, len(occupations), 2)]

    dev_slices: dict[str, list[Sample]] = {f"S{i}": [] for i in range(1, 7)}
    test_vault: list[Sample] = []

    # Assign for odd group.
    odd_slice_map = {"A": "S1", "B": "S3", "C": "S5"}
    for occ in odd_occs:
        tasks = by_occupation[occ]
        _assign_occupation_tasks(tasks, odd_slice_map, dev_slices, test_vault)

    # Assign for even group.
    even_slice_map = {"A": "S2", "B": "S4", "C": "S6"}
    for occ in even_occs:
        tasks = by_occupation[occ]
        _assign_occupation_tasks(tasks, even_slice_map, dev_slices, test_vault)

    # Validate.
    total_dev = sum(len(s) for s in dev_slices.values())
    assert total_dev == 132, f"Expected 132 dev tasks, got {total_dev}"
    assert len(test_vault) == 88, f"Expected 88 test vault tasks, got {len(test_vault)}"
    for slice_name, slice_samples in dev_slices.items():
        assert len(slice_samples) == 22, (
            f"{slice_name} has {len(slice_samples)} tasks, expected 22"
        )

    return ZipperSplit(
        dev_slices=dev_slices,
        test_vault=test_vault,
        occupation_groups={"odd": odd_occs, "even": even_occs},
    )


def _assign_occupation_tasks(
    tasks: list[Sample],
    slice_map: dict[str, str],
    dev_slices: dict[str, list[Sample]],
    test_vault: list[Sample],
) -> None:
    """Assign 5 tasks from one occupation: 3 dev (A/B/C) + 2 test vault."""
    if len(tasks) != 5:
        raise ValueError(f"Expected 5 tasks per occupation, got {len(tasks)}")
    labels = ["A", "B", "C"]
    for i, label in enumerate(labels):
        dev_slices[slice_map[label]].append(tasks[i])
    test_vault.append(tasks[3])
    test_vault.append(tasks[4])


def save_split(split: ZipperSplit, output_path: Path) -> None:
    """Save the zipper split assignment to JSON for reproducibility."""
    data = {
        "dev_slices": {
            name: [s.id for s in samples]
            for name, samples in split.dev_slices.items()
        },
        "test_vault": [s.id for s in split.test_vault],
        "occupation_groups": split.occupation_groups,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
