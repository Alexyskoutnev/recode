#!/usr/bin/env python3
"""Sample GDPval tasks from the unified interface.

Usage:
    python scripts/sample_gdpval.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.registry import DatasetRegistry
from src.data.sampler import UnifiedSampler


def print_rubric(sample) -> None:
    """Pretty-print the rubric from a GDPval sample."""
    rubric_json = sample.metadata.get("rubric_json")
    if rubric_json:
        items = json.loads(rubric_json) if isinstance(rubric_json, str) else rubric_json
        total = sum(item["score"] for item in items)
        print(f"  Rubric ({len(items)} items, {total} points total):")
        print(f"  {'─' * 76}")
        for i, item in enumerate(items, 1):
            score = item["score"]
            criterion = item["criterion"]
            wrapped = textwrap.fill(
                criterion, width=70, initial_indent="    ", subsequent_indent="    "
            )
            print(f"  {i:2d}. [+{score}] {criterion[:70]}")
            if len(criterion) > 70:
                for line in textwrap.wrap(criterion[70:], width=66):
                    print(f"           {line}")
        print(f"  {'─' * 76}")
        print(f"  Total: {total} points")
    else:
        print(f"  Rubric: {sample.reference}")


def main() -> None:
    registry = DatasetRegistry(data_root=Path("data/raw"))
    available = registry.load_available()

    if "gdpval" not in available:
        print("GDPval not found. Download it first:")
        print("  python scripts/download_datasets.py --only gdpval")
        return

    sampler = UnifiedSampler(registry, seed=42)

    # Sample 1 GDPval task.
    samples = sampler.sample("gdpval", n=1)

    for s in samples:
        print(f"{'=' * 80}")
        print(f"  ID:         {s.id}")
        print(f"  Sector:     {s.metadata.get('sector')}")
        print(f"  Occupation: {s.metadata.get('occupation')}")
        print(f"{'=' * 80}")
        print()
        print("  Prompt:")
        print(f"  {'─' * 76}")
        for line in s.prompt.splitlines():
            print(f"    {line}")
        print(f"  {'─' * 76}")
        print()
        print_rubric(s)
        print()


if __name__ == "__main__":
    main()
