#!/usr/bin/env python3
"""Demo: Load datasets and sample from the unified interface.

Usage:
    python scripts/demo_sampler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.registry import DatasetRegistry
from src.data.sampler import UnifiedSampler
from src.data.types import BenchmarkType


def main() -> None:
    print("=" * 60)
    print("DATASET REGISTRY — Loading available benchmarks")
    print("=" * 60)

    registry = DatasetRegistry(data_root=Path("data/raw"))
    available = registry.load_available()

    if not available:
        print("\nNo datasets found. Run download first:")
        print("  python scripts/download_datasets.py")
        return

    print(f"\nLoaded {len(available)} datasets: {available}")

    # Print info for each dataset.
    for name, info in registry.info().items():
        print(f"\n  {info.name}:")
        print(f"    Type:    {info.benchmark_type.value}")
        print(f"    Samples: {info.num_samples}")
        print(f"    Columns: {info.columns}")

    # Create sampler.
    print("\n" + "=" * 60)
    print("UNIFIED SAMPLER — Sampling from benchmarks")
    print("=" * 60)

    sampler = UnifiedSampler(registry, seed=42)
    print(f"\nStats: {sampler.stats()}")
    print(f"By type: {sampler.stats_by_type()}")

    # Sample from each available benchmark.
    for name in available:
        samples = sampler.sample(name, n=3)
        print(f"\n--- {name} (3 samples) ---")
        for s in samples:
            prompt_preview = s.prompt[:100].replace("\n", " ")
            print(f"  [{s.id}] {prompt_preview}...")

    # Sample by type.
    for btype in BenchmarkType:
        try:
            samples = sampler.sample_by_type(btype, n=2)
            print(f"\n--- {btype.value} (2 samples) ---")
            for s in samples:
                prompt_preview = s.prompt[:80].replace("\n", " ")
                print(f"  [{s.benchmark}:{s.id}] {prompt_preview}...")
        except KeyError:
            print(f"\n--- {btype.value}: no data loaded ---")

    # Mixed batch.
    if len(available) >= 2:
        print(f"\n--- Mixed batch (10 samples, uniform weights) ---")
        mixed = sampler.sample_mixed(n=10)
        for s in mixed:
            prompt_preview = s.prompt[:60].replace("\n", " ")
            print(f"  [{s.benchmark}] {prompt_preview}...")


if __name__ == "__main__":
    main()
