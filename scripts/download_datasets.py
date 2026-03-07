#!/usr/bin/env python3
"""Download all benchmark datasets from HuggingFace.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --only gdpval truthfulqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset


# Each entry: (local_name, hf_repo, hf_subset, hf_split).
DATASETS: list[tuple[str, str, str | None, str | None]] = [
    ("gdpval", "openai/gdpval", None, None),
    ("truthfulqa", "truthful_qa", "generation", None),
    ("simpleqa", "basicv8vc/SimpleQA", None, None),
    ("ifeval", "google/IFEval", None, None),
    ("harmbench", "huihui-ai/harmbench_behaviors", None, None),
    ("or_bench", "bench-llm/or-bench", "or-bench-hard-1k", None),
]

DATA_ROOT = Path("data/raw")


def download_one(
    name: str,
    hf_repo: str,
    hf_subset: str | None,
    hf_split: str | None,
) -> bool:
    """Download a single dataset and save as parquet.

    Returns True on success, False on failure.
    """
    out_dir = DATA_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({hf_repo})")
    print(f"  Subset: {hf_subset or 'default'}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    try:
        kwargs: dict = {}
        if hf_subset:
            kwargs["name"] = hf_subset
        if hf_split:
            kwargs["split"] = hf_split

        ds = load_dataset(hf_repo, **kwargs)

        # Save each split as a separate parquet file.
        if hasattr(ds, "keys"):
            # DatasetDict — multiple splits.
            for split_name in ds.keys():
                out_path = out_dir / f"{split_name}.parquet"
                ds[split_name].to_parquet(str(out_path))
                print(f"  Saved {split_name}: {len(ds[split_name])} rows -> {out_path}")
        else:
            # Single Dataset.
            out_path = out_dir / "data.parquet"
            ds.to_parquet(str(out_path))
            print(f"  Saved: {len(ds)} rows -> {out_path}")

        print(f"  Columns: {ds.column_names if hasattr(ds, 'column_names') else 'see splits'}")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only download these datasets (by local name).",
    )
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    targets = DATASETS
    if args.only:
        targets = [d for d in DATASETS if d[0] in args.only]
        if not targets:
            print(f"No matching datasets. Available: {[d[0] for d in DATASETS]}")
            sys.exit(1)

    results: dict[str, bool] = {}
    for name, repo, subset, split in targets:
        results[name] = download_one(name, repo, subset, split)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name:20s} {status}")

    failed = [n for n, s in results.items() if not s]
    if failed:
        print(f"\nFailed downloads: {failed}")
        print("You may need to search for alternative dataset names on HuggingFace.")
        sys.exit(1)


if __name__ == "__main__":
    main()
