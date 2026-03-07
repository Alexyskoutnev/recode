"""Loader for the OR-Bench over-refusal benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import BaseLoader
from src.data.types import BenchmarkType, DatasetInfo, Sample


class ORBenchLoader(BaseLoader):
    """Loads OR-Bench: prompts testing over-refusal of safe requests."""

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/or_bench")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            prompt = row.get("prompt", row.get("question", ""))
            samples.append(
                Sample(
                    id=f"or_bench_{idx}",
                    benchmark="or_bench",
                    benchmark_type=BenchmarkType.SAFETY,
                    prompt=str(prompt),
                    reference=row.get("label", row.get("is_safe", "")),
                    metadata={
                        "category": row.get("category", ""),
                        "reject_label": row.get("reject_label", ""),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="OR-Bench",
            benchmark_type=BenchmarkType.SAFETY,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["test"],
        )

    def _read_data(self) -> pd.DataFrame:
        for ext in ["*.parquet", "*.csv", "*.json", "*.jsonl"]:
            files = list(self._data_dir.rglob(ext))
            if files:
                path = files[0]
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                elif path.suffix == ".csv":
                    return pd.read_csv(path)
                elif path.suffix in (".json", ".jsonl"):
                    return pd.read_json(path, lines=path.suffix == ".jsonl")
        raise FileNotFoundError(
            f"No data files found in {self._data_dir}. "
            "Run the download script first."
        )
