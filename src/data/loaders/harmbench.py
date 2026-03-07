"""Loader for the HarmBench safety benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import BaseLoader
from src.data.types import BenchmarkType, DatasetInfo, Sample


class HarmBenchLoader(BaseLoader):
    """Loads HarmBench: 510 behaviors testing red-teaming robustness."""

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/harmbench")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            prompt = row.get("Behavior", row.get("behavior", row.get("prompt", "")))
            samples.append(
                Sample(
                    id=row.get("BehaviorID", f"harmbench_{idx}"),
                    benchmark="harmbench",
                    benchmark_type=BenchmarkType.SAFETY,
                    prompt=str(prompt),
                    reference="",
                    metadata={
                        "category": row.get("SemanticCategory", row.get("semantic_category", "")),
                        "functional_category": row.get("FunctionalCategory", row.get("functional_category", "")),
                        "tags": row.get("Tags", ""),
                        "context": row.get("ContextString", ""),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="HarmBench",
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
