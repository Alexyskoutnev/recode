"""Loader for OpenAI's SimpleQA benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import BaseLoader
from src.data.types import BenchmarkType, DatasetInfo, Sample


class SimpleQALoader(BaseLoader):
    """Loads SimpleQA: 4326 short factual questions testing calibration."""

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/simpleqa")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            # SimpleQA may have different column names depending on source.
            prompt = row.get("problem", row.get("question", row.get("prompt", "")))
            answer = row.get("answer", row.get("reference", ""))
            topic = row.get("metadata", row.get("topic", ""))
            samples.append(
                Sample(
                    id=f"simpleqa_{idx}",
                    benchmark="simpleqa",
                    benchmark_type=BenchmarkType.HALLUCINATION,
                    prompt=str(prompt),
                    reference=str(answer),
                    metadata={"topic": str(topic)},
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="SimpleQA",
            benchmark_type=BenchmarkType.HALLUCINATION,
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
