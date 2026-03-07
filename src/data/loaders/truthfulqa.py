"""Loader for the TruthfulQA benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import BaseLoader
from src.data.types import BenchmarkType, DatasetInfo, Sample


class TruthfulQALoader(BaseLoader):
    """Loads TruthfulQA: 817 questions testing factual accuracy."""

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/truthfulqa")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            samples.append(
                Sample(
                    id=f"truthfulqa_{idx}",
                    benchmark="truthfulqa",
                    benchmark_type=BenchmarkType.HALLUCINATION,
                    prompt=row["question"],
                    reference=row.get("best_answer", ""),
                    metadata={
                        "category": row.get("category", ""),
                        "correct_answers": row.get("correct_answers", []),
                        "incorrect_answers": row.get("incorrect_answers", []),
                        "source": row.get("source", ""),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="TruthfulQA",
            benchmark_type=BenchmarkType.HALLUCINATION,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["validation"],
        )

    def _read_data(self) -> pd.DataFrame:
        # TruthfulQA is typically in the 'validation' split.
        for subdir in ["validation", "generation", ""]:
            search_dir = self._data_dir / subdir if subdir else self._data_dir
            for ext in ["*.parquet", "*.csv", "*.json", "*.jsonl"]:
                files = list(search_dir.glob(ext))
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
