"""Loader for the GDPval dataset (OpenAI)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import BaseLoader
from src.data.types import BenchmarkType, DatasetInfo, Sample


class GDPvalLoader(BaseLoader):
    """Loads GDPval: 220 real-world professional tasks across 44 occupations."""

    EXPECTED_TASKS = 220
    TASKS_PER_OCCUPATION = 5

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/gdpval")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_parquet()
        samples = []
        for _, row in df.iterrows():
            samples.append(
                Sample(
                    id=row["task_id"],
                    benchmark="gdpval",
                    benchmark_type=BenchmarkType.ECONOMIC,
                    prompt=row["prompt"],
                    reference=row.get("rubric_pretty", ""),
                    metadata={
                        "sector": row.get("sector", ""),
                        "occupation": row.get("occupation", ""),
                        "rubric_json": row.get("rubric_json", ""),
                        "reference_files": row.get("reference_files", []),
                        "deliverable_files": row.get("deliverable_files", []),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_parquet()
        return DatasetInfo(
            name="GDPval",
            benchmark_type=BenchmarkType.ECONOMIC,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["full"],
        )

    def _read_parquet(self) -> pd.DataFrame:
        parquet_files = list(self._data_dir.glob("*.parquet"))
        if parquet_files:
            return pd.read_parquet(parquet_files[0])
        # Fall back to reading all parquets in train/ subdirectory.
        train_dir = self._data_dir / "train"
        if train_dir.exists():
            parquet_files = list(train_dir.glob("*.parquet"))
            if parquet_files:
                return pd.read_parquet(parquet_files[0])
        raise FileNotFoundError(
            f"No parquet files found in {self._data_dir}. "
            "Run the download script first."
        )

    def occupations(self) -> list[str]:
        """Return sorted list of unique occupations."""
        df = self._read_parquet()
        return sorted(df["occupation"].unique().tolist())

    def sectors(self) -> list[str]:
        """Return sorted list of unique sectors."""
        df = self._read_parquet()
        return sorted(df["sector"].unique().tolist())
