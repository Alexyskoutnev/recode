"""Loader for Google's IFEval benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import BaseLoader
from src.data.types import BenchmarkType, DatasetInfo, Sample


class IFEvalLoader(BaseLoader):
    """Loads IFEval: 541 prompts with verifiable instruction constraints."""

    def __init__(self, data_dir: Path | None = None) -> None:
        default_dir = Path("data/raw/ifeval")
        super().__init__(data_dir or default_dir)

    def load(self) -> list[Sample]:
        df = self._read_data()
        samples = []
        for idx, row in df.iterrows():
            prompt = row.get("prompt", row.get("instruction", ""))
            samples.append(
                Sample(
                    id=f"ifeval_{row.get('key', idx)}",
                    benchmark="ifeval",
                    benchmark_type=BenchmarkType.INSTRUCTION,
                    prompt=str(prompt),
                    reference="",
                    metadata={
                        "instruction_id_list": row.get("instruction_id_list", []),
                        "kwargs": row.get("kwargs", []),
                    },
                )
            )
        return samples

    def info(self) -> DatasetInfo:
        df = self._read_data()
        return DatasetInfo(
            name="IFEval",
            benchmark_type=BenchmarkType.INSTRUCTION,
            num_samples=len(df),
            columns=list(df.columns),
            splits_available=["train"],
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
