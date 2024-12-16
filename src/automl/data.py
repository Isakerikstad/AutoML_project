from __future__ import annotations

import pandas as pd
from pathlib import Path
from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


DATASET_MAP = dict(
    y_prop="361092",
    bike_sharing="361099",
    brazilian_houses="361098",
    exam_dataset="exam_dataset",
)


@dataclass
class Dataset:
    path: Path
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series | None = None

    @classmethod
    def load(cls, datadir: Path, task: str, fold: int) -> Dataset:
        if task == "exam_dataset":
            path = datadir / task / str(1)
        else:
            path = datadir / DATASET_MAP[task] / str(fold)
        if not path.exists():
            raise FileNotFoundError(path)

        X_train_path = path / "X_train.parquet"
        y_train_path = path / "y_train.parquet"
        X_test_path = path / "X_test.parquet"
        y_test_path = path / "y_test.parquet"

        return Dataset(
            path=path,
            X_train=pd.read_parquet(X_train_path),
            y_train=pd.read_parquet(y_train_path).iloc[:, 0],
            X_test=pd.read_parquet(X_test_path),
            y_test=pd.read_parquet(y_test_path).iloc[:, 0]
            if y_test_path.exists()
            else None,
        )
