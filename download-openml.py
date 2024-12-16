from __future__ import annotations

import pandas as pd
import openml
import warnings
import argparse
import zipfile
from pathlib import Path
from typing import Iterator

import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
ROOT = FILE.parent
OPENML_CACHE_DIR = ROOT / ".openml_cache"
DEFAULT_DATADIR = ROOT / "data"


def download_exam_dataset(datadir: Path) -> None:
    # The `datadir` is the directory where the data should be downloaded to.
    import urllib.request
    import shutil

    URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-24-tabular/exam_dataset.zip"

    path_to_put = datadir / "exam_dataset" / "1"
    path_to_put.mkdir(parents=True, exist_ok=True)

    zipfile_path = path_to_put / "exam_dataset.zip"
    if not zipfile_path.exists():
        _urlopen = urllib.request.urlopen
        with _urlopen(URL) as response, zipfile_path.open("wb") as f:
            shutil.copyfileobj(response, f)

    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(path_to_put)

    print(f"Downloaded and extracted exam dataset to {path_to_put}!")


def openml_task_folds(
    openml_task_id: int,
    *,
    openml_cache_directory: Path = OPENML_CACHE_DIR,
) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    logging.info(f"Getting task {openml_task_id}")
    openml.config.set_root_cache_directory(openml_cache_directory)

    # NOTE: There's a FutureWarning which we can't avoid here -_-
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        task = openml.tasks.get_task(
            openml_task_id,
            download_splits=True,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )

        logging.info(f"Done getting task {openml_task_id}")
        _, folds, _ = task.get_split_dimensions()
        for fold in range(folds):
            train_idx, test_idx = task.get_train_test_split_indices(fold=fold)
            X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            assert isinstance(X_train, pd.DataFrame)
            cat_features = X_train.select_dtypes(
                include=["category", bool, object],
            ).columns.tolist()
            X_train[cat_features] = X_train[cat_features].astype("category")
            X_test[cat_features] = X_test[cat_features].astype("category")
            yield X_train, y_train, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--datadir", type=Path, default=DEFAULT_DATADIR)
    parser.add_argument("--openml-cache-dir", type=Path, default=OPENML_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    if args.task == "exam_dataset":
        download_exam_dataset(args.datadir)
        exit(0)

    task_dir: Path = args.datadir / str(args.task)
    if not task_dir.exists():
        task_dir.mkdir(parents=True)
    else:
        if not args.overwrite:
            print(f"Task {args.task} already exists in {task_dir}")
            exit(0)

    for fold, (X_train, y_train, X_test, y_test) in enumerate(
        openml_task_folds(
            int(args.task),
            openml_cache_directory=args.openml_cache_dir,
        ),
        start=1,
    ):
        fold_dir = task_dir / str(fold)
        if not fold_dir.exists():
            fold_dir.mkdir(parents=True)
        logger.info(f"Writing data for {fold} to {fold_dir}")

        X_train.to_parquet(fold_dir / "X_train.parquet")
        y_train.to_frame().to_parquet(fold_dir / "y_train.parquet")
        X_test.to_parquet(fold_dir / "X_test.parquet")
        y_test.to_frame().to_parquet(fold_dir / "y_test.parquet")

    print(f"Downloaded data for task {args.task} to {task_dir}")
