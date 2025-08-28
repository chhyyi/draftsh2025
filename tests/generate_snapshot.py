"""generate snapshot for test

hard coded
"""
from pathlib import Path
import importlib.resources as resources

import pandas as pd
import numpy as np

from draftsh.dataset import Dataset
from draftsh.feature import Featurizer


def generate_snapshot(data_dir: Path):
    dataset = Dataset(data_dir.joinpath("test_dataset.xlsx"), config="default.json")
    np.savez(data_dir.joinpath("snapshot_dataset.npz"), data = dataset.dataframe, columns = dataset.dataframe.columns, col_dtypes = dataset.dataframe.dtypes, allow_pickle=True)

    # dataset.dataframe is pandas DataFrame
    assert isinstance(dataset.dataframe, pd.DataFrame)

    # get featurized dataset
    featurizer = Featurizer(config=r"test.json")
    X_train, Y_train, X_test, Y_test = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
    np.savez(data_dir.joinpath("snapshot_featurized.npz"), X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)


if __name__ == "__main__":
    with resources.as_file(resources.files("draftsh.data.tests") /"dummy") as path:
        tests_data_dir = path.parent
    generate_snapshot(tests_data_dir)