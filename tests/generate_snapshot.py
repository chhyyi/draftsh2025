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
    """
    generate snapshots for test.
    """
    dataset = Dataset(data_dir.joinpath("test_dataset.xlsx"), config="default.json")
    dataset_snapshot_path = data_dir.joinpath("snapshot_dataset.json")
    print(f"writing dataset snapsho as json: {dataset_snapshot_path}")
    dataset_to_save = dataset.dataframe.drop(columns=["comps_pymatgen"])
    dataset_to_save.to_json(dataset_snapshot_path, orient="table", indent=4, index=None)
    
    # dataset.dataframe is pandas DataFrame
    assert isinstance(dataset.dataframe, pd.DataFrame)

    # get featurized dataset
    featurizer = Featurizer(config=r"xu.json")
    featurized_ds = dataset.featurize_and_split(
        featurizer=featurizer, test_size=0.2,
        shuffle=False, to_numpy=True)
    x_train, y_train, x_test, y_test = featurized_ds
    
    np.savez(data_dir.joinpath("snapshot_featurized.npz"),
             x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test,
             allow_pickle = False)


if __name__ == "__main__":
    with resources.as_file(resources.files("draftsh.data.tests") /"dummy") as path:
        tests_data_dir = path.parent
    generate_snapshot(tests_data_dir)
