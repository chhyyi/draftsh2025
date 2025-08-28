"""package's test module

Test main features, assert reproduction, comparing with snapshot

main features:
    * load test version of in-house dataset
    * featurize compositional features
    * load XuDataset (see comparison.XuDataset for details)

    $ python -m unittest test.py
"""

import unittest
from pathlib import Path
import importlib.resources as resources

import pandas as pd
import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.metrics import r2_score

from draftsh.dataset import Dataset
from draftsh.feature import Featurizer
from draftsh.comparison import XuDataset

class Test(unittest.TestCase):
    """Test core class methods"""

    def test_dataset(self):
        """in-house dataset features test
        
        load test version of in-house dataset and featurize, 
        assert reproducing snapshot
        """
        with resources.as_file(resources.files("draftsh.data.tests") /"dummy") as path:
            data_dir = path.parent #test data dir 
        dataset = Dataset(data_dir.joinpath("test_dataset.xlsx"), config="default.json")
        dataset_snapshot = np.load(data_dir.joinpath("snapshot_dataset.npz"), allow_pickle=True)
        dataset_df = pd.DataFrame(data = dataset_snapshot["data"], columns = dataset_snapshot["columns"])
        col_dtypes = dataset_snapshot["col_dtypes"] # skip index column
        for col, dtype in zip(dataset_df.columns, col_dtypes):
            dataset_df[col] = dataset_df[col].astype(dtype=dtype)
        pd.testing.assert_frame_equal(dataset_df, dataset.dataframe, check_exact=False)

        # test featurized dataset
        featurizer = Featurizer(config=r"test.json")
        featurized_np = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
        npz_loaded = np.load(data_dir.joinpath("snapshot_featurized.npz"))
        featurized_snapshot = [npz_loaded["X_train"], npz_loaded["Y_train"], npz_loaded["X_test"], npz_loaded["Y_test"]]
        for i in range(4):
            assert_almost_equal(featurized_snapshot[i], featurized_np[i])

    #def xu_val_r2score(self):
    #    """test XuDataset reproduces the snapshot"""
        xu_dataset = XuDataset()
        r2score = r2_score(xu_dataset.dataframe["Experimental_T_c(K)"], xu_dataset.dataframe["Predicted_T_c(K)"])
        print(f"r2_score:{r2score}")
        self.assertAlmostEqual(r2score, 0.9246, places=4)

if __name__ == '__main__':
    unittest.main()
    # temproal_test for split and init, test output generetions
