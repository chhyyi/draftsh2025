"""package's test module

Test main features, assert reproduction, comparing with snapshot

main features:
    * load test version of in-house dataset
    * featurize compositional features
    * load XuDataset (see comparison.XuDataset for details)

    $ python -m unittest test.py
"""

import unittest
import importlib.resources as resources

import pandas as pd
import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.metrics import r2_score

from draftsh.dataset import Dataset
from draftsh.feature import Featurizer
from draftsh.comparison import XuTestHEA
from draftsh.utils.test_utils import specific_value, compare_as_dataframe

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
        dataset_snapshot_df = pd.read_json(data_dir.joinpath("snapshot_dataset.json"), orient="table")

        # test featurized dataset
        featurizer = Featurizer(config=r"xu.json")
        featurized_np = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
        npz_loaded = np.load(data_dir.joinpath("snapshot_featurized.npz"), allow_pickle=False)
        featurized_snapshot = [npz_loaded["x_train"], npz_loaded["y_train"], npz_loaded["x_test"], npz_loaded["y_test"]]

        # compare specific data, analyzing
        specific_value(live_data=featurized_np, snapshot=featurized_snapshot, featurizer=featurizer)
        specific_value(live_data=featurized_np, snapshot=featurized_snapshot, featurizer=featurizer, stats=["ap_mean"], specific_idx=(0,7,227))

#        print(compare_as_dataframe(live_arrays=featurized_np, snapshot_arrays=featurized_snapshot,
#                             dataset=dataset, featurizer=featurizer,
#                             assert_every_features_in_common="mae_20250904"))
        
        # compare values with snapshot
        pd.testing.assert_frame_equal(dataset_snapshot_df, dataset.dataframe.drop(columns=["comps_pymatgen"]), check_exact=False)
        for i in range(4):
            print(f"test_almost_equal snapshot[{i}] and live[{i}]")
            assert_almost_equal(featurized_snapshot[i], featurized_np[i])

        #xu_val_r2score: test XuDataset reproduces the snapshot
        xu_dataset = XuTestHEA()
        r2score = r2_score(xu_dataset.dataframe["Experimental_T_c(K)"], xu_dataset.dataframe["Predicted_T_c(K)"])
        print(f"r2_score:{r2score}")
        self.assertAlmostEqual(r2score, 0.9246, places=4)
        
if __name__ == '__main__':
    unittest.main()
    
