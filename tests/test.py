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
from draftsh.comparison import XuTestHEA

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
        pd.testing.assert_frame_equal(dataset_snapshot_df, dataset.dataframe.drop(columns=["comps_pymatgen"]), check_exact=False)

        # test featurized dataset
        featurizer = Featurizer(config=r"test.json")
        featurized_np = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
        npz_loaded = np.load(data_dir.joinpath("snapshot_featurized.npz"), allow_pickle=False)
        featurized_snapshot = [npz_loaded["x_train"], npz_loaded["y_train"], npz_loaded["x_test"], npz_loaded["y_test"]]
        for i in range(4):
            assert_almost_equal(featurized_snapshot[i], featurized_np[i])

    #def xu_val_r2score(self):
    #    """test XuDataset reproduces the snapshot"""
        xu_dataset = XuTestHEA()
        r2score = r2_score(xu_dataset.dataframe["Experimental_T_c(K)"], xu_dataset.dataframe["Predicted_T_c(K)"])
        print(f"r2_score:{r2score}")
        self.assertAlmostEqual(r2score, 0.9246, places=4)

if __name__ == '__main__':
    unittest.main()
    # temproal_test for split and init, test output generetions

#%%

import unittest
from pathlib import Path
import importlib.resources as resources

import pandas as pd
import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.metrics import r2_score

from draftsh.dataset import Dataset
from draftsh.feature import Featurizer
from draftsh.comparison import XuTestHEA
with resources.as_file(resources.files("draftsh.data.tests") /"dummy") as path:
    data_dir = path.parent #test data dir 
dataset = Dataset(data_dir.joinpath("test_dataset.xlsx"), config="default.json")
dataset_snapshot_df = pd.read_json(data_dir.joinpath("snapshot_dataset.json"), orient="table")
pd.testing.assert_frame_equal(dataset_snapshot_df, dataset.dataframe.drop(columns=["comps_pymatgen"]), check_exact=False)

# test featurized dataset
featurizer = Featurizer(config=r"test.json")
featurized_np = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
npz_loaded = np.load(data_dir.joinpath("snapshot_featurized.npz"), allow_pickle=False)
featurized_snapshot = [npz_loaded["x_train"], npz_loaded["y_train"], npz_loaded["x_test"], npz_loaded["y_test"]]

#%% iterate x_train, y_train, t_test, y_test
# compare as dataframe:
errors = []
divide_by_zero_indices = []

for i in range(4):
    insts, cols = featurized_np[i].shape
    assert featurized_np[i].shape == featurized_snapshot[i].shape
    errors.append(np.zeros_like(featurized_np[i]))

    for j in range(insts):
        for k in range(cols):
            difference = np.abs(featurized_np[i][j][k] - featurized_snapshot[i][j][k])
            abs_mean = (np.abs(featurized_np[i][j][k])+np.abs(featurized_snapshot[i][j][k]))/2.0
            if abs_mean == 0 and difference == 0:
                print(f"invalid abs_mean at i, j, k: ({i}, {j}, {k})")
                # assert "Valence" in featurizer.col_names["matminer_expanded"][k], (featurizer.col_names["matminer_expanded"][k])
                errors[i][j, k] = 0
                divide_by_zero_indices.append((i, j, k))
            else:
               errors[i][j, k] = difference/abs_mean
# %%
col_names_input = featurizer.col_names["matminer_expanded"] + featurizer.col_names["xu_eight"]
col_names_target = ['avg_Tc', 'std_Tc']

col_indices = []
feature_type_col = []
error_col = []
for i in range(4):
    # switch col_names between input/target
    if i%2==0:
        col_names = col_names_input
    elif i%2==1:
        col_names = col_names_target
    
    # flatten errors
    a = pd.DataFrame(data = errors[i], columns=col_names)
    for idx, row in a.iterrows():
        for col_idx, col in enumerate(col_names):
            col_indices.append(idx)
            feature_type_col.append(col_idx)
            error_col.append(row[col])

    assert len(col_indices)==len(feature_type_col)==len(error_col)

    # use pd.DataFrame for sort.
    flat_df = pd.DataFrame(index=col_indices, data=feature_type_col, columns=["feature_type"])
    flat_df["error"] = error_col

    flat_df = flat_df.sort_values(by="error")
    print(flat_df.tail(15))
    break
#%% checkout dataset.dataframe.loc[7], 'w_NpValence_mean'
# which is source of featurized_snapshot[0][7, 28] and featurized_np[0][7, 28]

from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

prop = ElementProperty(data_source="magpie", features=["NpValence"], stats=["mean"])
comp = Composition("Ru0.075Rh0.075Pd0.075Ir0.075Pt0.70Sb")
elements, fractions = zip(*comp.element_composition.items())

all_attributes = []
for attr in prop.features:
    elem_data = [prop.data_source.get_elemental_property(e, attr) for e in elements]

    for stat in prop.stats:
        all_attributes.append(prop.pstats.calc_stat(elem_data, stat, fractions))

print(f"from attributes of ElementProperty class: {all_attributes}")
print(f"from attributes of ElementProperty.featurize() method:{prop.featurize(comp)}")
mean_temp = np.sum([elem_data[i]*fractions[i] for i in range(len(elem_data))])/np.sum(fractions)
print(f"from elem_data and fractions:{mean_temp}")
print(f"from featurized_snapshot: {featurized_snapshot[0][7, 28]}")
print(f"from featurized_np (live): {featurized_np[0][7, 28]}")
