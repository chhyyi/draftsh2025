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

from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

from draftsh.dataset import Dataset
from draftsh.feature import Featurizer, MyElementProperty
from draftsh.comparison import XuTestHEA
# checkout dataset.dataframe.loc[7], 'w_NpValence_mean'
# which is source of featurized_snapshot[0][7, 28] and featurized_np[0][7, 28]

def specific_value(live_data: list[np.array], snapshot: list[np.array], featurizer: Featurizer, comp: str = "Ru0.075Rh0.075Pd0.075Ir0.075Pt0.70Sb", specific_idx: tuple[int] = (0, 7, 28), source: str = "magpie", feature: list[str] = ["NpValence"], stats: list[str] = ["mean"], weight:str = "w"):
    assert weight=="w", weight
    prop = MyElementProperty(data_source=source, features=feature, stats=stats)
    comp = Composition(comp)
    elements, fractions = zip(*comp.element_composition.items())

    # find column idx
    found_col = 0
    for idx, feat_col in enumerate(featurizer.col_names["matminer_expanded"]):
        if feat_col == f'{weight}_{feature[0]}_{stats[0]}':
            found_col = found_col+1
            print(idx)
            if idx!=specific_idx[2]:
                print(f"fixing specific_idx from: {specific_idx[2]} to: {idx}")
                specific_idx = (specific_idx[0], specific_idx[1], idx)
    if found_col == 1:
        pass
    elif found_col >1:
        raise KeyError(found_col)
    else:
        raise ValueError(found_col)
                
    
    all_attributes = []
    for attr in prop.features:
        elem_data = [prop.data_source.get_elemental_property(e, attr) for e in elements]

        for stat in prop.stats:
            all_attributes.append(prop.pstats.calc_stat(elem_data, stat, fractions))

    print(f"from attributes of ElementProperty class: {all_attributes}")
    print(f"from attributes of ElementProperty.featurize() method:{prop.featurize(comp)}")
    print(f"elements: {comp}, elem_data: {elem_data}, fractions:{fractions}")
    print(f"from featurized_snapshot: {snapshot[specific_idx[0]][*specific_idx[1:]]}")
    print(f"from featurized_np (live): {live_data[specific_idx[0]][*specific_idx[1:]]}")

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
        featurizer = Featurizer(config=r"xu.json")
        featurized_np = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
        npz_loaded = np.load(data_dir.joinpath("snapshot_featurized.npz"), allow_pickle=False)
        featurized_snapshot = [npz_loaded["x_train"], npz_loaded["y_train"], npz_loaded["x_test"], npz_loaded["y_test"]]
        for i in range(4):
            assert_almost_equal(featurized_snapshot[i], featurized_np[i])
        
           
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
                        # print(f"invalid abs_mean at i, j, k: ({i}, {j}, {k})")
                        # assert "Valence" in featurizer.col_names["matminer_expanded"][k], (featurizer.col_names["matminer_expanded"][k])
                        errors[i][j, k] = 0
                        divide_by_zero_indices.append((i, j, k))
                    else:
                        errors[i][j, k] = difference/abs_mean
        
        # compare as a dataframe - calculate error, lists high-error features/insts
        col_names_input = featurizer.col_names["matminer_expanded"] + featurizer.col_names["xu_eight"]
        col_names_target = ['avg_Tc', 'std_Tc']

        col_indices = []
        feature_type_col = []
        error_col = []
        high_error_features = []
        for i in range(4):
            # switch col_names between input/target
            if i%2==0:
                col_names = col_names_input
            elif i%2==1:
                col_names = col_names_target
            else:
                raise ValueError(i)
            # flatten errors
            a = pd.DataFrame(data = errors[i], columns=col_names)
            for idx, row in a.iterrows():
                for col_idx, col in enumerate(col_names):
                    col_indices.append(idx)
                    feature_type_col.append(col_idx)
                    error_col.append(row[col])

            assert len(col_indices)==len(feature_type_col)==len(error_col)

            # use pd.DataFrame for sort.
            flat_df = pd.DataFrame(data={"instance":col_indices, "feature_type": feature_type_col, "error": error_col})
            #flat_sorted_df["error"] = error_col

            #find largest error:
            largest_error = flat_df.sort_values(by="error").iloc[-1]
            print(largest_error)
            dataset_idx = largest_error["instance"]
            if i>1: #for X_test or Y_test
                dataset_idx = dataset_idx + len(featurized_np[0])
            
            print(f"largest error found for instance: {dataset.dataframe.loc[dataset_idx]["composition"]}, feature_type: {col_names_input[int(largest_error["feature_type"])]}")

            # capture high-error features
            flat_sorted_df = flat_df.drop(columns="instance").sort_values(by="error").reset_index(drop=False)
            
            high_error_ordered = flat_sorted_df["feature_type"].tail(10).value_counts()
            high_error_features += list(high_error_ordered.reset_index(drop=False)["feature_type"][:2])
        # compare specific data..
        specific_value(live_data=featurized_np, snapshot=featurized_snapshot, featurizer=featurizer)
        specific_value(live_data=featurized_np, snapshot=featurized_snapshot, featurizer=featurizer, stats=["ap_mean"], specific_idx=(0,7,227))

        #xu_val_r2score: test XuDataset reproduces the snapshot
        xu_dataset = XuTestHEA()
        r2score = r2_score(xu_dataset.dataframe["Experimental_T_c(K)"], xu_dataset.dataframe["Predicted_T_c(K)"])
        print(f"r2_score:{r2score}")
        self.assertAlmostEqual(r2score, 0.9246, places=4)
        
if __name__ == '__main__':
    unittest.main()
    
