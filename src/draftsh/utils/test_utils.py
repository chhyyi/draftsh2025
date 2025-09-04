"""utility functions for package test

functions:
    * specific_value():
        test specific values, for `repo_root/tests/test.py`
    * compare_as_dataframe():
        test error of dataframe, for `repo_root/tests/test.py`
"""

import pandas as pd
import numpy as np

from pymatgen.core.composition import Composition

from draftsh.dataset import Dataset
from draftsh.feature import Featurizer, MyElementProperty

def specific_value(
        live_data: list[np.array], snapshot: list[np.array], featurizer: Featurizer,
        comp: str = "Ru0.075Rh0.075Pd0.075Ir0.075Pt0.70Sb",
        specific_idx: tuple[int] = (0, 7, 28),
        source: str = "magpie", feature: list[str] = ["NpValence"],
        stats: list[str] = ["mean"], weight:str = "w",
        fixing_columns: bool = True,
        col_type: str = "matminer_expanded"
        ):
    """
    checkout specific data by specific_idx
    
    examples: 
        * snapshot[0][7, 28] was the 'w_NpValence_mean' of 7th instance before `b65a887` (2025 SEP 3)
    """
    
    # which is source of featurized_snapshot[0][7, 28] and featurized_np[0][7, 28]
    assert weight=="w", weight
    prop = MyElementProperty(data_source=source, features=feature, stats=stats)
    comp = Composition(comp)
    elements, fractions = zip(*comp.element_composition.items())

    # find column idx
    found_col = 0
    if fixing_columns:
        if col_type == "matminer_expanded":
            for idx, feat_col in enumerate(featurizer.col_names[col_type]):
                if feat_col == f'{weight}_{feature[0]}_{stats[0]}':
                    found_col = found_col+1
                    print(idx)
                    if idx!=specific_idx[2]:
                        print(f"fixing specific_idx from: {specific_idx[2]} to: {idx} (col_name: {featurizer.col_names[col_type][idx]})")
                        specific_idx = (specific_idx[0], specific_idx[1], idx)
            if found_col == 1:
                pass
            elif found_col >1:
                raise KeyError(found_col)
            else:
                raise ValueError(found_col)
        else:
            raise NotImplementedError(col_type)
    else:
        pass
                
    
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

def compare_as_dataframe(
        live_arrays: list[np.array], snapshot_arrays: list[np.array],
        dataset: Dataset, featurizer: Featurizer,
        assert_every_features_in_common: str | None = None
        ) -> list[str]:
    """compare as dataframe

    analyze relative error of live and snapshot. 

    return:
        * high_error_features: extract 2 feature types
            which generates larger error,
            respectedly from (X_train, Y_train, X_test, Y_test). (so total 8)
        * for instances,
            `flat_sorted_df_by_instance` can be used to extract
            high_error_instances (not implemented further)

    arguments: 
        * assert_every_features_in_common: see below for.
    
    assert_every_features_in_common: str | None = "mae_20250904"
        * What it does: 
            * `assert all("mae" in col_names[i] for i \
            in flat_sorted_df_by_features[-320:]["feature_type"].values)`
            * sort by error, extract 320 data with higher error
                (data: all the feature values every instances).
                make sure theier col_name have "mae" in common.
        * why I did it: 
            * on 2025 sep 04, 5PM(working on commit `393f3b4`) 
                I found that `mae` was wrong.
                (method mae() of draftsh.feature.MyPropertyStats)
            * after I have fixed it,
                `assert_almost_equal(featurized_snapshot[i], featurized_np[i])`
                showed that 320 was not almost equal.    
    """

    errors = relative_error(live_arrays, snapshot_arrays)
    
    # compare as a dataframe - calculate error, lists high-error features/insts
    col_names_input = featurizer.col_names["matminer_expanded"] + featurizer.col_names["xu_eight"]
    col_names_target = ['avg_Tc', 'std_Tc']

    high_error_features = []

    # concatenate errors into input_erros and target_errors
    input_errors = np.concatenate((errors[0], errors[2]), axis=0)
    target_errors = np.concatenate((errors[1], errors[3]), axis=0)
    for input_or_tar in ["input", "target"]:

        # switch col_names between input/target
        if input_or_tar == "input":
            col_names = col_names_input
            test_errors = input_errors
        elif input_or_tar == "target":
            col_names = col_names_target
            test_errors = target_errors
        else:
            raise ValueError(input_or_tar)
        col_indices = []
        feature_type_col = []
        error_col = []
        # flatten errors
        test_errors_df = pd.DataFrame(data = test_errors, columns=col_names)
        for idx, row in test_errors_df.iterrows():
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
        
        print(f"largest error found for instance: {dataset.dataframe.loc[dataset_idx]["composition"]}, feature_type: {col_names_input[int(largest_error["feature_type"])]}")

        # capture high-error features, high-error instances
        flat_sorted_df_by_features = flat_df.drop(columns="instance").sort_values(by="error").reset_index(drop=False)
        flat_sorted_df_by_instance = flat_df.drop(columns="feature_type").sort_values(by="error").reset_index(drop=False)
        
        high_error_ordered = flat_sorted_df_by_features["feature_type"].tail(10).value_counts()
        high_error_features += list(high_error_ordered.reset_index(drop=False)["feature_type"][:2])

        #specify only a `specific type of columns` generate error: see docstring
        if assert_every_features_in_common is None:
            pass
        elif assert_every_features_in_common == "mae_20250904":
            if input_or_tar == "input": # if input (X_tarin, X_test), test
                 for j in flat_sorted_df_by_features[-320:]["feature_type"].values:
                    assert "mae" in col_names[j], f"col_idx: {j}, col_name: {col_names[j]}"
                #print(flat_sorted_df_by_features[-300:]["feature_type"].value_counts())
                #print(flat_sorted_df_by_instance[-300:]["instance"].value_counts())
            else: # if target(Y_train, Y_test), pass 
                pass
        else:
            raise ValueError(assert_every_features_in_common)

    return [col_names[feature_idx] for feature_idx in high_error_features]

def relative_error(live_arrays: list[np.array], snapshot_arrays: list[np.array], return_zero_division_idx:bool = False) -> list[np.array]:
    assert not return_zero_division_idx, NotImplementedError(return_zero_division_idx)
    errors = []
    divide_by_zero_indices = []
    for i in range(4):
        insts, cols = live_arrays[i].shape
        assert live_arrays[i].shape == snapshot_arrays[i].shape
        errors.append(np.zeros_like(live_arrays[i]))

        for j in range(insts):
            for k in range(cols):
                difference = np.abs(live_arrays[i][j][k] - snapshot_arrays[i][j][k])
                abs_mean = (np.abs(live_arrays[i][j][k])+np.abs(snapshot_arrays[i][j][k]))/2.0
                if abs_mean == 0 and difference != 0:
                    raise ValueError(f"invalid abs_mean, difference: {abs_mean}, {difference} at i, j, k: ({i}, {j}, {k})")
                elif abs_mean == 0 and difference == 0:
                    errors[i][j, k] = 0
                    divide_by_zero_indices.append((i, j, k))
                else:
                    errors[i][j, k] = difference/abs_mean
    return errors