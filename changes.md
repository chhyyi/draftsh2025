# Changes 
## since `f394951` (Sep 1, 2025)
* regenerated snapshot; fixed `draftsh.parsers.FracParser()`, not to leave `list[str]`.
* merged `xu_eight` features as a matminer inherited classes.

## since `0d1d5d9` (Sep 4, 2025)
### Code changes:  
* mae: duplicate of PropertyStats.avg_dev so removed.
    * `mae` <-> `avd_dev`: old `mae` normalizes by sum of weights.
        * mae
            ```python
            @staticmethod
            def mae(data_lst, weights = None):
                mae = 0.0
                mean = PropertyStats.mean(data_lst, weights)
                if weights != None:
                    for v, w in zip(data_lst, weights):
                        mae = mae + w*np.abs(v-mean)
                    mae = mae / np.sum(weights)
                elif weights == None:
                    for v in data_lst:
                        mae = mae + np.abs(v-mean)
                    mae = mae / len(data_lst)
                return mae
            ```
        * `PropertyStats.avg_dev`
            ```python
            @staticmethod
            def avg_dev(data_lst, weights=None):
                mean = PropertyStats.mean(data_lst, weights)
                return np.average(np.abs(np.subtract(data_lst, mean)), weights=weights)
            ```
* `src\draftsh\config\feature\xu.json`: replacing mae to avg_dev, found a mistake put avg_dev instead of std_dev..
* ap error: found a lot..
    * `MyPropertyStats.all_ap()` 
        * np.mean(da+db) -> np.mean(da, db)
        * all_ap() now returns tuple of two lists (APs, weights) or (APs, None)
### validation ([tests/test.py](tests/test.py)):
* specific values(2): for 7th instance, now generating correct `w_NpValence_ap_mean`
    now it gives correct answer, `1.6080402010050252`
    ```bash
    fixing specific_idx from: 227 to: 164 (col_name: w_NpValence_ap_mean)
    from attributes of ElementProperty class: [np.float64(1.6080402010050252)]
    from attributes of ElementProperty.featurize() method:[np.float64(1.6080402010050252)]
    elements: Ru0.075 Rh0.075 Pd0.075 Ir0.075 Pt0.7 Sb1, elem_data: [0.0, 0.0, 0.0, 0.0, 0.0, 3.0], fractions:(0.075, 0.075, 0.075, 0.075, 0.7, 1.0)
    from featurized_snapshot: 0.05360134003350084
    from featurized_np (live): 1.6080402010050252
    ```
* 1979(X_train), 622(X_test) mistmatched elements, compared with previous snapshot.
    * assert_almost_equal(featurized_snapshot[i], featurized_np[i]) gives (1979, 0, 622, 0) mistmatched elements.
    * possible numbers of mismatches by code changes
        * ap_set: all features call `all_aps()`: 4 (ap_mean, ap_minimum, ap_maximum, ap_range) * 16 (features; 14 magpie features, pymatgen.thermal_conductivity, bccfermi) * 2 (weighted, unweighted) * 16 = 2048
            * include `ap_mean`: 18+1+1(magpie, paymatgen.thermal_conductivity, bccfermi), 2(weight) * 21 = 840
        * avg_dev_fix_set: in `xu.json`, corrected `avg_dev` to `std_dev`: 18(feature) * 21(instance_num) = 378
            * magpie, weighted, ("mean", "std_dev", "avg_dev"), 
        * avg_dev_old_mae_set: if weighted, `mae` normalized(divide by sum(weights)) but `avg_dev` does not: 20 * 21 = 420
    * validation in `compare_as_dataframe` with `assert_every_features_in_common == "20250905s"`:
        * from the extracted set of `high_error_features`,
        * assert `len(high_errors_feature_set - ap_set - avg_dev_fix_set - avg_dev_old_mae_set)==0`
            * ap_set: any name include "ap"
            * avg_dev_fix_set
            * avg_dev_old_mae_set: was previously `mae`.

## Since `51b7091` (Sep 2, 2025)
* **Fixed test error claimed on all unweighted features were invalid!**