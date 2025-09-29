"""comparisons with literatures

XuDataset: Xu et al. (2025)

Todo:
    * duplicates; in __init__, `elem_list=[]..`

References
    - Xu, S. Predicting superconducting temperatures with new hierarchical neural network AI model. Front. Phys. 20, 14205 (2025).
"""
#%%
from pathlib import Path
import importlib.resources as resources
from urllib.parse import urlparse
from typing import Optional

import pandas as pd
from pymatgen.core.composition import Composition

from draftsh.dataset import D2TableDataset, BaseDataset, Dataset

class XuTestHEA(D2TableDataset):
    """reproduce test HEA set of Xu et al. 2025.
    
    see supple. Table 1
    """
    def __init__(self):
        with resources.as_file(resources.files("draftsh.data.miscs") /"xu2025_validation_HEAs.csv") as path:
            xls_path = path

        super().__init__(xls_path=xls_path, notebook="Sheet1", exception_col=None)
        elem_list=[]
        frac_list=[]
        for _, row in self.dataframe.iterrows():
            comp=Composition(row["formula"])
            elem_list.append(comp.as_data_dict()["elements"])
            frac_list.append(
                [comp.get_atomic_fraction(comp.as_data_dict()["elements"][i])
                 for i in list(range(comp.as_data_dict()["nelements"]))])
        self.dataframe["elements"]=elem_list
        self.dataframe["elements_fraction"]=frac_list
        self.pymatgen_comps()

class StanevSuperCon(BaseDataset):
    """load processed SuperCon csv from Stanev et al. 2018

    preprocess:
        * see `src\draftsh\data\miscs\preprocess_supercon.py`
        * note that Tc_upper_bound is hard coded, because it is close to the 5213 entries of xu et al
    """
    def __init__(self, drop_cols = None, exception_col = None, maxlen: Optional[int] = None):
        super().__init__(data_path=None, drop_cols=drop_cols, exception_col=exception_col)
        self.maxlen = maxlen
        self.load_data()
        elem_list=[]
        frac_list=[]
        drop_rows=[]
        
        for idx, row in self.dataframe.iterrows():
            if row["Tc"]>12:
                drop_rows
            try:
                comp=Composition(row["name"])
                row_fracs =[comp.get_atomic_fraction(comp.as_data_dict()["elements"][i])
                    for i in list(range(comp.as_data_dict()["nelements"]))]
                elem_list.append(comp.as_data_dict()["elements"])
                frac_list.append(row_fracs)
            except:
                drop_rows.append(idx)
        self.dataframe = self.dataframe.drop(index=drop_rows, axis=0)
        self.dataframe["elements"]=elem_list
        self.dataframe["elements_fraction"]=frac_list
        self.dataframe: pd.DataFrame = self.dataframe.reset_index(drop=True)
        
    def load_data(self):
        with resources.as_file(resources.files("draftsh.data.miscs") /"preprocessed_supercon.csv") as path:
            self.dataframe = pd.read_csv(path, nrows = self.maxlen)
        return self.dataframe
        
class XuDataset():
    """Reproduced XuDataset

    1. Data collection and cleaning
    """
    pass

if __name__=="__main__":
    # -----search xu2025 composition formula on merged dataset, check duplicated--------
    #   * does not consider which references it came from
    #   
    xu_test = XuTestHEA()
    dataset = Dataset(xls_path=r"C:\Users\chyi\draftsh2025\temp_devs\merged_dataset_forward.csv", config="default_forward.json", exception_col=None)

    #exception_map, as we currently use "nominal composition" only.
    exception_map={
        7:324,  # (nominal - dendrite_phase) xu(7)->319 in `index_0810`. `sample C5`
        20:326, # (nominal - dendrite_phase) xu(20)->321 in `index_0810``
    }
    xu_test.pymatgen_duplicates(dataset.dataframe, rtol=0.1, save_dir="temp_devs/xu_to_0918_duplicates_rtol0_1.json", exception_map=exception_map)
    for i in range(len(xu_test.dataframe)):
        print(list(xu_test.duplicated_comps_group[i].keys())[:-1])
    # print index_0810    
    for i in range(len(xu_test.dataframe)):
        index_0810s=[]
        for key in xu_test.duplicated_comps_group[i].keys():
            if isinstance(key, int):
                if pd.isna(dataset.dataframe.loc[key, 'index_0810']):
                    index_0810s.append("new_0918")
                else:
                    index_0810s.append(int(dataset.dataframe.loc[key, 'index_0810']))
        print(index_0810s)

    # revert dictionary
    idx0918_to_xu_test={}
    for k, v in xu_test.duplicated_comps_group.items():
        for k1, _ in v.items():
            if isinstance(k1, int):
                idx0918_to_xu_test[k1]=k
            
    for i in range(len(dataset.dataframe)):
        print(idx0918_to_xu_test.get(i, "false"))
    a=[1234,534] #placeholder..