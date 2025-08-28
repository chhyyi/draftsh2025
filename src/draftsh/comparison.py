"""comparisons with literatures

XuDataset: Xu et al. (2025)

Todo:
    * duplicates; in __init__, `elem_list=[]..`

References
    - Xu, S. Predicting superconducting temperatures with new hierarchical neural network AI model. Front. Phys. 20, 14205 (2025).
"""
from pathlib import Path
import importlib.resources as resources
from urllib.parse import urlparse

import pandas as pd
from pymatgen.core.composition import Composition

from draftsh.dataset import XlsxDataset, BaseDataset

class XuDataset(XlsxDataset):
    """reproduce dataset of Xu et al. 2025."""
    def __init__(self):
        with resources.as_file(resources.files("draftsh.data.miscs") /"xu2025_validation_HEAs.xlsx") as path:
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

class StanevSuperCon(BaseDataset):
    """load processed SuperCon csv from Stanev et al. 2018"""
    def __init__(self, drop_cols = None, exception_col = None):
        super().__init__(data_path=None, drop_cols=drop_cols, exception_col=exception_col)
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
        self.dataframe: pd.DataFrame = self.dataframe.reset_index()
        
    def load_data(self):
        with resources.as_file(resources.files("draftsh.data.miscs") /"Supercon_data.csv") as path:
            self.dataframe = pd.read_csv(path)
        return self.dataframe
        
#%%
