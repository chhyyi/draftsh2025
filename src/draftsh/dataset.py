"""load in-house dataset

Todo
    * refactor DLDataset
"""

import json
from pathlib import Path
from abc import ABC, abstractmethod
import importlib.resources as resources
import string

import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from sklearn.model_selection import train_test_split
from matminer.featurizers.composition import ElementProperty

from draftsh.parsers import CellParser, FracParser, ElemParser
from draftsh.utils.utils import config_parser
from draftsh.feature import MultiSourceFeaturizer
from draftsh.utils.conversion_utils import process_targets, almost_equals_pymatgen_atomic_fraction


#from matminer.featurizers.composition import composite

class BaseDataset(ABC):
    """Base Class for Dataset classes, 
    
        * drop_cols: drop these columns
    """
    def __init__(
            self, data_path: Path | str | None,
            drop_cols: list[str] | None = None,
            exception_col: str | list[str] | None = "Exceptions"):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols
        self.dset_path: Path = data_path
        self.dataframe: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    def parse_col(self, col: str, cell_parser: CellParser, to_list: bool, data_type: type = str):
        for idx, row in self.dataframe.iterrows():
            self.dataframe.at[idx, col] = cell_parser.parse(row[col])
        if to_list:
            raise NotImplementedError
            #self.dataframe[col].astype(data_type)

    def elemental_stats(self):
        """set of elements in the dataset"""
        elems=set()
        for _, row in self.dataframe.iterrows():
            elems.update(row["elements"])
        assert elems.issubset(set([el.symbol for el in Element]))
        return elems
    
    def pymatgen_comps(self, inplace = True) -> pd.Series | None:
        comps_pymatgen = self.dataframe.apply(lambda row: Composition(zip(row["elements"], row["elements_fraction"])), axis=1)
        assert len(comps_pymatgen) == len(self.dataframe)
        if inplace:
            self.dataframe["comps_pymatgen"] = comps_pymatgen
            return None
        else:
            return self.dataframe["comps_pymatgen"]
        
    def pymatgen_duplicates(self, save_dir=None, rtol=0.1, comps_pymatgen_col='comps_pymatgen'):
        self.duplicated_comps_group={}
        self.duplicated_comps=set()
        for idx, row in self.dataframe.iterrows():
            if idx not in self.duplicated_comps:
                duplicated_row={}
                for iidx, rrow in self.dataframe[idx:].iterrows():
                    if almost_equals_pymatgen_atomic_fraction(rrow[comps_pymatgen_col], row[comps_pymatgen_col], rtol=rtol):
                        duplicated_row[iidx]=rrow[comps_pymatgen_col]
                if len(duplicated_row)>=2: #including self, so it should be larger than 1.
                    print(f"idx, duplicated_idx: {idx}, {duplicated_row}")
                    self.duplicated_comps.update(list(duplicated_row.keys()))
                    self.duplicated_comps_group[idx]={
                        k: str(v) for k, v in duplicated_row.items()
                        }
        if save_dir is not None:
            with open("duplicates_group.json", 'w', encoding="utf-8") as f:
                json.dump(self.duplicated_comps_group, f, indent=4, ensure_ascii=False)
    
    def validate_by_composition(self):
        allowed = set(string.ascii_letters + string.digits + '.')
        for idx, row in self.dataframe.iterrows():
            if set(row["composition"]) <= allowed:
                comp=Composition(row["composition"])
            else:
                comp=None
            if comp is not None:
                    if idx in list(range(27, 40)):
                        pass
                    else:
                        assert almost_equals_pymatgen_atomic_fraction(row["comps_pymatgen"], comp, rtol=0.001), f"idx: {idx}, comp:{comp}, comps_pymatgen:{row['comps_pymatgen']}"

                    
    def merge_duplicated_comps(self, rule: str):
        pass
        

    def assign_dtypes(self):
        for col in self.dataframe.columns:
            if not self.dataframe.col.dtype in (list, dict, float, str):
                self.dataframe[col] = self.dataframe[col].astype(str)

class XlsxDataset(BaseDataset):
    """Base Class for Dataset classes, load xlsx file
    
    load MS Excel(xlsx) file, generate features and an export array for ML.

    Parameters:
        * xls_path: path to the the xlsx file
        * notebook: Excel sheet name to load dataset

    See Also(inherited by):
        - Dataset()
        - XuDataset(): Load Xu 2025 HEAs used to validate.
    """
    def __init__(
            self, xls_path: Path | str,
            notebook: str,
            drop_cols: list[str] | None = None,
            exception_col: str | list[str] | None = "Exceptions"):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols
        if isinstance(xls_path, str):
            xls_path = Path(xls_path)
        if xls_path.is_absolute():
            pass
        else:
            with resources.as_file(resources.files("draftsh.data").joinpath(xls_path)) as pth:
                xls_path = pth
            assert xls_path.is_file(),FileNotFoundError

        super().__init__(xls_path, drop_cols = drop_cols)
        self.sheet = notebook
        self.maxlen: int | None = None

        df = self.load_data()
        if exception_col is not None:
            df = df[df[exception_col].apply(pd.isna)]
        else:
            pass
        self.dataframe: pd.DataFrame = df.reset_index(drop=True)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.dset_path,
                           sheet_name=self.sheet,
                           nrows=self.maxlen)
        return df.drop(columns=self.drop_cols)

class Dataset(XlsxDataset):
    """
    - methods
        - __init__
            - load sc_hea_dataset (xlsx file).
            - SC-HEA Dataset
                - LiteratureReview/datatable/SC_HEA_dataset_CY_JH_temp.xlsx
    - attributes
        - exception_col: simple filter by specific column values
    """
    def __init__(self, 
                 xls_path, 
                 config: str | dict | Path = "default",
                 drop_cols: list[str] | None = None,
                 exception_col: str | list[str] | None = "Exceptions"):
        self.config = config_parser(config, mode="dataset")
        super().__init__(
            xls_path, 
            notebook = self.config["sheetname"], 
            drop_cols = self.config.get("drop_cols", drop_cols), 
            exception_col=self.config.get("exception_col", exception_col))
        self.parse_elements_col()
        self.parse_frac_col()
        self.validate_elem_frac_length()
        self.elemental_set: set = self.elemental_stats()
        self.pymatgen_comps()

    def validate_elem_frac_length(self):
        assert self.dataframe.apply(
            lambda x: (len(x["elements"])==len(x["elements_fraction"])), axis=1).all(),\
            self.dataframe.loc[self.dataframe.apply(
            lambda x: len(x["elements"])!=len(x["elements_fraction"]), axis=1)]

    def parse_elements_col(self, colname: str="elements"):
        """parse string of xlsx cell with elements in list form"""
        cell_parser = ElemParser()
        return self.parse_col(colname, cell_parser, False)

    def parse_frac_col(self, colname: str="elements_fraction"):
        """parse string of xlsx cell with fractions in list form"""
        cell_parser = FracParser()
        return self.parse_col(colname, cell_parser, False)
    
    def featurize_and_split(self,
                            featurizer: MultiSourceFeaturizer, test_size: float = 0.2,
                            shuffle: bool = True, seed: int = 42,
                            to_numpy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """process target and features(input), split.

        arguments:
        """
        target_df = process_targets(self.dataframe, targets = self.config["targets"])
        featurized_df = featurizer.featurize_all(self.dataframe)
        # featurized_df and target_df
        featurized_df = featurized_df.reset_index(drop=True)
        target_df = target_df.reset_index(drop=True)
        assert len(featurized_df) == len(target_df)

        #shuffle, split.
        N = len(featurized_df)
        tr_idx, te_idx = train_test_split(
            np.arange(N),
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle,
        )
        if to_numpy:
            return (featurized_df.loc[tr_idx].to_numpy(),
                    target_df.loc[tr_idx].to_numpy(),
                    featurized_df.loc[te_idx].to_numpy(),
                    target_df.loc[te_idx].to_numpy())
        else:
            return (featurized_df.loc[tr_idx],
                    target_df.loc[tr_idx],
                    featurized_df.loc[te_idx],
                    target_df.loc[te_idx])

class DLDataset(BaseDataset):
    """
    mostly copy of XlsxDataset and Dataset. should be refactored
    """
    def __init__(self, data_path: Path | str, drop_cols: list[str] | None = None):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.is_absolute(), f"data_path: {data_path} should be absolute"
        super().__init__(data_path, drop_cols, exception_col = None)
        df = self.load_data()
        self.dataframe: pd.DataFrame = df.reset_index(drop=True)
        self.parse_elements_col()
        self.parse_frac_col()
        self.elemental_set: set = self.elemental_stats()        
        self.pymatgen_comps()


    def load_data(self) -> pd.DataFrame:
        return pd.read_excel(self.dset_path, index_col=0) 

    def parse_elements_col(self, colname: str="elements"):
        """parse string of xlsx cell with elements in list form"""
        cell_parser = ElemParser()
        return self.parse_col(colname, cell_parser, False)

    def parse_frac_col(self, colname: str="elements_fraction"):
        """parse string of xlsx cell with fractions in list form"""
        cell_parser = FracParser()
        return self.parse_col(colname, cell_parser, False)
    
    def featurize_and_split(self, test_size: float = 0.2, shuffle: bool = True, seed: int = 42, to_numpy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        config = {"targets": ["max_Tc"]}
        target_df = process_targets(self.dataframe, targets = config["targets"], exception_row=None)
        featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
        featurized_df = featurizer.featurize_dataframe(self.dataframe, col_id='comps_pymatgen', ignore_errors=True)
        
        # featurized_df and target_df
        featurized_df = featurized_df.reset_index(drop=True)
        target_df = target_df.reset_index(drop=True)
        assert len(featurized_df) == len(target_df)

        #shuffle, split.
        N = len(featurized_df)
        tr_idx, te_idx = train_test_split(
            np.arange(N),
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle,
        )
        if to_numpy:
            return (featurized_df.loc[tr_idx].to_numpy(),
                    target_df.loc[tr_idx].to_numpy(),
                    featurized_df.loc[te_idx].to_numpy(),
                    target_df.loc[te_idx].to_numpy())
        else:
            return (featurized_df.loc[tr_idx],
                    target_df.loc[tr_idx],
                    featurized_df.loc[te_idx],
                    target_df.loc[te_idx])