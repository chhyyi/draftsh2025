"""load in-house dataset

Todo
    * refactor DLDataset
"""

import json
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import importlib.resources as resources
import string
from typing import Optional, Literal

import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from sklearn.model_selection import train_test_split
from matminer.featurizers.composition import ElementProperty

from draftsh.parsers import CellParser, FracParser, ElemParser
from draftsh.utils.utils import config_parser
from draftsh.feature import MultiSourceFeaturizer
from draftsh.utils.conversion_utils import process_targets, almost_equals_pymatgen_atomic_fraction, norm_fracs


#from matminer.featurizers.composition import composite

class BaseDataset(ABC):
    """Base Class for Dataset classes, 
    
    argument:
        * drop_cols: drop these columns
        * exception_col: column to simply except some rows.
             e.g.) boolean column
    """
    def __init__(
            self, data_path: Optional[Path | str],
            comps_pymatgen_col: str="comps_pymatgen",
            drop_cols: Optional[list[str]] = None,
            exception_col: Optional[str | list[str]] = None,):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols
        self.dset_path: Path = data_path
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.exception_col = exception_col
        self.comps_pymatgen_col = comps_pymatgen_col

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
    
    def pymatgen_comps(self, inplace = True) -> Optional[pd.Series]:
        comps_pymatgen = self.dataframe.apply(lambda row: Composition(zip(row["elements"], row["elements_fraction"])), axis=1)
        assert len(comps_pymatgen) == len(self.dataframe)
        if inplace:
            self.dataframe["comps_pymatgen"] = comps_pymatgen
            return None
        else:
            return self.dataframe["comps_pymatgen"]
        
    def pymatgen_duplicates(self, other_df:Optional[pd.DataFrame]=None, save_dir=None, exception_map:Optional[dict]=None, rtol=0.1, return_dict:bool=True):
        """
        compare elements and elements_fraction, find duplicates. 

        update attributes;
            * `self.duplicated_comps_group`(dictionary)
            * `self.duplicated_comps` (set)
        to compare, elements fractions are always normalized 
            - to meet `sum(fractions)==1.0`

        arguments:
            * other_df: Optional[pd.DataFrame].
                * if None, compare in-between (default)
                * if not None, other_df should have comps_pymatgen_col column().
                    * **note that it will overwrite `self attributes`**
        
        notes:
            * pymatgen.core.Composition().almost_equals() does not normalizes as I wanted, So I couldn't use that.

        arguments 
        other_df:Optional[pd.dataframe]=None
        """
        comps_pymatgen_col = self.comps_pymatgen_col
        self.duplicated_comps_group={}
        self.duplicated_comps=set()
        if other_df is None:
            df1=self.dataframe
        else:
            assert type(self).__name__=='XuTestHEA', "compare with other_df is implemented only for XuTestHEA"
            df1=other_df
        for idx0, row0 in self.dataframe.iterrows():
            if idx0 not in self.duplicated_comps:
                duplicated_row={}
                idx1_start = 0 if other_df is not None else idx0 # when compare self, (i,j)==(j,i)
                row0_compare=row0[comps_pymatgen_col]
                if exception_map is not None:
                    if idx0 in exception_map:
                        row0_compare=df1.loc[exception_map[idx0], comps_pymatgen_col]

                for idx1, row1 in df1[idx1_start:].iterrows():
                    if almost_equals_pymatgen_atomic_fraction(row1[comps_pymatgen_col], row0_compare, rtol=rtol):
                        duplicated_row[idx1]=row1[comps_pymatgen_col]

                for idx1, row1 in df1[idx1_start:].iterrows():
                    if almost_equals_pymatgen_atomic_fraction(row1[comps_pymatgen_col], row0[comps_pymatgen_col], rtol=rtol):
                        duplicated_row[idx1]=row1[comps_pymatgen_col]
                minimum_length = 2 if other_df is None else 1 #when self compare, it includes self.
                if len(duplicated_row)>=minimum_length:
                    print(f"idx0, duplicates: {idx0}, {duplicated_row}")
                    self.duplicated_comps_group[idx0]={
                        k: str(v) for k, v in duplicated_row.items()
                        }
                    if other_df is None:
                        # when other_df, `idx0 not in self.duplicated_comps` should be false always
                        self.duplicated_comps.update(list(duplicated_row.keys()))
                    else:
                        # 
                        self.duplicated_comps_group[idx0][f"self_{idx0}"]=str(row0[comps_pymatgen_col])
                        if exception_map is not None:
                            if idx0 in exception_map:
                                self.duplicated_comps_group[idx0][f"replace_to_nominal_comp_idx0918({exception_map[idx0]})"]=str(row0_compare)

        if save_dir is not None:
            with open(save_dir, 'w', encoding="utf-8") as f:
                json.dump(self.duplicated_comps_group, f, indent=4, ensure_ascii=False)
    
    def validate_by_composition(self, rtol:float=0.001):
        allowed = set(string.ascii_letters + string.digits + '.')
        for idx, row in self.dataframe.iterrows():
            comp=None
            try:
                if set(row["composition"]) <= allowed:
                    comp=Composition(row["composition"])
            except:
                comp=None
            if comp is not None:
                if idx in list(range(27, 40)): # Oxygen not included on composition
                    pass
                elif idx in list(range(88, 93)): #nominal - actual
                    pass
                elif idx in list(range(270, 276)): #x variable
                    pass
                else:
                    if almost_equals_pymatgen_atomic_fraction(row["comps_pymatgen"], comp, rtol=rtol):
                        pass
                    else:
                        assert row["comps_pymatgen"].elements==comp.elements
                        warnings.warn(f"idx: {idx}, comp:{comp}, comps_pymatgen:{row['comps_pymatgen']}")
                        print(f"while norm_fracs(comp):{norm_fracs(comp)}, norm_fracs_comps_pymatgen:{norm_fracs(row['comps_pymatgen'])}")

                    
    def add_duplicated_comps_column(self, criteria_rule: Literal['single_ref'], inplace=True):
        assert criteria_rule in ['single_ref'], NotImplementedError(criteria_rule)
        duplicate_groups=[] # will be a new column, group name = first instance idx(in self.dataset.duplicated_comps_group.keys)
        for idx0, row0 in self.dataframe.iterrows():
            group_row=np.nan
            if idx0 in self.duplicated_comps_group.keys():
                for idx1 in self.duplicated_comps_group[idx0].keys():            
                    cite0=row0["full citation"]
                    cite1=self.dataframe.loc[idx1, "full citation"]
                    group_row = idx0 if cite0==cite1 else np.nan
            duplicate_groups.append(group_row)
        self.dataframe['duplicated_group']=duplicate_groups
        return duplicate_groups
    
    def assign_dtypes(self):
        for col in self.dataframe.columns:
            if not self.dataframe.col.dtype in (list, dict, float, str):
                self.dataframe[col] = self.dataframe[col].astype(str)

class D2TableDataset(BaseDataset):
    """Base Class for Dataset classes, load xlsx file
    
    load csv or MS Excel(xlsx) file, generate features and an export array for ML.

    Parameters:
        * xls_path: path to the the xlsx file
        * notebook: Excel sheet name to load dataset

    See Also(inherited by):
        - Dataset()
        - XuDataset(): Load Xu 2025 HEAs used to validate.
    """
    def __init__(
            self, xls_path: Path | str,
            notebook: str = None,
            drop_cols: Optional[list[str]] = None,
            exception_col: Optional[str | list[str]] = "Exceptions"):
        if drop_cols is None:
            self.drop_cols = []
        else:
            self.drop_cols = drop_cols
        if isinstance(xls_path, str):
            xls_path = Path(xls_path)
        if xls_path.is_absolute():
            pass
        else:
            if xls_path.is_file():
                pass
            else:
                with resources.as_file(resources.files("draftsh.data").joinpath(xls_path)) as pth:
                    xls_path = pth
                assert xls_path.is_file(),FileNotFoundError

        super().__init__(xls_path, drop_cols = drop_cols)
        self.sheet = notebook
        self.maxlen: Optional[int] = None

        df = self.load_data()
        if exception_col is not None:
            df = df[df[exception_col].apply(pd.isna)]
        else:
            pass
        self.dataframe: pd.DataFrame = df.reset_index(drop=True)

    def load_data(self) -> pd.DataFrame:
        if self.dset_path.suffix==".xlsx" or self.dset_path.stem==".xls":
            df = pd.read_excel(self.dset_path,
                            sheet_name=self.sheet,
                            nrows=self.maxlen)
        elif self.dset_path.suffix==".csv":
            df = pd.read_csv(self.dset_path,
                            nrows=self.maxlen)
        else:
            raise TypeError("read only xlsx, xls, csv files(should have suffix).")
        return df.drop(columns=self.drop_cols)

class Dataset(D2TableDataset):
    """
    attributes:
        * exception_col: simple filter by specific column values

    methods:
        * validate_elem_frac_length:
    
    """
    def __init__(self, 
                 xls_path, 
                 config: str | dict | Path = "default",
                 drop_cols: Optional[list[str]] = None,
                 exception_col: Optional[str | list[str]] = "Exceptions"):
        self.config = config_parser(config, mode="dataset")
        super().__init__(
            xls_path, 
            notebook = self.config.get("sheetname"), 
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
    def __init__(self, data_path: Path | str, drop_cols: Optional[list[str]] = None):
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