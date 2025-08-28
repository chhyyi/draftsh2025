"""load in-house dataset"""

from pathlib import Path
from abc import ABC, abstractmethod
import importlib.resources as resources

import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from sklearn.model_selection import train_test_split

from draftsh.parsers import CellParser, FracParser, ElemParser, process_targets
from draftsh.utils import config_parser
from draftsh.feature import Featurizer


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
    
    def pymatgen_comps(self) -> pd.Series:
        comps_pymatgen = []
        for _, row in self.dataframe.iterrows():
            assert pd.isna(row["Exceptions"])
            rep_chem_comp=""

            for elem, frac in zip(row["elements"], row["elements_fraction"]):
                rep_chem_comp+=(f"{elem}{frac}")

            test_comp = Composition(rep_chem_comp)
            comps_pymatgen.append(test_comp)
        
        assert len(comps_pymatgen) == len(self.dataframe)
        self.dataframe["comps_pymatgen"] = comps_pymatgen
        return self.dataframe["comps_pymatgen"]

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
            with resources.as_file(resources.files("draftsh.data") / xls_path) as pth:
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
        self.dataframe: pd.DataFrame = df.reset_index()

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.dset_path, sheet_name=self.sheet, nrows=self.maxlen)
        return df.drop(columns=self.drop_cols)

    

class Dataset(XlsxDataset):
    """
    - methods
        - __init__
            - load sc_hea_dataset (xlsx file).
            - SC-HEA Dataset
                - LiteratureReview/datatable/SC_HEA_dataset_CY_JH_temp.xlsx
    - attributes
        - ....
    """
    def __init__(self, xls_path, config: str | dict | Path = "default"):
        self.config = config_parser(config, mode="dataset")
        super().__init__(xls_path, notebook = self.config["sheetname"], drop_cols = self.config["drop_cols"])
        self.parse_elements_col()
        self.parse_frac_col()
        self.elemental_set: set = self.elemental_stats()
        self.pymatgen_comps()


    def parse_elements_col(self, colname: str="elements"):
        """parse string of xlsx cell with elements in list form"""
        cell_parser = ElemParser()
        return self.parse_col(colname, cell_parser, False)

    def parse_frac_col(self, colname: str="elements_fraction"):
        """parse string of xlsx cell with fractions in list form"""
        cell_parser = FracParser()
        return self.parse_col(colname, cell_parser, False)
    
    def featurize_and_split(self, featurizer: Featurizer, test_size: float = 0.2, shuffle: bool = True, seed: int = 42, to_numpy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target_df = process_targets(self.dataframe, targets = self.config["targets"])
        featurized_df = featurizer.featurize(self.dataframe)
        # merge featurized_df and target_df
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
