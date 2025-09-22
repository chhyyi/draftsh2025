"""utility functions 

functions:
    * config_parser()
"""

import json, math
from pathlib import Path
from typing import Literal, Iterator
import importlib.resources as resources
from itertools import product

import pandas as pd
import numpy as np

def config_parser(config: str | dict | Path, mode: Literal["dataset", "feature", "convert"]):
    """
    load and parse config
    """
    if isinstance(config, str):
        config = Path(config)
    else:
        pass

    if isinstance(config, Path):
        if config.is_absolute():
            xls_path = config
        else:
            with resources.as_file(resources.files(f"draftsh.config").joinpath(mode).joinpath(config.name)) as path:
                xls_path = path
                assert xls_path.is_file(), FileNotFoundError(xls_path)
        fp = open(xls_path, "r", encoding = "utf-8")
        assert Path.is_file(xls_path)
        config = json.load(fp)
        fp.close()
    
    assert isinstance(config, dict), f"config: {config}"
    return config

    
def merge_dfs(src_df: pd.DataFrame, featurized_df: pd.DataFrame, reset_index=True):
    """
    merge dfs with same length"""
    assert reset_index, NotImplementedError(reset_index)

    shape_df = np.shape(featurized_df)
    shape_src_df = np.shape(src_df)
    featurized_df = featurized_df.reset_index(drop=True).join(src_df.reset_index(drop=True))

    assert featurized_df.shape[1] == shape_df[1]+shape_src_df[1]
    return featurized_df

class ConfigSingleSource():
    """For a single run of featurize_dataframe() of `draftsh.feature.MyElementalProperty()` class.

    args:
        * config_dic: dict
        * iter_vars: list[str]
            * config_dic[iter_vars[idx]]: list[str] for valid idx.
            * should follow the loop - hierarchy of featurizer like `matminer.featurizers.composition.composite.Elementproperty().featurize()`
        * non_iter_vars: list[str].
            * return(tuple) starts with [config_dict[v] for v in non_iter_vars] before iter_vars.
        
    method:
        * iter_config: return iterator
            `return product(*self.iterate_lists)`
    
    config_dic example:
        {
            "src": str,
            "unweighted": list["uw" | "w"],
            "feature": list[str],
            "stat": list[str]
        }
    """
    def __init__(self, config_dict: dict, non_iter_vars:list[str] = ["src"], iter_vars:list[str] = ["feature", "stat"]):
        self.config = config_dict
        self.iterate_lists = [[config_dict[v]] for v in non_iter_vars]
        self.iterate_lists = self.iterate_lists + [self.config[var] for var in iter_vars]
        
        self.shape = map(len, self.iterate_lists)
        self.len = math.prod(self.shape)

    def __len__(self):
        return self.len
    
    def iter_config(self) -> Iterator[tuple[str, ...]]:
        return product(*self.iterate_lists)
    
if __name__ == "__main__":
    testdict={
        "src": "src",
        "unweighted": "ABC",
        "feature": [1,2,3],
        "stat": "abc"
    }
    a = ConfigSingleSource(testdict)
    for i in a.iter_config():
        print(i)