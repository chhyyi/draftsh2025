"""utility functions 

functions:
    * config_parser()
"""

import json
from pathlib import Path
from typing import Literal, Iterator
import importlib.resources as resources
from itertools import product

def config_parser(config: str | dict | Path, mode: Literal["dataset", "feature"]):
    """
    load and parse config
    """
    if isinstance(config, str):
        config = Path(config)
    else:
        pass

    if isinstance(config, Path):
        if config.is_absolute():
            pass
        else:
            with resources.as_file(resources.files(f"draftsh.config.{mode}") / config) as path:
                xls_path = path
                assert xls_path.is_file(), FileNotFoundError(xls_path)
        fp = open(xls_path, "r", encoding = "utf-8")
        assert Path.is_file(xls_path)
        config = json.load(fp)
        fp.close()
    
    assert isinstance(config, dict), f"config: {config}"
    return config

class ConfigDictSingle():
    """For a single run of featurize_dataframe() of `draftsh.feature.MyElementalProperty()` class.

    args:
        * iter_vars: should follow the loop - hierarchy of `matminer.featurizers.composition.composite.Elementproperty().featurize()`

    method:
        * iter_config: return iterator
    config_dic:
        {
            "src": str,
            "unweighted": list["uw" | "w"],
            "feature": list[str],
            "stat": list[str]
        }
    """
    def __init__(self, config_dict: dict, non_iter_vars = ["src"], iter_vars = ["unweighted", "feature", "stat"]):
        self.config = config_dict
        self.iterate_lists = [[v] for v in non_iter_vars]
        self.iterate_lists = self.iterate_lists + [self.config[var] for var in iter_vars]
        
        self.shape = map(len, self.iterate_lists)
        self.len = 1
        self.len = [self.len*ll for ll in self.shape]

    def __len__(self):
        return self.len
    
    def iter_config(self) -> Iterator[tuple[str, str, str, str]]:
        return product(*self.iterate_lists)
    
if __name__ == "__main__":
    testdict={
        "src": "src",
        "unweighted": "ABC",
        "feature": [1,2,3],
        "stat": "abc"
    }
    a = ConfigDictSingle(testdict)
    for i in a.iter_config():
        print(i)