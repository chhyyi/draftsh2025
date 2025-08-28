"""utility functions 

config_parser
"""

import json
from pathlib import Path
from typing import Literal
import importlib.resources as resources

import numpy as np

def config_parser(config: str | dict | Path, mode: Literal["dataset", "feature"]):
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
        fp = open(xls_path, "r")
        assert Path.is_file(xls_path)
        config = json.load(fp)
        fp.close()
    
    assert isinstance(config, dict), f"config: {config}"
    return config