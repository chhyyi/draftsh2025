# Vendored from matminer
#   (https://github.com/hackingmaterials/matminer) <dcbaf06> @  on 2025-09-11 (ver 0.9.3)
# SPDX-License-Identifier: BSD-3-Clause-LBNL
# Copyright (c) 2015 The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to DOE approvals).
# All rights reserved. See vendor/matminer/LICENSE for full terms.
#
# Local modifications:
#   * refactored `__init__` method of `MagpieData`
#   * add some arguments;
#   * Removed every other class except for the MagpieData, import matminer's instead
# (c) 2025 Changhyun Yi for modifications, licensed under the same terms.

"""
Utility classes for retrieving elemental properties. Provides
a uniform interface to several different elemental property resources
including ``pymatgen`` and ``Magpie``.
"""

#import os
from pathlib import Path
import warnings
from glob import glob

import numpy as np
from pymatgen.core.periodic_table import Element
from matminer.utils.warnings import IMPUTE_NAN_WARNING
from matminer.utils.data import AbstractData, OxidationStatesMixin


__author__ = "Kiran Mathew, Jiming Chen, Logan Ward, Anubhav Jain, Alex Dunn"

module_dir = Path(__file__).absolute().parent

class MagpieData(AbstractData, OxidationStatesMixin):
    """
    Class to get data from Magpie files. See also:
    L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A general-purpose machine
    learning framework for predicting properties of inorganic materials,
    Npj Comput. Mater. 2 (2016) 16028.


    Finding the exact meaning of each of these features can be quite difficult.
    Reproduced in ./data_files/magpie_elementdata_feature_descriptions.txt.

    Args:
        impute_nan (bool): if True, the features for the elements
            that are missing from the data_source or are NaNs are replaced by the
            average of each features over the available elements.
    """

    def __init__(self,
                 data_dir: Path | str = None,
                 encoding: str = "utf-8",
                 _props: list[str] = None,
                 skip_lines_table: int = None,
                 features: list[str] | str = "all",
                 impute_nan: bool=False):
        self.all_elemental_props = {}
        available_props = []

        if data_dir is None:
            self.data_dir=Path(module_dir).joinpath("data_files", "magpie_elementdata")
        elif isinstance(data_dir, str):
            data_dir = Path(data_dir)
            if not data_dir.is_absolute():
                data_dir = data_dir.absolute()
            self.data_dir = data_dir
        elif isinstance(data_dir, Path):
            self.data_dir = data_dir.absolute()
        else:
            raise TypeError(data_dir)

        self.encoding=encoding
        self.impute_nan = impute_nan

        # Make a list of available properties
        if features=="all":
            for datafile in Path(self.data_dir).glob("*.table"):
                available_props.append(datafile.stem)
        elif isinstance(features, list):
            assert all([Path(self.data_dir).joinpath(f"{x}.table") for x in features])
            available_props = features
        else:
            raise TypeError(features)
        self.available_props = available_props

        # parse and store elemental properties
        if not skip_lines_table:
            skip_lines_table = 0
        self.parse_store_elemental_props(skip_lines_table)


    def parse_store_elemental_props(self, skiplines: int = None):
        """parse and store elemental properties

        mm..
        """
        for descriptor_name in self.available_props:
            with open(Path(self.data_dir).joinpath(f"{descriptor_name}.table"), encoding=self.encoding) as f:
                self.all_elemental_props[descriptor_name] = dict()
                lines = f.readlines()[skiplines:]
                for atomic_no in range(1, 118 + 1):  # (max Z=118)
                    try:
                        if descriptor_name in ["OxidationStates"]:
                            prop_value = [float(i) for i in lines[atomic_no - 1].split()]
                        else:
                            prop_value = float(lines[atomic_no - 1])
                    except (ValueError, IndexError):
                        prop_value = float("NaN")
                    self.all_elemental_props[descriptor_name][Element.from_Z(atomic_no).symbol] = prop_value

        if self.impute_nan:
            for prop, prop_data in self.all_elemental_props.items():
                if prop == "OxidationStates":
                    nested_props = list(prop_data.values())
                    flatten_props = []
                    for l in nested_props:
                        if isinstance(l, list):
                            for e in l:
                                flatten_props.append(e)
                        else:
                            flatten_props.append(l)

                    avg_prop = np.round(np.nanmean(flatten_props))
                    for e in Element:
                        if (
                            e.symbol not in prop_data
                            or prop_data[e.symbol] == []
                            or np.any(np.isnan(prop_data[e.symbol]))
                        ):
                            prop_data[e.symbol] = [avg_prop]
                else:
                    avg_prop = np.nanmean(list(prop_data.values()))
                    for e in Element:
                        if e.symbol not in prop_data or np.isnan(
                            prop_data[e.symbol]
                        ):
                            prop_data[e.symbol] = avg_prop
        else:
            warnings.warn(f"{self.__class__.__name__}(impute_nan=False):\n" + IMPUTE_NAN_WARNING)

    def get_elemental_property(self, elem, property_name):
        return self.all_elemental_props[property_name][elem.symbol]

    def get_oxidation_states(self, elem):
        return self.all_elemental_props["OxidationStates"][elem.symbol]