"""generate features(descriptors in ML)

main functions:
    - featurize, combining
        - elements, elements_fraction from SC-HEA dataset / using matminer
        - elements properties table
        - and equations of each descriptor

    - make `targets`

Todo:
    * ap_weights rule?
"""

import re
import math
from pathlib import Path


import numpy as np
import pandas as pd

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.utils.stats import PropertyStats

from draftsh.utils import config_parser

# functions to generate 8/909 descriptors of xu 2025
def mixing_entropy_per_r(fractions):
    """
    implemented by JH Park, 
    mixing entropy instead of configurational entropy
    """
    arr = np.array(fractions, dtype=float)
    return -np.sum(arr * np.log(arr))

def val_electron_occupation(test_comp):
    """
    occupation state of valence electron
    """
    orbs = ["NsValence", "NpValence", "NdValence", "NfValence",]
    occu4orbits=ElementProperty(data_source="magpie", features = orbs, stats=["mean"], impute_nan=False)
    n_valence=ElementProperty(data_source="magpie", features = ["NValence"], stats=["mean"], impute_nan=False)
    return np.array(occu4orbits.featurize(comp=test_comp))/np.array(n_valence.featurize(comp=test_comp))

def ionicity(test_comp):
  prop=ElementProperty(data_source="deml", features = ["electronegativity",], stats=["mean", "maximum"], impute_nan=True) #for unidentified reason, nan masked(from pandas) inputed with valid floating numbers

  mean_ene, max_ene = prop.featurize(test_comp)
  elems, fracs = zip(*test_comp.element_composition.items())
  enes = [float(prop.data_source.get_elemental_property(e, "electronegativity")) for e in elems]

  if pd.isna(enes).any():
    print(f"is_nan enes, fracs:{fracs}, {pd.isna(enes)}, elem:{elems}, types:{[type(ene) for ene in enes]}, np.ma.is_masked:{[np.ma.is_masked(ene) for ene in enes]}")
  def ionicity_calc(fracs, enes, criteria):

    sum_sum=np.sum([frac*np.abs(ene-criteria) for ene, frac in zip(enes, fracs)])
    return 1-np.exp(-(1/4)*sum_sum)

  ion_mean = ionicity_calc(fracs, enes, mean_ene)
  ion_max = ionicity_calc(fracs, enes, max_ene)

  return ion_mean, ion_max, 1 if ion_mean>1.7 else 0

# parse string of a float with uncertainty
def parse_value_with_uncertainty(s: str):
    """
    Parse a number string with optional uncertainty in parentheses and return (mean, std).
    Examples:

      "4.563(8)"      -> (4.563, 0.008)
      "4.315"         -> (4.315, 1e-3 / sqrt(12))   # from rounding to last digit
      "-12.0"         -> (-12.0, 1e-1 / sqrt(12))
      "7"             -> (7.0, 1 / sqrt(12))        # rounded to integer
    """
    s_clean = s.strip()
    # Extract a trailing "(digits)" if present
    m_unc = re.search(r"\((\d+)\)\s*$", s_clean)
    unc_digits = None
    if m_unc:
        unc_digits = m_unc.group(1)
        s_core = s_clean[:m_unc.start()].strip()
    else:
        s_core = s_clean

    # Split core into mantissa and optional exponent
    m = re.fullmatch(r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+))", s_core)
    if not m:
        raise ValueError(f"Could not parse numeric value from: {s!r}")

    mantissa_str = m.group(1)
    exponent = 0

    # Mean as normal float
    mean = float(mantissa_str) * (10 ** exponent)

    # Count decimals in the mantissa (digits after the dot)
    if "." in mantissa_str:
        decimals = len(mantissa_str.split(".")[1])
    else:
        decimals = 0

    # Compute std
    if unc_digits is not None:
        # Parentheses uncertainty applies to the last shown decimals of the mantissa
        # std = (integer in parens) * 10^(exponent - decimals)
        std = int(unc_digits) * (10 ** (exponent - decimals))
    else:
        # No uncertainty: assume rounding to last decimal place
        # ULP (one unit in last place) at this magnitude:
        ulp = 10 ** (exponent - decimals)
        # Standard deviation of uniform error over width=ULP
        std = ulp / math.sqrt(12.0)

    return {"mean":float(mean), "std": float(std)}


class Featurizer():
    """featurizer for in-house dataset
    
    config file should be a json file in this structure:
        {
            "sources": [source1, source2, source3..],
            "source1": <some object>,
            "source2": 
        }

    currently implemented sources and their values format
        * matminer: 
            list[{"src": "magpie" | "pymatgen" | "deml" |...,
                "feature": list[str(valid feature)],
                "stat": list[str(valid matminer stat)]}]
            for matminer's composite.ElementProperty.
            for valid feature and matminer string,
                see 
        * BCCfremi:
            temporal from MAST-ML
        * xu_eight: bool. 8/909 features of Xu et al (2025)
        * materials_project: 
            elemental properties from materials project api
    """
    def __init__(self, config: dict | str | Path):
        self.config = config_parser(config, mode="feature")

        assert len(self.config["sources"])>0
        self.feature_count = {}
        # init matminer configs
        if "matminer" in self.config["sources"]:
            assert self.config["matminer"]
            num_matminer_features = sum([len(src["feature"])*len(src["stat"])\
                                for src in self.config["matminer"]])
            self.matminer_col_names = []
            for src in self.config["matminer"]:
                for feat in src["feature"]:
                    for stat in src["stat"]:
                        self.matminer_col_names.append(f"{feat}_{stat}")
            self.feature_count["matminer"] = num_matminer_features
        # init config for 8/909 descriptors of Xu 2025
        if "xu_eight" in self.config["sources"]:
            self.xu_eight: bool = bool(self.config["xu_eight"])
            self.feature_count["xu_eight"] = 8
        # init configs for materials project api
        if "materials_project" in self.config["sources"]:
            raise NotImplementedError
        print(f"featurizer initialized; {self.feature_count}")
    
    def featurize_matminer(self, data = pd.DataFrame) -> pd.DataFrame:
        featurized_dset=[]
        for _, row in data.iterrows():
            assert pd.isna(row["Exceptions"])
            feature_row = []
            for _, inp_desc in enumerate(self.config["matminer"]):
                (src, feat, stat)=(inp_desc["src"], inp_desc["feature"], inp_desc["stat"])
                elem_prop=ElementProperty(data_source=src, features = feat, stats=stat, impute_nan=False)
                feature_row = np.append(feature_row, elem_prop.featurize(comp=row["comps_pymatgen"])).flatten()
            featurized_dset.append(feature_row)
        
        featurized_df = pd.DataFrame(data=featurized_dset, columns=self.matminer_col_names)
        return featurized_df
    
    def featurize_bccfermi(self, df: pd.DataFrame) -> pd.DataFrame:
        """featurize_bccfermi

        License: MIT License

            Copyright (c) 2019 UW-Madison Computational Materials Group

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
        """
        print(f"accessing [mast-ml](https://github.com/uw-cmg/MAST-ML) files, with LICENSE:\n{self.__doc__}")
        pass
        
    
    def featurize_xu8(self, df: pd.DataFrame) -> pd.DataFrame:
        inhouse_cols=[]
        inhouse_cols+=["elec_occu_s", "elec_occu_p","elec_occu_d","elec_occu_f"]
        inhouse_cols.append("mixing_entropy_perR")
        inhouse_cols+=["ionicity_ave", "ionicity_max", "ionicity_bool"]

        features_generated = np.zeros((len(df), len(inhouse_cols)), dtype=float)
        for row_idx, row in df.iterrows():
            gen4row = []
            # valence_electron_occupation
            gen4row += list(val_electron_occupation(row["comps_pymatgen"]))
            #mixing entropy per R
            gen4row.append(mixing_entropy_per_r(row["elements_fraction"]))
            #ionicity
            gen4row += ionicity(row["comps_pymatgen"])
            features_generated[row_idx] = np.array(gen4row, dtype=float)

            featurized_df = pd.DataFrame(data=features_generated, columns=inhouse_cols, dtype = float)
        return featurized_df
    
    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        first = True
        featurized_df: pd.DataFrame
        for src in self.config["sources"]:
            if src == "matminer":
                src_df = self.featurize_matminer(df)
            elif src == "xu_eight":
                src_df = self.featurize_xu8(df)
            else:
                raise NotImplementedError
            if first:
                featurized_df = src_df
                first = False
            else:
                shape_df = np.shape(featurized_df)
                shape_src_df = np.shape(src_df)
                featurized_df = featurized_df.reset_index(drop=True).join(src_df.reset_index(drop=True))
                assert np.shape(featurized_df) == (len(df), shape_df[1]+shape_src_df[1])
            assert len(df) == len(featurized_df)
        return featurized_df

class MyPropertyStats(PropertyStats):
    
    @staticmethod
    def iter_pair(data_lst: list[float], weights: list[float] | None, weights_rule = "temp") -> list[tuple[float, float, float]] | list[tuple[float, float]]:
        """
        iter all data pairs in data_list: list[float]

        weights_rule = "temp": is arbitraly chosen one,
            * $min_{i<j}(w_{ij} AP_{ij})$, where $w_{ij} = \frac{x_i x_j}{sum_{p<q}{x_p x_q}}$
        """
        if weights_rule is not "temp":
            raise NotImplementedError
        len_lst = len(data_lst)
        pairs = []
        if weights is not None:
            weights_sum = 0.0
            for i in range(4):
                for j in range(i+1, 4):
                    pair_weight = weights[i]*weights[j]
                    pairs.append([data_lst[i], data_lst[j], pair_weight])
                    weights_sum += pair_weight
            for pair in pairs:
                pair[2] = pair[2]/weights_sum
        else:
            for i in range(len_lst):
                for j in range(i, len_lst):
                    pairs.append((data_lst[i], data_lst[j]))
            return pairs

    @staticmethod
    def all_aps(data_lst: list[float], weights: list[float] | None, weights_rule = "temp") -> tuple[list[float], list[float]]:
        """
        return all absolute percentages as a list
        """
        assert weights_rule == "temp", NotImplementedError
        list_out = []
        if weights is not None:
            for da, db, weight in MyPropertyStats.iter_pair(data_lst, weights):
                list_out.append(weight*np.abs(da-db)/np.mean(da+db))
        else:
            for da, db, weight in MyPropertyStats.iter_pair(data_lst, None):
                list_out.append((np.abs(da-db)/np.mean(da+db)))
        return list_out

    @staticmethod
    def ap_mean(data_lst, weights = None):
        return np.average(MyPropertyStats.all_aps(data_lst, weights))
    
    @staticmethod
    def ap_maximum(data_lst, weights = None):
        return np.maximum(MyPropertyStats.all_aps(data_lst, weights))
    
    @staticmethod
    def ap_minimum(data_lst, weights = None):
        return np.minimum(MyPropertyStats.all_aps(data_lst, weights))

    @staticmethod
    def ap_range(data_lst, weights = None):
        return MyPropertyStats.ap_maximum(data_lst, weights)-MyPropertyStats.ap_minimum(data_lst, weights)
    