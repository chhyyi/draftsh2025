"""generate features(descriptors in ML)

main functions:
    - featurize, combining
        - elements, elements_fraction from SC-HEA dataset / using matminer
        - elements properties table
        - and equations of each descriptor

    - make `targets`

Todo:
    * ap_weights rule?
    * if np.isnan(out):
"""

import re
import math
from pathlib import Path
import importlib.resources as resources
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pandas as pd

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import AbstractData, OxidationStatesMixin, MagpieData

from draftsh.utils.utils import config_parser, ConfigSingleSource
import warnings

__all__ = ["Featurizer", "MultiSourceFeaturizer"]

def __getattr__(name):
    if name == "Featurizer":
        warnings.warn(
            "Featurizer is deprecated; use MultiSourceFeaturizer",
            DeprecationWarning,
            stacklevel=2,
        )
        return MultiSourceFeaturizer
    raise AttributeError(name)

# functions to generate 8/909 descriptors of xu 2025
def mixing_entropy_per_r(fractions):
    """
    implemented by JH Park, 
    mixing entropy instead of configurational entropy
    """
    arr = np.array(fractions, dtype=float)
    return -np.sum(arr * np.log(arr))

def val_electron_occupation(test_comp, impute_nan: bool = True):
    """
    occupation state of valence electron
    """
    orbs = ["NsValence", "NpValence", "NdValence", "NfValence",]
    occu4orbits=ElementProperty(data_source="magpie", features = orbs,
                                stats=["mean"], impute_nan=impute_nan)
    n_valence=ElementProperty(data_source="magpie", features = ["NValence"],
                              stats=["mean"], impute_nan=impute_nan)
    return np.array(occu4orbits.featurize(comp=test_comp))/np.array(n_valence.featurize(comp=test_comp))

def ionicity(test_comp, impute_nan: bool = True):
    """ionicity

    following supplement table 2 of Xu et al.(2024) **Not 2025**
        * [Xu et al 2024](https://www.nature.com/articles/s41524-024-01386-4)
    $I = 1-e^{1/4 \\sum_i{x_i |f_i - \\bar{f}|}}$,
        where $f_i$ is electronegativity.
    and bool criteria (True if I>1.7 else False)
    """
    prop=ElementProperty(data_source="deml", features = ["electronegativity",],
                         stats=["mean", "maximum"], impute_nan=impute_nan)
    #for unidentified reason, nan masked(from pandas) inputed with valid floating numbers

    mean_ene, max_ene = prop.featurize(test_comp)
    elems, fracs = zip(*test_comp.element_composition.items())
    enes = [float(prop.data_source.get_elemental_property(e, "electronegativity")) for e in elems]

    if pd.isna(enes).any():
        print(f"is_nan enes, fracs:{fracs}, {pd.isna(enes)}, \
              elem:{elems}, types:{[type(ene) for ene in enes]}, \
                np.ma.is_masked:{[np.ma.is_masked(ene) for ene in enes]}")
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

class MastMLMagpieData(MagpieData):
    """
    all I need to change is the MagpieData.data_dir when its instance is initiallized.
    But its super(self).__init__() is too long without defining any functions so hard to override
    """
    def __init__(self, impute_nan: bool):
        #dummy_magpie = MagpieData(impute_nan=impute_nan)
        super().__init__(impute_nan=impute_nan)
        self.dummy_data_dir = self.data_dir
        with patch('matminer.utils.data.os.path.join', new = self.path_hack):
            self.path_hack_worked = False
            super().__init__(impute_nan=impute_nan)
    
    def path_hack(self, *args):
        if not self.path_hack_worked:
            assert Path(args[0])==Path(self.dummy_data_dir).parent.parent, (args, self.dummy_data_dir)
            with resources.as_file(resources.files("draftsh.data.miscs.vendor.mastml.mastml.magpie")) as path:
                self.path_hack_worked = True
                return path
        else:
            return Path(args[0]).joinpath(*args[1:])

class MyElementProperty(ElementProperty):
    """
    def featurize_uw to 'unweight'.
    """
    def __init__(self, data_source: AbstractData, features: list[str], stats: list[str], impute_nan=False, unweight: bool = False):
        super().__init__(data_source, features, stats, impute_nan)
        self.unweight = unweight
        assert unweight == False, NotImplementedError
        self.pstats = CustomPropertyStats() #overrided
        if data_source == "mast-ml":
            self.data_source = MastMLMagpieData(impute_nan = self.impute_nan)

    def featurize_uw(self, comp, unweighted: str):
        """
        if unweight, `fractions = None`
        mostly copy of ElementryProperty.featurize
        """
        warnings.deprecated(f"featurize_uw is deprecated, let {CustomPropertyStats.__name__} handle weight")
        if unweighted == "uwd":
            all_attributes = []
            elements, _ = zip(*comp.element_composition.items())

            for attr in self.features:
                if isinstance(self.data_source, str) and self.data_source=="bccfermi": \
                    # for bccfermi (later for other features):
                    elem_data = [self.bccfermi(e) for e in elements]
                else:
                    elem_data = [self.data_source.get_elemental_property(e, attr) for e in elements]

                for stat in self.stats:
                    all_attributes.append(self.pstats.calc_stat(elem_data, stat, weights=None))

            return all_attributes
        elif unweighted == "wd":
            if isinstance(self.data_source, str) and self.data_source=="bccfermi": \
                # for bccfermi (later for other features):
                all_attributes = []
                elements, weights = zip(*comp.element_composition.items())

                elem_data = [self.bccfermi(e) for e in elements]
                for stat in self.stats:
                    all_attributes.append(self.pstats.calc_stat(elem_data, stat, weights=weights))
                return all_attributes
            else:
                return super().featurize(comp)
        else:
            raise ValueError(unweighted)
    
    def bccfermi(self, e):
        """License: MIT License

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
            SOFTWARE."""
        warnings.warn("accessing [mast-ml](https://github.com/uw-cmg/MAST-ML) files, with LICENSE: \n"+self.bccfermi.__doc__, UserWarning)
        with resources.as_file(resources.files("draftsh.data.miscs") /"BCCfermi.csv") as path:
            csv_path = path
        supercon_preprocessed = pd.read_csv(csv_path)
        supercon_dict = supercon_preprocessed.set_index("element").to_dict()['BCCfermi']
        return supercon_dict.get(e.symbol)

class MultiSourceFeaturizer():
    """featurizer for in-house dataset and features
    
    load config, run multiple MyElementProperty class.
    To apply different set of stats and features combination,
    Multiple Featurizer(`MyElementProperty` class) will be used

    args:
        * config: config json file in this structure:
            {
                "sources": [source1, source2, source3..],
                "source1": input configs for the `source1: MyElementProperty()` class,
                "source2": input configs for the `source2`
                ...
            }

    currently implemented sources and their values format
        * matminer_expanded:
            * requires config json file;
                list[{"src": "magpie" | "pymatgen" | "deml" |...,
                    "feature": list[str(valid feature)],
                    "stat": list[str(valid matminer stat)]}]
            * additional features
                * matminer_mastml: 
                    * [MAST-ML](https://github.com/uw-cmg/MAST-ML) tables
                    * including BCCfremi, the main descriptor of Xu 2025
                        
                * xu_eight: bool. 8/909 features of Xu et al (2025)
                * (NotImplemented) materials_project: 
                    elemental properties from materials project api

    config preset:
        * `xu.json`: reproducing xu et al 2025 except for many elemental properties on Table 1
    """
    def __init__(self, config: dict | str | Path):
        self.config = config_parser(config, mode="feature")

        assert len(self.config["sources"])>0, self.config["sources"]
        self.feature_count, self.col_names = self.init_feature_config(self.config)
        # init config for 8/909 descriptors of Xu 2025
    
    def init_feature_config(self, config: dict):
        # init matminer configs
        num_matminer_features = 0
        matminer_col_names = []
        for source in config["sources"]:
            if source == "matminer" or source == "matminer_expanded":
                assert isinstance(config[source], list), f"config[source] should be a list of dictionaries but: {config[source]}"
                for config_1source in config[source]:
                    config_single_source = ConfigSingleSource(config_1source)
                    num_matminer_features += len(config_single_source)
                    for srcc, feat, stat in config_single_source.iter_config():
                        matminer_col_names.append(f"{srcc}_{feat}_{stat.replace("::","_")}")
            elif source == "materials_project":
                raise NotImplementedError(source)
            else:
                raise ValueError(source)
        return num_matminer_features, matminer_col_names
        

        
        if "xu_eight" in self.config["sources"]:
            raise ValueError("xu_eight is merged with matminer_expanded")
            self.xu_eight: bool = bool(self.config["xu_eight"])
            self.feature_count["xu_eight"] = 8
        # init configs for materials project api
        if "materials_project" in self.config["sources"]:
            raise NotImplementedError
        print(f"featurizer initialized; {self.feature_count}")
        return 

    def featurize_matminer(self,
                           data: pd.DataFrame,
                           config: dict,
                           comps_col: str = "comps_pymatgen",
                           save_npz_dir: str | None = None,
                           file_name: str | None = "matminer_features.npz",
                           impute_nan: bool = True,
                           ) -> pd.DataFrame:
        """ return results of `Featurizer.featurize_dataframe`

        For `featurize_dataframe`, see `matminer.featurizers.base.BaseFeaturizer()`

        args:
            * pandas.DataFrame with `comps_pymatgen` column
                * `comps_pymatgen` column: pymatgen....Composition object
            * config: for single source, shoud have "src", "feature", "stat" keys.
        """
        if save_npz_dir is not None:
            assert Path(save_npz_dir).is_dir(), NotImplementedError(save_npz_dir)
            save_npz_pth = Path(save_npz_dir).joinpath(file_name)
        for config_single_source in config:
            featurizer = MyElementProperty(config_single_source["src"], config_single_source["feature"], config_single_source["stat"])
            featurized_df = featurizer.featurize_dataframe(data, col_id = comps_col, inplace=False)
        return featurized_df
        config_1source = ConfigSingleSource(config)
        for src, feautre, stat in config_1source.iter_config():
            featurized_dset=np.zeros((len(data), self.feature_count["matminer_expanded"]), dtype=float)
        for idx, row in data.iterrows():
            assert pd.isna(row.get("Exceptions", None))
            feature_row = []
            for inp_desc in self.config["matminer"]:
                for uw in inp_desc.get('unweighted', ["w",]):
                    # process weighted
                    assert uw=="w" or uw=="uw"
                    (src, feat, stat)=(inp_desc["src"], inp_desc["feature"], inp_desc["stat"])
                    elem_prop=MyElementProperty(data_source=src, features = feat, stats=stat, impute_nan=impute_nan)
                    feature_row = np.append(feature_row, elem_prop.featurize_uw(comp=row["comps_pymatgen"], unweighted = uw)).flatten()
            featurized_dset[idx] = np.array(feature_row, dtype=float)
            if idx%100==0:
                print(f"processed matminer features. {idx}/{len(data)}")
        if save_npz_dir is not None:
            np.savez(save_npz_pth, featurized_dset)
        
        featurized_df = pd.DataFrame(data=featurized_dset, columns=self.col_names["matminer_expanded"])
        return featurized_df
    
    def featurize_xu8(self, df: pd.DataFrame,
                      save_npz_dir: str | None = None,
                      file_name: str | None = "xu8_features.npz"
                      ) -> pd.DataFrame:
        if save_npz_dir is not None:
            assert Path(save_npz_dir).is_dir(), NotImplementedError(save_npz_dir)
            save_npz = Path(save_npz_dir).joinpath(file_name)

        inhouse_cols=[]
        inhouse_cols+=["elec_occu_s", "elec_occu_p","elec_occu_d","elec_occu_f"]
        inhouse_cols.append("mixing_entropy_perR")
        inhouse_cols+=["ionicity_ave", "ionicity_max", "ionicity_bool"]
        lendata = len(df)
        self.col_names["xu_eight"] = inhouse_cols


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

            if row_idx%100==0:
                print(f"processed xu8 features. {row_idx}/{lendata}")
        if save_npz_dir is not None:
            np.savez(save_npz, features_generated)
        featurized_df = pd.DataFrame(data=features_generated, columns=inhouse_cols, dtype = float)

        return featurized_df
    
    def featurize_all(self, df: pd.DataFrame, save_npz_dir: str | None = None, featurized_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """featurize dataframe

        arguments:
            * df: dataframe with "comps_pymatge" column of `pymatgen`'s `Composition` class.
            * save_npz: str | None = None
                if not none, save processed features of each featurize functions respectedly not a whole one. 
                should be directory path only

        return:
            * featurized_df: 
        """
        if save_npz_dir is not None:
            assert(Path(save_npz_dir).is_dir())

        featurized_df = df[["comps_pymatgen"]]
        for src in self.config["sources"]:
            if src == "matminer" or src == "matminer_expanded":
                featurized_df = self.featurize_matminer(featurized_df, config=self.config[src], save_npz_dir=save_npz_dir)
            else:
                raise NotImplementedError(src["src"])
        
        #drop temporal column, "comps_pymatgen"
        featurized_df.drop(columns=["comps_pymatgen"], inplace=True)
        assert featurized_df.shape == (len(df), len(self.col_names))
        return featurized_df
    
    def merge_feature_dfs(self, src_df: pd.DataFrame, featurized_df: pd.DataFrame, reset_index=True):
        assert reset_index, NotImplementedError(reset_index)

        shape_df = np.shape(featurized_df)
        shape_src_df = np.shape(src_df)
        featurized_df = featurized_df.reset_index(drop=True).join(src_df.reset_index(drop=True))

        assert featurized_df.shape[1] == shape_df[1]+shape_src_df[1]
        return featurized_df
        

class CustomPropertyStats(PropertyStats):
    def __init__(self):
        self.all_aps: tuple[list, None] | tuple[list, list] = None
        self.init_ap_inputs: tuple[list,list] | tuple[list, None] | None = None

    def call_ap(self, data_lst: list[float], weights: list[float] | None, weights_rule = "temp"):
        if (data_lst, weights) == self.init_ap_inputs:
            pass
        else:
            self.all_aps = self.init_all_aps(data_lst, weights, weights_rule = weights_rule)
        return self.all_aps

    def calc_stat(self, data_lst, stat, weights=None):
        """
        override to handle 
            * weighted/unweighted stats 
            * and aps.

        see `super().calc_stat` for further
        """

        if "::" in stat:
            statistics = stat.split("::")
            if len(statistics)==2:
                unweighted = statistics.pop()
                if unweighted=="uw":
                    weights = None
                elif unweighted=="w":
                    pass
                else:
                    raise NotImplementedError("I did not considered to use some extra statistics parameters except for the `unweighted: 'uw' | 'w'`.")
            else:
                raise NotImplementedError("I did not considered to use some extra statistics parameters except for the `unweighted: 'uw' | 'w'`.")

        else:
            statistics = [stat]
        return getattr(self, statistics[0])(data_lst, weights, *statistics[1:])
        
    @staticmethod
    def iter_pair(data_lst: list[float], weights: list[float] | None, weights_rule = "temp") -> list[tuple[float, float, float]] | list[tuple[float, float]]:
        """
        iter all data pairs in data_list: list[float]

        weights_rule = "temp": is arbitraly chosen one, to reproduce xu et al (2025)
            * $min_{i<j}(w_{ij} AP_{ij})$, where $w_{ij} = \frac{x_i x_j}{sum_{p<q}{x_p x_q}}$
            * I don't think AP_miminum is reasonable but if I ignore weight,
                just 2 same number will be generated(weighted, unweighted).
        """
        if weights_rule!="temp":
            raise NotImplementedError
        if weights is not None:
            pairs = []
            weights_sum = 0.0
            for (da, wa), (db, wb) in combinations(zip(data_lst, weights), r=2):
                pair_weight = wa*wb
                pairs.append([da, db, pair_weight])
                weights_sum += pair_weight
            for j, pair in enumerate(pairs):
                pairs[j][2] = pair[2]/weights_sum
        elif weights is None:
            pairs = [(da, db) for da, db in combinations(data_lst, r=2)]
        else:
            raise ValueError(weights)
        return pairs

    @staticmethod
    def init_all_aps(data_lst: list[float], weights: list[float] | None, weights_rule = "temp") -> tuple[list[float], list[float] | None]:
        """
        return all absolute percentages as a list
        """
        assert weights_rule == "temp", NotImplementedError(weights_rule)
        # it is not defined for negative or near-zero values. 
        assert all([d >= 0.0 for d in data_lst]), ValueError([d >= 0.0 for d in data_lst]) 
        ap_out = []
        if len(data_lst)==1:
            return [0,], None #according to the definition of AP_{ij}
        if len(data_lst)>1 and len(data_lst)<100:
            if weights is not None:
                weight_out = []
                for da, db, weight in CustomPropertyStats.iter_pair(data_lst, weights):
                    out = np.abs(da-db)/np.mean((da, db))
                    if np.isnan(out):
                        assert da==0.0 and db==0.0
                        out = 0.0
                    ap_out.append(out)
                    weight_out.append(weight)
                return ap_out, weight_out
            else:
                for da, db in CustomPropertyStats.iter_pair(data_lst, None):
                    out = np.abs(da-db)/np.mean((da, db))
                    if np.isnan(out):
                        assert da==0.0 and db==0.0
                        out = 0.0
                    ap_out.append(out)
                return ap_out, None
        else:
            raise ValueError(f"len(data_lst):{len(data_lst)}")

    def ap_mean(self, data_lst, weights = None):
        aps, ap_weights = self.call_ap(data_lst, weights)
        return np.average(aps, weights=ap_weights)
    
    def ap_maximum(self, data_lst, weights = None):
        aps, ap_weights = self.call_ap(data_lst, weights)
        if weights is not None:
            aps = np.multiply(aps, ap_weights)
        return PropertyStats.maximum(data_lst=aps, weights=weights)
    
    def ap_minimum(self, data_lst, weights = None):
        aps, ap_weights = self.call_ap(data_lst, weights)
        if weights is not None:
            aps = np.multiply(aps, ap_weights)
        return PropertyStats.minimum(data_lst=aps, weights=weights)
    
    def ap_range(self, data_lst, weights = None):
        return self.ap_maximum(data_lst, weights)-self.ap_minimum(data_lst, weights)
