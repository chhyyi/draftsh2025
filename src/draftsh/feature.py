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
    * `impute_nan` is hard coded in many parts.
    * featurizer.set_n_jobs(self.config['n_jobs']) 
        * because of some non-staticmethod for ap calc..
"""

import re
import math
from pathlib import Path
import importlib.resources as resources
from itertools import combinations
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.data import AbstractData

from draftsh.data.vendor.matminer.data import MagpieData
from draftsh.utils.utils import config_parser, ConfigSingleSource, merge_dfs

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
    Added arguments on `MagpieData` from `draftsh.data.vendor.matminer.data`}
        * data_dir: save as MagpieData().data_dir
        * skip_lines_table: I put some header about license on every vendored files so I should skip 3 lines
    But its super(self).__init__() is too long without defining any functions so hard to override
    """
    def __init__(self, impute_nan: bool):
        #dummy_magpie = MagpieData(impute_nan=impute_nan)
        with resources.as_file(resources.files("draftsh").joinpath("data", "vendor", "mastml", "mastml", "magpie")) as f:
            super().__init__(data_dir = f, skip_lines_table=3, impute_nan=impute_nan)

class MyElementProperty(ElementProperty):
    """
    def featurize_uw to 'unweight'.
    """
    def __init__(self, data_source: AbstractData | str, features: list[str], stats: list[str], impute_nan=False, unweight: bool = False, config=None):
        super().__init__(data_source, features, stats, impute_nan)
        self.unweight = unweight
        assert unweight == False, NotImplementedError
        self.pstats = CustomPropertyStats() #overrided
        if data_source == "mast-ml":
            self.data_source = MastMLMagpieData(impute_nan = impute_nan)
        else:
            assert isinstance(self.data_source, AbstractData)

class InhouseSecondary(AbstractData):
    """
    secondary features, using elemental features from MagpieData
    
    to featurize 8 descriptors of table 2 of xu et al 2025,
    self.get_elemental_property() return required data.
    Will be consumed by MyElementProperty().featurize()
    """
    def __init__(self, configs: dict, impute_nan: bool = False):
        assert impute_nan==False
    
        self.all_elemental_props={}
        self.available_props=[]
        self.config_per_sources = [ConfigSingleSource(config_1source) for config_1source in configs]
    
        self.first_properties=["NsValence", "NpValence", "NdValence", "NfValence", "NValence", "Electronegativity"]
        with resources.as_file(resources.files("matminer.utils.data")) as path:
            magpie_data = MagpieData(data_dir=path.joinpath("data_files", "magpie_elementdata"), impute_nan=impute_nan, features=self.first_properties)
        self.xu_eights_init(magpie_data=magpie_data)

    def xu_eights_init(self, magpie_data: MagpieData):
        self.configurational_entropy_init()
        self.occu_ve_init(magpie_data=magpie_data)
        self.ionicity_init(magpie_data=magpie_data)
        for config_1source in self.config_per_sources:
            for names in config_1source.iter_config():
                col_name="_".join(names)
                #delete some parameters used when featurize. see `draftsh\config\feature\xu.json`
                col_name.replace("_self_prop::", "_")
                col_name.replace("_self_prop", "")
                self.available_props.append(col_name)
        
    def configurational_entropy_init(self) -> int:
        """it depends only on fractions

        return number of final columns when this property called.
        """
        self.all_elemental_props["configurational_entropy"]={}
        for el in Element:
            self.all_elemental_props["configurational_entropy"][el.symbol]={
                "func": self.configurational_entropy
            }
        return 1 
    
    @staticmethod
    def configurational_entropy(_, fractions):
        """per R.

        implemented by JH Park, 
        instead of configurational entropy.
        """
        arr = np.array(fractions, dtype=float)
        return -np.sum(arr * np.log(arr))
    
    def occu_ve_init(self, magpie_data)->int:
        """elemental data, occupation state of valence electron

        from for xu et al 2025 table II
        should return s,p,d,f valence electrons altogether, to reproduce the equation

        return number of final columns when this property called
        """
        orbs = ["NsValence", "NpValence", "NdValence", "NfValence"]
        feature_names_orbs = ["occu_ve_s", "occu_ve_p", "occu_ve_d", "occu_ve_f"]
        for fn_orb in feature_names_orbs:
            self.all_elemental_props[fn_orb]={}

        for orb, fn_orb in zip(orbs, feature_names_orbs):
            for el in Element:
                nv = magpie_data.get_elemental_property(el, "NValence")
                self.all_elemental_props[fn_orb][el.symbol]={
                    "NValence": nv,
                    "OrbValence":magpie_data.get_elemental_property(el, orb),
                    "func": self.occu_ve
                    }
        return 4

    def occu_ve_s(self, el):
        return self.all_elemental_props["occu_ve_s"][el]
    
    def occu_ve_p(self, el):
        return self.all_elemental_props["occu_ve_p"][el]
    
    def occu_ve_d(self, el):
        return self.all_elemental_props["occu_ve_d"][el]
    
    def occu_ve_f(self, el):
        return self.all_elemental_props["occu_ve_f"][el]    
    
    @staticmethod
    def occu_ve(data_lst, weights = None):
        """
        args:
            assert orb in ("s", "p", "d", "f")
            data_lst should be `[{orb:val, "NValence":val}, ...]
        """
        assert weights is not None
        denom = np.dot([el_data["NValence"] for el_data in data_lst], weights)
        return np.dot([el_data["OrbValence"] for el_data in data_lst], weights)/denom
    
    def ionicity_init(self, magpie_data)-> int:
        """from for xu et al 2025 table II

        return number of final columns when this property called(1)
        """
        self.all_elemental_props["ionicity"]={}
        for el in Element:
            self.all_elemental_props["ionicity"][el.symbol]={
                "func": self.ionicity,
                "electronegativity": float(magpie_data.get_elemental_property(el, "Electronegativity"))
            }
        return 1

    @staticmethod
    def ionicity(data_lst, fracs, last_stat):
        """ionicity

        following supplement table 2 and the descriptions from of Xu et al.(2024) **Not 2025**
            * [Xu et al 2024](https://www.nature.com/articles/s41524-024-01386-4)        
            *descriptions:
                ```
                In the seventh row, f pertains to electronegativity, with $\\bar{f}$ indicating either the average or maximum electronegativity, corresponding to ‘ave’ and ‘max’ in the table; for the ‘boolean’ value, when $\\bar{f}$ is the average and I exceeds 1.7, it is set to 1, otherwise, it is 0 (this serves as the criterion for determining the formation of ionic bonds), totaling three items. 
                ```
            * equation on the table
                $I = 1-e^{1/4 \\sum_i{x_i |f_i - \\bar{f}|}}$,
                    where $f_i$ is electronegativity.
            * bool criteria (True if I>1.7 else False)
        """
        enes = [e["electronegativity"] for e in data_lst]

        if pd.isna(enes).any():
            print(f"is_nan enes, fracs:{fracs}, {pd.isna(enes)}, \
                enes:{enes}, types:{[type(ene) for ene in enes]}, \
                    np.ma.is_masked:{[np.ma.is_masked(ene) for ene in enes]}")
        def ionicity_calc(fracs, enes, criteria):
            sum_sum=np.sum([frac*np.abs(ene-criteria) for ene, frac in zip(enes, fracs)])
            return 1-np.exp(-(1/4)*sum_sum)
        if last_stat=="mean":
            return ionicity_calc(fracs, enes, np.mean(enes))
        elif last_stat=="max":
            return ionicity_calc(fracs, enes, np.max(enes))
        elif last_stat=="bool":
            return 1 if ionicity_calc(fracs, enes, np.mean(enes))>1.7 else 0
    
    def get_elemental_property(self, elem, property_name):
        return self.all_elemental_props[property_name][elem.symbol]
    
    def get_elemental_properties(self, elems, property_name):
        """Get elemental properties for a list of elements

        Args:
            elems - ([Element]) list of elements
            property_name - (str) property to be retrieved
        Returns:
            [float], properties of elements
        """
        return [self.get_elemental_property(e, property_name) for e in elems]

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
        num_features = 0
        col_names = []
        for source in config["sources"]:
            if source in ("matminer", "matminer_expanded", "matminer_secondary"):
                assert isinstance(config[source], list), f"config[source] should be a list of dictionaries but: {config[source]}"
                for config_1source in config[source]:
                    config_single_source = ConfigSingleSource(config_1source)
                    num_features += len(config_single_source)
                    for srcc, feat, stat in config_single_source.iter_config():
                        #delete some parameters used when featurize. see `draftsh\config\feature\xu.json`
                        col_name=f"{srcc}_{feat}_{stat.replace("::","_")}"
                        col_name=col_name.replace("_self_prop::", "_")
                        col_name=col_name.replace("_self_prop", "")
                        col_names.append(col_name)
            elif source == "materials_project":
                raise NotImplementedError(source)
            else:
                raise ValueError(source)
        return num_features, col_names
        # if "xu_eight" in self.config["sources"]:
        #     raise ValueError("xu_eight is merged with matminer_expanded")
        #     self.xu_eight: bool = bool(self.config["xu_eight"])
        #     self.feature_count["xu_eight"] = 8
        # # init configs for materials project api
        # if "materials_project" in self.config["sources"]:
        #     raise NotImplementedError
        # print(f"featurizer initialized; {self.feature_count}")
        # return 

    def featurize_matminer(self,
                           featurized_df: pd.DataFrame,
                           config: dict,
                           comps_col: str = "comps_pymatgen",
                           impute_nan: bool = True,
                           ) -> pd.DataFrame:
        """ return results of `Featurizer.featurize_dataframe`

        For `featurize_dataframe`, see `matminer.featurizers.base.BaseFeaturizer()`

        args:
            * pandas.DataFrame with `comps_pymatgen` column
                * `comps_pymatgen` column: pymatgen....Composition object
            * config: for single source, shoud have "src", "feature", "stat" keys.
        """
        # set data_source
        for config_single_source in config:
            print(f"start featurize {len(config_single_source["feature"])*len(config_single_source["stat"])} features: {config_single_source}")
            data_source=config_single_source["src"]
            featurizer = MyElementProperty(data_source = data_source, features=config_single_source["feature"], stats=config_single_source["stat"], config=config_single_source)
            featurizer.set_n_jobs(self.config['n_jobs'])
            featurizer.featurize_dataframe(featurized_df, col_id = comps_col, inplace=True)
        return featurized_df
    
    def featurize_matminer2nd(self, 
                           featurized_df: pd.DataFrame,
                           config: dict,
                           comps_col: str = "comps_pymatgen",
                           impute_nan: bool = True,
                           ) -> pd.DataFrame:
        """
        in this case, do not init featurizer but just use one featurizer
        """
        data_source=InhouseSecondary(configs=config)
        for config_single_source in config:
            print(f"start featurize {len(config_single_source["feature"])*len(config_single_source["stat"])} features: {config_single_source}")
            featurizer=MyElementProperty(data_source=data_source, features=config_single_source["feature"],
                                         stats=config_single_source["stat"], config=config_single_source)
            featurizer.set_n_jobs(self.config['n_jobs'])
            featurizer.featurize_dataframe(featurized_df, col_id = comps_col, inplace=True)
        return featurized_df
    
        
    
    def featurize_all(self,
                      df: pd.DataFrame,
                      featurized_df: pd.DataFrame = None,
                      save_file: str = None,
                      merge_both: str = False) -> pd.DataFrame:
        """featurize dataframe

        arguments:
            * df: dataframe with "comps_pymatge" column of `pymatgen`'s `Composition` class.
            * save_npz: str = None
                if not none, save processed features of each featurize functions respectedly not a whole one. 
                should be directory path only

        return:
            * featurized_df: 
        """
        featurized_df = df[["comps_pymatgen"]].copy(deep=False)
        for src in self.config["sources"]:
            if src in ("matminer", "matminer_expanded"):
                featurized_df = self.featurize_matminer(featurized_df, config=self.config[src], impute_nan=False)
            elif src=="matminer_secondary":
                featurized_df = self.featurize_matminer2nd(featurized_df, config=self.config[src], impute_nan=False)
            else:
                raise NotImplementedError(src["src"])
        
        #drop temporal column, "comps_pymatgen"
        featurized_df.drop(columns=["comps_pymatgen"], inplace=True)
        assert featurized_df.shape == (len(df), len(self.col_names))

        # replace matminer column names to stored ones..
        featurized_df=featurized_df.rename(columns=dict(zip(featurized_df.columns.tolist(), self.col_names)))

        # save featurized_df
        if save_file is not None:
            if merge_both:
                featurized_df = merge_dfs(df, featurized_df)

            if isinstance(save_file, str):
                save_file = Path(save_file)
            assert not Path(save_file).is_dir(), NotImplementedError("save_file should be a file name")
            if save_file.suffix == ".json":
                featurized_df.to_json(save_file, orient="table", indent=4, index=None)
            elif save_file.suffix == ".npz":
                np.savez(save_file, featurized_df=featurized_df.to_numpy(), allow_pickle=False)
            elif save_file.suffix == ".csv":
                featurized_df.to_csv(save_file, index=None)
            else:
                NotImplementedError("save_file should be a '.csv' or `*.npz` or `*.json` file path")
        return featurized_df
        

class CustomPropertyStats(PropertyStats):
    def __init__(self):
        self.all_aps: tuple[list, None] | tuple[list, list] = None
        self.init_ap_inputs: tuple[list,list] | tuple[list, None] = None

    def call_ap(self, data_lst: list[float], weights: list[float], weights_rule = "temp"):
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

        statistics = stat.split("::")
        unparsed_stat_args=[]
        if len(statistics)>=2:
            for stat_arg in statistics[1:]:
                if stat_arg=="uw":
                    weights = None
                elif stat_arg=="w":
                    pass
                else:
                    unparsed_stat_args.append(stat_arg)

        return getattr(self, statistics[0])(data_lst, weights, *unparsed_stat_args)
        
    @staticmethod
    def iter_pair(data_lst: list[float], weights: list[float], weights_rule = "temp") -> list[tuple[float, float, float]] | list[tuple[float, float]]:
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
                pairs[j][2] = 0.0 if weights_sum==0 else pair[2]/weights_sum
        elif weights is None:
            pairs = [(da, db) for da, db in combinations(data_lst, r=2)]
        else:
            raise ValueError(weights)
        return pairs

    @staticmethod
    def init_all_aps(data_lst: list[float], weights: list[float], weights_rule = "temp") -> tuple[list[float], list[float]]:
        """
        return all absolute percentages as a list
        """
        assert weights_rule == "temp", NotImplementedError(weights_rule)
        # it is not defined for negative or near-zero values. 
        if all([d >= 0.0 for d in data_lst]):
            pass
        else:
            warnings.warn(f"some elemental properties are negative. In this case, AP is not clearly defined. using |f_i-f_j|/(|f_i|+|f_j|) instead.")
        ap_out = []
        if len(data_lst)==1:
            return [0,], None #according to the definition of AP_{ij}
        if len(data_lst)>1 and len(data_lst)<100:
            if weights is not None:
                weight_out = []
                for da, db, weight in CustomPropertyStats.iter_pair(data_lst, weights):
                    out = np.abs(da-db)/np.mean((np.abs(da), np.abs(db)))
                    if np.isnan(out) or (da==0.0 and db==0.0):
                        assert da==0.0 and db==0.0
                        out = 0.0
                    ap_out.append(out)
                    weight_out.append(weight)
                return ap_out, weight_out
            else:
                for da, db in CustomPropertyStats.iter_pair(data_lst, None):
                    out = np.abs(da-db)/np.mean((np.abs(da), np.abs(db)))
                    if np.isnan(out) or (da==0.0 and db==0.0):
                        assert da==0.0 and db==0.0
                        out = 0.0
                    ap_out.append(out)
                return ap_out, None
        else:
            raise ValueError(f"len(data_lst):{len(data_lst)}")
        
    @staticmethod
    def self_prop(data_lst, weights, last_stat = None)->float | bool | int:
        """
        when property requires fractions

        returns a number or bool(float, usually)
        """
        if last_stat is None:
            return data_lst[0]["func"](data_lst, weights)
        else:
            return data_lst[0]["func"](data_lst, weights, last_stat)
        
    @staticmethod
    def ionicity(data_lst, weights):
        assert weights is not None
        
    
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
