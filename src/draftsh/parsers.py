"""some precedures for dataset.Dataset()

CellParser: xlsx cell Parsers.
    * ElemParser
    * FracParser

other Functions iterate DataFrame rows:
    * filter_row: filter rows with specific condition, return index to filter
    * 

Todo:
    - target_cols: hard coded
    - non_sc_rule: hard coded
    - vals = vals["nominal"]: hard coded
    - if val== 'Non-superconducting': hard coded


"""
from abc import ABC, abstractmethod
import json
from typing import Callable
import re
import math

from pymatgen.core.periodic_table import Element
import pandas as pd
import numpy as np

class CellParser(ABC):
    """xlsx cell parser
    policy = | elements | number: how to parse the cell.
    
    when policy=object
    * Dictionary to dictionary, list to list
    * sometimes, 'Key' strings are not wrapped with quotation marks \
        (Should I handle this here?)
    * numbers in string format
        * just string
        * fractional number
        * (not implemented)with uncertainty
        * (not implemented) an inequations like '<2' 
    """
    def __init__(self, policy: str="object"):
        self.policy=policy

    @abstractmethod
    def parse(self, inp: str) -> object:
        pass

class ElemParser(CellParser):
    """
    parsing elements columns of the dataset
    """
    def __init__(self, policy="elem"):
        super().__init__(policy)

    def parse(self, val: str) -> list[str]:
        elements_in_ptable = [el.symbol for el in Element]
        val = str(val).replace("'",'"')
        val = json.loads(val)
        assert all([x in elements_in_ptable for x in val]), f"Elements list {val} should include only elements symbols"
        return val

class FracParser(CellParser):
    """parse string of frac lists"""
    def __init__(self, policy="frac"):
        super().__init__(policy)
    def temp_parser(self, v):
        """temp parser
        """
        assert isinstance(v, str)
        if "(" in v:
            v=v[:v.find("(")]
        elif "/" in v:
            frac_pos=v.find("/")
            v=float(v[:frac_pos])/float(v[frac_pos+1:])
        else:
            v=float(v)
        return v

    # parse elemental_fractions as list
    def parse(self, fracs):
        fracs = str(fracs).replace("'",'"')
        fracs = json.loads(fracs)
        if isinstance(fracs, dict):
            fracs = fracs["nominal"] #Todo
        for idx, v in enumerate(fracs):
            if isinstance(v, str):
                fracs[idx]=self.temp_parser(v)

        assert isinstance(fracs, list)
        return fracs

# other functions that requires iterate whole data

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

def process_targets(df: pd.DataFrame, targets: list[str], non_sc_rule: str = "old") -> pd.DataFrame:
    """process targets, return a new pd.DataFrame"""
    #checkout targets
    valid_targets = ["avg_Tc", "max_Tc", "std_Tc"]
    if all([t in valid_targets for t in targets]):
        pass
    else:
        raise NotImplementedError(f"current valid targets:{valid_targets}")
    # targets=["avg_Tc", "std_Tc"] # 'temporal option', iterate Tc(K) columns, get average of available 'T_c's
    tc_cols=["Tc(K).resistivity.mid", "Tc(K).magnetization.mid", "Tc(K).resistivity.None", "Tc(K).magnetization.onset", "Tc(K).magnetization.None", "Tc(K).resistivity.zero", "Tc(K).specific_heat.mid", "Tc(K).other.None", "Tc(K).resistivity.onset", "Tc(K).specific_heat.onset", "Tc(K).specific_heat.None", "Tc(K).magnetization.zero", "Tc(K).other.onset", "Tc(K).other.mid", "Tc(K).specific_heat.zero"]
    target_array = []
    for row_idx, row in df.iterrows():
        assert pd.isna(row["Exceptions"])
        # gather tcs, not_passed_tc_cols.
        tcs=[]
        not_passed_tc_cols=[]
        for key in tc_cols:
            val=row[key]
            try:
                if type(val)==str:
                    if "<" in val: # non-SC observed on measure
                        pass
                    elif "≈" in val or "~" in val:
                        val=val.replace("≈","")
                        val=val.replace("~","")
                        tcs.append(parse_value_with_uncertainty(val))
                        not_passed_tc_cols.append(key)
                    else:
                        tcs.append(parse_value_with_uncertainty(val))
                        not_passed_tc_cols.append(key)
                elif pd.isna(val):
                    pass
                    #tcs.append(parse_value_with_uncertainty(str(val)))
                    #not_passed_tc_cols.append(key)
                elif type(val)==float or type(val)==int:
                    tcs.append(parse_value_with_uncertainty(str(val)))
                    not_passed_tc_cols.append(key)
                else:
                    raise ValueError(f"exception: Tc(K):{val}, type:{type(val)} on {key} is not a string, not nan\nrow_idx: {row_idx}, comp:{row["comps_pymatgen"]}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_passed_tc_cols]}")
            except:
                not_passed_tc_cols.append(key)
                if val== 'Non-superconducting':
                    pass
                else:
                    raise ValueError(f"exception: {val}, type:{type(val)}, pd.isna{pd.isna(val)}, row_idx: {row_idx}, comp:{row["comps_pymatgen"]}")
        
        # update target_array
        assert non_sc_rule=="old", NotImplemented
        row_target = []
        if len(tcs)==0:
            print(f"no valid tc parsed. assign Tc=0 for row_idx: {row_idx}, comp:{row["comps_pymatgen"]}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_passed_tc_cols]}")
            for target in targets:
                if target=="avg_Tc" or "max_Tc":
                    row_target.append(0.1) #0.1 is an arbitrary offset for non_sc
                elif target=="std_Tc":
                    row_target.append(0.8) # 0.8 is an arbitrary standard deviation, make `offset+2sigma
                else:
                    raise ValueError(f"target: {target} is not valid")
            assert len(row_target)==len(targets)
            target_array.append(row_target)
        elif len(tcs)==1:
            for target in targets:
                if target == "avg_Tc" or "max_Tc":
                    row_target.append(tcs[0]["mean"]) # it is mean of a single Tc value!!
                elif target=="std_Tc":
                    row_target.append(tcs[0]["std"])
            assert len(row_target)==len(targets)
            target_array.append(row_target)
        else:
            for target in targets:
                if target == "avg_Tc":
                    row_target.append(np.mean([tc["mean"] for tc in tcs]))
                elif target == "max_Tc":
                    row_target.append(np.max([tc["mean"] for tc in tcs]))
                elif target == "std_Tc":
                    averaging_std = np.std([tc["mean"] for tc in tcs])
                    propagated_std = np.sum([tc["std"] for tc in tcs])/len(tcs)
                    row_target.append(averaging_std if averaging_std>propagated_std else propagated_std)
            assert len(row_target)==len(targets)
            target_array.append(row_target)
    assert len(target_array)==len(df)
    return pd.DataFrame(data=target_array, columns=targets, dtype=float)

def mask_exception(df: pd.DataFrame, rule: Callable[[any], bool], filter_col: str = "Exceptions", print_count = True) -> pd.Series:
    mask = df.apply(rule[filter_col], axis=1)
    if print_count:
        print(f"mask.value_counts: {mask.value_counts()}")
    return mask
