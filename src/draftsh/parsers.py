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
from typing import Callable
import re, math, ast, fractions

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
    def parse(self, inp: str) -> any:
        raise NotImplementedError("self.parse()")
    
    def num_string_parser(self, v: str, uncertainty: bool = False, as_float: bool = True):
        """parsing string of a number

        args:
            * inp: string of a number. It can be a fraction or floats with uncertainty.
        """
        if uncertainty:
            raise NotImplementedError(uncertainty)
        if not as_float:
            raise NotImplementedError(as_float)
        
        assert isinstance(v, str)
        if "(" in v: 
            # number with uncertainty. then, it cannot be a fraction
            assert v.strip()[-1]==")",v.strip()[-1]
            _ = v[v.find("("):-1] # uncertainty not implemented
            v=ast.literal_eval(v[:v.find("(")])
        elif "/" in v:
            frac_pos=v.find("/")
            numerator = ast.literal_eval(v[:frac_pos])
            denominator = ast.literal_eval(v[frac_pos+1:])
            assert isinstance(numerator, int), numerator
            assert isinstance(denominator, int), denominator
            v=fractions.Fraction(numerator, denominator)
        else:
            pass

        if as_float and not uncertainty:
            return float(v)
        else:
            raise NotImplementedError(as_float, not uncertainty)
    
class ElemParser(CellParser):
    """
    parsing elements columns of the dataset
    """
    def __init__(self, policy="elem"):
        super().__init__(policy)

    def parse(self, val: str) -> list[str]:
        elements_in_ptable = [el.symbol for el in Element]
        val = ast.literal_eval(val)
        assert all([x in elements_in_ptable for x in val]), f"Elements list {val} should include only elements symbols"
        return val

class FracParser(CellParser):
    """parse string of frac lists"""
    def __init__(self, policy="frac"):
        super().__init__(policy)

    def parse(self, fracs, multiphase_rule: str | None = "nominal", as_float: bool = True):
        """ parse elemental_fractions as list of numbers
        
        args:
            * multiphase_rule:
                if "nominal", use fracs["nominal"].
        """
        fracs = ast.literal_eval(fracs)
        out: list[int] = []
        if isinstance(fracs, dict):
            if multiphase_rule == "nominal":
                fracs = fracs["nominal"]
            else: 
                raise NotImplementedError(multiphase_rule)
        for v in fracs:
            if isinstance(v, str):
                out.append(self.num_string_parser(v, as_float = as_float))
            else:
                assert isinstance(v, float) or isinstance(v, int), ValueError(v)
                out.append(float(v))
        assert isinstance(out, list)
        assert len(fracs) == len(out)
        return out

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

def process_targets(df: pd.DataFrame, targets: list[str], non_sc_rule: str = "old", exception_row: str | None = "Exceptions") -> pd.DataFrame:
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
        if exception_row:
            assert pd.isna(row[exception_row])
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
                    raise ValueError(f"exception: Tc(K):{val}, type:{type(val)} on {key} is not a string, not nan\nrow_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_passed_tc_cols]}")
            except:
                not_passed_tc_cols.append(key)
                if val== 'Non-superconducting':
                    pass
                else:
                    raise ValueError(f"exception: {val}, type:{type(val)}, pd.isna{pd.isna(val)}, row_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}")
        
        # update target_array
        assert non_sc_rule=="old", NotImplemented
        row_target = []
        if len(tcs)==0:
            print(f"no valid tc parsed. assign Tc=0 for row_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_passed_tc_cols]}")
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
