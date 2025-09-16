"""
utility functions to convert datatables.

Todo: refactor process_target(too long)
"""
import pandas as pd
import numpy as np
from draftsh.parsers import parse_value_with_uncertainty
from pymatgen.core import Composition

import ast
import warnings

def norm_fracs(comp: Composition | str, elems: list=None, norm:bool=True):
    if isinstance(comp, str):
        comp = Composition(comp)
    fracs=[]
    if elems is not None:
        elems_to_iter=elems
    else:
        elems_to_iter=comp.elements
    for el in elems_to_iter:
        fracs.append(comp.get_atomic_fraction(el))
    return fracs/np.sum(fracs)

def almost_equals_pymatgen_atomic_fraction(a: Composition, b: Composition, rtol: float = 0.1)->bool:
    if set(a.elements)!=set(b.elements):
        return False
    a_fracs=norm_fracs(a, elems=a.elements)
    b_fracs=norm_fracs(b, elems=a.elements)
    return np.allclose(a_fracs, b_fracs, rtol=rtol)

def series_comps_gen(el_symbols:list[str], func, inps):
    """generate chemical composition string from..
    func(inps[idx]) should return fractions(float) list, same length with el_symbols.

    example usage:
        ```python
        inps=[0.88, 0.76, 0.65, 0.54, 0.43, 0.33, 0.21, 0.13, 0.04]
        series_comps_gen(el_symbols=['Ta', 'Nb', 'Zr', 'Hf', 'Ti'], func=(lambda x: [(1-x)/2]*2+[x/3]*3), inps=inps)
        >>> [0.06, 0.06, 0.29333333, 0.29333333, 0.29333333]\
            [0.12, 0.12, 0.25333333, 0.25333333, 0.25333333]\
            [0.175, 0.175, 0.21666667, 0.21666667, 0.21666667]\
            ...
        ```
        
    """
    for i in inps:
        fracs = [f"{fr:.08}" for fr in func(i)]
        fracs = f"[{', '.join(fracs)}]"
        print(fracs)
    return [func(i) for i in inps]

def process_targets(
        df: pd.DataFrame, 
        targets: list[str],
        non_sc_rule: str = "old", # "old" or "nan"
        exception_row: str = "Exceptions",
        valid_targets: tuple[str] = ("avg_Tc", "max_Tc", "std_Tc", "min_Tc"),
        return_num_tcs: bool=False,
        tc_cols=("Tc(K).resistivity.mid", "Tc(K).magnetization.mid", "Tc(K).resistivity.None", "Tc(K).magnetization.onset", "Tc(K).magnetization.None", "Tc(K).resistivity.zero", "Tc(K).specific_heat.mid", "Tc(K).other.None", "Tc(K).resistivity.onset", "Tc(K).specific_heat.onset", "Tc(K).specific_heat.None", "Tc(K).magnetization.zero", "Tc(K).other.onset", "Tc(K).other.mid", "Tc(K).specific_heat.zero")
        ) -> pd.DataFrame:
    """process targets, return a new pd.DataFrame
    
    arguments:
        * return_num_tcs: add "num_valid_tc" column.
    """
    #checkout targets
    assert all([t in valid_targets for t in targets]), NotImplementedError(f"current valid targets:{valid_targets}")
    target_array = []
    if return_num_tcs:
        targets.insert(0, "num_valid_tc")
    for row_idx, row in df.iterrows():
        if exception_row:
            assert pd.isna(row[exception_row])
        # gather tcs, not_passed_tc_cols.
        tcs=[]
        not_none_tc_cols=[]
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
                        not_none_tc_cols.append(key)
                    else:
                        tcs.append(parse_value_with_uncertainty(val))
                        not_none_tc_cols.append(key)
                elif pd.isna(val):
                    pass
                    #tcs.append(parse_value_with_uncertainty(str(val)))
                    #not_passed_tc_cols.append(key)
                elif type(val)==float or type(val)==int:
                    tcs.append(parse_value_with_uncertainty(str(val)))
                    not_none_tc_cols.append(key)
                else:
                    raise ValueError(f"exception: Tc(K):{val}, type:{type(val)} on {key} is not a string, not nan\nrow_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_none_tc_cols]}")
            except:
                not_none_tc_cols.append(key)
                if val=='Non-superconducting' or isinstance(ast.literal_eval(val), dict):
                    warnings.warn(f"skip row: Tc(K):{val}, type:{type(val)} on {key} is not a string, not nan\nrow_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_none_tc_cols]}")
                else:
                    raise ValueError(f"exception: {val}, type:{type(val)}, pd.isna{pd.isna(val)}, row_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}")
        
        # update target_array
        if non_sc_rule=="old":
            row_target_default_mean = 0.1 #0.1 is an arbitrary offset for non_sc
            row_target_default_std = 0.8 # 0.8 is an arbitrary standard deviation, make `offset+2sigma
        elif non_sc_rule=="nan":
            row_target_default_mean = np.nan
            row_target_default_std = np.nan
        else:
            NotImplemented(non_sc_rule)
        row_target = []
        num_tc_row=len(tcs)
        if return_num_tcs:
            row_target.append(num_tc_row)
        if num_tc_row==0:
            print(f"no valid tc parsed. assign Tc=0 for row_idx: {row_idx}, comp:{row.get("comps_pymatgen", "unknown comps")}, tcs:{tcs}, not_passed_tc_cols: {[row[ex_col] for ex_col in not_none_tc_cols]}")
            for target in targets:
                if target in ("avg_Tc", "max_Tc", "min_Tc"):
                    row_target.append(row_target_default_mean) 
                elif target=="std_Tc":
                    row_target.append(row_target_default_std) 
                elif target=="num_valid_tc":
                    pass
                else:
                    raise ValueError(f"target: {target} is not valid")
            assert len(row_target)==len(targets)
        elif num_tc_row==1:
            for target in targets:
                if target in ("avg_Tc", "max_Tc", "min_Tc"):
                    row_target.append(tcs[0]["mean"]) # it is mean of a single Tc value!!
                elif target=="std_Tc":
                    row_target.append(tcs[0]["std"])
                elif target=="num_valid_tc":
                    pass
                else:
                    raise ValueError
            assert len(row_target)==len(targets)
        else:
            for target in targets:
                if target == "avg_Tc":
                    row_target.append(np.mean([tc["mean"] for tc in tcs]))
                elif target == "max_Tc":
                    row_target.append(np.max([tc["mean"] for tc in tcs]))
                elif target == "std_Tc":
                    averaging_std = np.std([tc["mean"] for tc in tcs])
                    propagated_std = np.sum([tc["std"] for tc in tcs])/num_tc_row
                    row_target.append(averaging_std if averaging_std>propagated_std else propagated_std)
                elif target == "min_Tc":
                    row_target.append(np.min([tc["mean"] for tc in tcs]))
                elif target=="num_valid_tc":
                    pass
                else:
                    raise ValueError
            assert len(row_target)==len(targets)
        target_array.append(row_target)
    assert len(target_array)==len(df)
    return pd.DataFrame(data=target_array, columns=targets, dtype=float)