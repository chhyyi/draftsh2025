"""
conversion between data formats

'5 columns' format: 
    * following the agreement of the 2025-09-12 meeting with D.Lee and J. Park.
    * column: Elements, Fraction, Tc_max, Tc_min, Tc_avg.
    * Use `pandas.read_excel()` to read the `inhouse_dataset_merged_0810.xlsx`
    * cell format: 
        * `ast.literal_eval(elements | fraction)` should return list[str] | list[float]
        for each cell of the 2D table ()
        * Tc_max, Tc_min, Tc_avg should be a `float`
        
Todo:
    * refactor - 
"""
import ast
import argparse
import json
from abc import abstractmethod

import pandas as pd
import numpy as np

from draftsh.dataset import Dataset, config_parser
from draftsh.utils.utils import merge_dfs
from draftsh.utils.conversion_utils import process_targets

class TcMerger():
    def __init__(self, idx, criteria, rule):
        self.idx0 = idx
        self.criteria = criteria
        self.crit0 = criteria[idx]
        self.idx_to_be_merged=[]
        
        if rule=="all_tcs":
            self.merge_method=self.merge_all_tcs
        elif rule=="max_Tc":
            self.merge_method=self.merge_to_single_row
        else:
            raise ValueError(rule)
        self.rule=rule
    
    @abstractmethod
    def merge_method(self, five_col_rows: pd.DataFrame, targets: list[str])->dict:
        raise NotImplementedError

    def re_init(self, idx, init_config_criteria:bool=False):
        assert not init_config_criteria
        self.idx0 = idx
        self.crit0 = self.criteria[idx]
        self.idx_to_be_merged=[]

    def merge_all_tcs(self, five_col_rows:pd.DataFrame, targets: list[str]=["max_Tc", "min_Tc", "avg_Tc"])->dict:
        out={}
        for ta in targets:
            if ta=="max_Tc":
                max_tc = np.max([tc for tc in five_col_rows[:][ta] if not pd.isna(tc)])
                out[ta]=max_tc
            elif ta=="min_Tc":
                min_tc = np.min([tc for tc in five_col_rows[:][ta] if not pd.isna(tc)])
                out[ta]=min_tc
            elif ta=="avg_Tc":
                mean_tcs = [] # to calculate new mean, 
                for j, row in five_col_rows.iterrows():
                    mean_tcs=mean_tcs+[row["avg_Tc"]]*int(row["num_valid_tc"])
                out[ta] = np.mean(mean_tcs)
        return out

    def merge_to_single_row(self, five_col_rows:pd.DataFrame, targets: list[str]=["max_Tc", "min_Tc", "avg_Tc"])->dict:
        if self.rule=="max_Tc": # choose row have highest Tc.
            out = five_col_rows.sort_values(by=self.rule, ascending=False)
        elif self.rule=="min_Tc": # choose row have lowest Tc.
            out = five_col_rows.sort_values(by=self.rule, ascending=True)
        else:
            raise ValueError
        return out.loc[:,targets].iloc[0].to_dict()
    
    def end_of_criteria(self, idx, df, targets, idx_to_drop):
        if len(self.idx_to_be_merged)==1:
            pass
        elif len(self.idx_to_be_merged)>1:
            dict = self.merge_method(df.loc[[self.idx0]+self.idx_to_be_merged.copy()], targets)
            for k, v in dict.items():
                df.loc[self.idx0, k]=v
            assert self.idx0 not in self.idx_to_be_merged
            idx_to_drop=idx_to_drop+self.idx_to_be_merged
        if idx is not None:
            self.re_init(idx)
        return idx_to_drop, df
            

class Converter():
    def __init__(self, data_path, convert_config, test:bool = False) -> None:
        self.config = config_parser(config=convert_config, mode="convert")
        self.log={} # temporal logging dict..
        self.log['command_line_args']={
            "data_path":data_path,
            "convert_config":convert_config,
            "test":test}
        self.log["config"]=self.config.copy()
        self.dataset = Dataset(data_path, self.config['dataset_config'])
        self.dataset.validate_by_composition()
        if self.config.get("duplicates_rule") is not None:
            self.dataset.pymatgen_duplicates(rtol=0.02)
            self.log["duplicated_comps"]=self.dataset.duplicated_comps_group
            self.dataset.add_duplicated_comps_column(criteria_rule=self.config['duplicates_rule'].get("criteria"))
        self.save_dir = self.config.get('save_compositional5_dir')
        self.test = test
        self.converted_df: pd.DataFrame


    def convert(self):
        out_df=self.convert_tables(self.dataset)
        if self.test:
            from io import StringIO
            buffer = out_df.to_csv(index=False)
            b = pd.read_csv(StringIO(buffer))
            assert b["elements"].apply(lambda x: all(isinstance(i, str) for i in ast.literal_eval(x))).all()
            assert b["elements_fraction"].apply(lambda x: all(isinstance(i, (int, float)) for i in ast.literal_eval(x))).all()
            assert b.drop(columns=["elements", "elements_fraction"]).apply(lambda x: all(isinstance(i, float) for i in x)).all()
        elif self.save_dir is not None:
            out_df.to_csv(self.save_dir, index=False)
            if self.config.get('save_log_dir') is not None:
                with open(self.config['save_log_dir'], 'w', encoding="utf-8") as f:
                    json.dump(self.log, f, indent=4, ensure_ascii=False)
        else:
            self.converted_df = out_df
    
    def convert_tables(self, dataset: Dataset):
        config=self.config
        targets=dataset.config["targets"]

        target_df = process_targets(
            df=dataset.dataframe,
            targets=targets,
            return_num_tcs=True,
            exception_row=None,
            non_sc_rule='nan')
        
        out_df = merge_dfs(target_df, dataset.dataframe.loc[:,config["keep_cols_from_dataset"]])
        if config.get("keep_merged_dataset_index") is not None:
            out_df[config["keep_merged_dataset_index"]]=list(range(len(out_df)))
        if config.get("duplicates_rule") is not None:
            out_df = self.merge_duplicates(config, out_df, targets)
        out_df = out_df.drop(columns=config["drop_cols_after_merge_duplicates"])
        out_df = self.exclude_exceptions(config, out_df)
        return out_df
    
    def merge_duplicates(self, config, out_df: pd.DataFrame, targets):
        """should be ran before self.exclude_exceptions().

        modify representative(merged) row, return indices to drop.
        """
        groups=self.dataset.duplicated_comps_group
        criteria_rule = config["duplicates_rule"]["criteria"]
        tc_rule = config["duplicates_rule"]["tc"]
        if criteria_rule=="single_ref":
            criteria=out_df["full citation"].to_list()
        elif criteria_rule=="dataset":
            criteria=list(range(len(out_df)))
        else:
            raise NotImplementedError(config["duplicates_rule"]["criteria"])
    
        idx_to_drop=[]
        for idx, group_dict in groups.items():
            merger = TcMerger(idx, criteria, tc_rule)
            for idx1 in groups[idx].keys():
                if idx1==idx:
                    pass
                elif merger.crit0==criteria[idx1]:
                    merger.idx_to_be_merged.append(idx1)
                else:
                    # if criteria changed
                    idx_to_drop, out_df=merger.end_of_criteria(idx1, out_df, targets, idx_to_drop)
            idx_to_drop, out_df=merger.end_of_criteria(None, out_df, targets, idx_to_drop)
        
        out_df = out_df.drop(idx_to_drop)
        return out_df
            

    def exclude_exceptions(self, config, out_df):
        """
        drop rows with invalid values
        """
        self.log["exceptions"]={}
        print(f"shape of df before exceptions: {out_df.shape}")
        self.log["exceptions"]["shape(df)_before_exceptions"]=out_df.shape
        # filter1. non_sc_observed
        out_df = out_df.dropna()
        print(f"shape of df after pd.dropna: {out_df.shape}")
        #filter1. num_elements:
        num_el_range = config["exceptions"].get("num_elements")
        keep_rows=[]
        if num_el_range is not None:
            for idx, row in out_df.iterrows():
                elements_number = len(row["elements"])
                for el in row["elements"]:
                    if el in config["exceptions"]["num_elements"]["ignore_elements"]:
                        elements_number=elements_number-1

                if elements_number<= num_el_range['min']:
                    keep_rows.append(False)
                elif elements_number >= num_el_range['max']:
                    keep_rows.append(False)
                else:
                    keep_rows.append(True)
        out_df = out_df.loc[keep_rows]
        print(f"shape of df satisfying num_elements condition: {out_df.shape}")
        self.log["exceptions"]["shape(df)_after_num_elements_exceptions"]=out_df.shape
        #filter2. Tc
        tc_range = config["exceptions"].get("tc")
        if tc_range is not None:
            out_df = out_df[out_df["max_Tc"] > tc_range['min']]
            out_df = out_df[out_df['min_Tc']<tc_range['max']]
        print(f"shape of df satisfying Tc condition: {out_df.shape}")
        self.log["exceptions"]["shape(df)_after_tc_exceptions"]=out_df.shape
        return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='convert_tables',
                    description='convert between different formats')
    parser.add_argument('dataset')
    parser.add_argument('convert_config')
    parser.add_argument('-t', '--test', action='store_true', default=False)
    args=parser.parse_args()
    
    converter = Converter(args.dataset, args.convert_config, args.test)
    converter.convert()

# `python conversion path\to\merged_dataset_forward.xlsx compositional5_all_tc.json `