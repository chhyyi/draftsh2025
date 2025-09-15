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

import pandas as pd

from draftsh.dataset import Dataset, config_parser
from draftsh.utils.utils import merge_dfs
from draftsh.utils.conversion_utils import process_targets

class Converter():
    def __init__(self, dataset: Dataset, dataset_config, convert_config, save_dir=None, test:bool = False) -> None:
        self.dataset = Dataset(dataset, config=dataset_config)
        self.dataset.pymatgen_duplicates(rtol=0.02)
        self.dataset.validate_by_composition()
        self.dataset.merge_duplicated_comps(rule=convert_config.get("duplicates_rule"))
        self.config = convert_config
        self.save_dir = save_dir
        self.test = test
        self.converted_df: pd.DataFrame

    def convert(self):
        out_df=self.convert_tables(self.dataset, config=self.config)
        if self.test:
            from io import StringIO
            buffer = out_df.to_csv(index=False)
            b = pd.read_csv(StringIO(buffer))
            assert b["elements"].apply(lambda x: all(isinstance(i, str) for i in ast.literal_eval(x))).all()
            assert b["elements_fraction"].apply(lambda x: all(isinstance(i, (int, float)) for i in ast.literal_eval(x))).all()
            assert b.drop(columns=["elements", "elements_fraction"]).apply(lambda x: all(isinstance(i, float) for i in x)).all()
        elif self.save_dir is not None:
                out_df.to_csv(self.save_dir, index=False)
        else:
            self.converted_df = out_df
    
    def convert_tables(self, dataset: Dataset, config = "five_col.json"):
        config=config_parser("five_col.json", mode="convert")
        target_df = process_targets(df=dataset.dataframe, targets=dataset.config["targets"], exception_row=None, non_sc_rule='nan')
        out_df = merge_dfs(target_df, dataset.dataframe[config["keep_cols"]])
        tc_cols = dataset.config['targets']

        print(f"shape of df before exceptions: {out_df.shape}")
        # filter1. non_sc_observed
        out_df = out_df.dropna()
        print(f"shape of df after pd.dropna: {out_df.shape}")
        #filter1. num_elements:
        num_el_range = config["exceptions"].get("num_elements")
        if num_el_range is not None:
            out_df = out_df[[len(x)>= num_el_range['min'] for x in out_df["elements"]]]
            out_df = out_df[[len(x)<= num_el_range['max'] for x in out_df["elements"]]]
        print(f"shape of df satisfying num_elements condition: {out_df.shape}")
        #filter2. Tc
        tc_range = config["exceptions"].get("tc")
        if tc_range is not None:
            out_df = out_df[out_df["min_Tc"] > tc_range['min']]
            out_df = out_df[out_df['max_Tc']<tc_range['max']]
        print(f"shape of df satisfying Tc condition: {out_df.shape}")
        
        return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='convert_tables',
                    description='convert between different formats')
    parser.add_argument('dataset')
    parser.add_argument('dataset_config', default="default_forward.json")
    parser.add_argument('convert_config', default="five_col.json")
    parser.add_argument('-s', '--save_dir', default=None)
    parser.add_argument('-t', '--test', action='store_true', default=False)
    args=parser.parse_args()
    
    converter = Converter(args.dataset, args.dataset_config, args.convert_config, save_dir=args.save_dir, test=args.test)
    converter.convert()

# `python conversion path\to\merged_dataset_forward.xlsx default_forward.json `