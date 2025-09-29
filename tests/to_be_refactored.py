#%% generate compositional5.csv files
from draftsh.data.conversion import Converter 
converter = Converter(r"C:\Users\chhyyi\git_repos\draftsh2025\temp_devs\merged_dataset_0918.csv", "compositional5_temp_nonSC.json", output_dir="./")
converter.convert()
#python src\draftsh\data\conversion.py C:\Users\chhyyi\git_repos\draftsh2025\temp_devs\merged_dataset_0918.csv compositional5.json
#%% featurize_from_5cols_0922.py
from draftsh.dataset import D2TableDataset
from draftsh.feature import Featurizer, MultiSourceFeaturizer
from draftsh.utils.utils import merge_dfs
import pandas as pd
dataset = D2TableDataset(r"C:\Users\chhyyi\git_repos\draftsh2025\temp_devs\compositional5.csv", exception_col=None)
dataset.pymatgen_comps()

featurizer=MultiSourceFeaturizer(config="xu.json")
featurized_df = featurizer.featurize_all(dataset.dataframe, merge_both=True, save_file="featurized_temp_nonsc.csv")

#%%