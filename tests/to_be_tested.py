import pandas as pd

from draftsh.dataset import Dataset, DLDataset
from draftsh.feature import MultiSourceFeaturizer
from draftsh.comparison import StanevSuperCon, XuTestHEA

# %% Load XuDataset, StanevSuperCon and featurize
if True:
    import warnings
    ss_dataset = StanevSuperCon()
    ss_dataset.pymatgen_comps()

    xu_test_set = XuTestHEA()
    featurizer = MultiSourceFeaturizer(config=r"xu.json")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message = "invalid value encountered in scalar divide")
        ss_featurized_df = featurizer.featurize_all(ss_dataset.dataframe, save_npz_dir=r"C:\Users\chyi\draftsh2025\temp_devs\supercon_features_temp")

#%%
    ss_featurized_df.to_json("temp.json", indent=4)

# %%
#if __name__ == "__main__":
#    a = DLDataset(r"C:\Users\hms_l\OneDrive\projects\CWNU미팅\2025년8월\0829\250720 HESC dataset.xlsx")
#    a.featurize_and_split()