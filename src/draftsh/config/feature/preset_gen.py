#%% [markdown]
# This code will generate json object to run this code
# `matminer.featurizers.composition.composite.ElementProperty(data_source=src, features = [feat], stats=[stat])`
# in this format
# ```
#   [
#       {"src":"magpie", "feature":"FirstIonizationEnergy", "stat":"mean"},
#       {"src":"magpie", "feature":"GSestBCClatcnt", "stat":"minimum"},
#   ]
# ```
#
# Documentations and codes on Matminer
# * For `composite` class, see [composite](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.composition.html#module-matminer.featurizers.composition.composite)
#   * For `data_source`, see [data](https://hackingmaterials.lbl.gov/matminer/matminer.utils.html#module-matminer.utils.data)  
#       * and [data.py](https://github.com/hackingmaterials/matminer/blob/main/matminer/utils/data.py)  
#       * and for presets [composite module](https://github.com/hackingmaterials/matminer/blob/main/matminer/featurizers/composition/composite.py)  
#   * For `feature`, see each data_source's description.  
#   * For `stats`, see [PropertyStats](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.utils.html#matminer.featurizers.utils.stats.PropertyStats)  
# 
# note that, in [composite module](https://github.com/hackingmaterials/matminer/blob/main/matminer/featurizers/composition/composite.py), fractions working as a weight. if you want non-weighted values, provide equimolar chemical formula in when you declare Composite. (pymatgen.core.composition.Composition)

# %% preset by sources
import json

#%%
out_lst=[]

srcs=["magpie", "pymatgen"]
features = [[
    "Column",
    "Row",
    "CovalentRadius",
    "Electronegativity",
    "NsValence",
    "NpValence",
    "NdValence",
    "NfValence",
    "NValence",
    "NsUnfilled",
    "NpUnfilled",
    "NdUnfilled",
    "NfUnfilled",
    "NUnfilled",
    "FirstIonizationEnergy", 
    "GSestBCClatcnt", 
    "GSvolume_pa",
    "SpaceGroupNumber",
    ],
    [
        "thermal_conductivity",
    ]
]

stats = [["minimum", "maximum", "range", "mean", "avg_dev"], ["minimum", "maximum", "range", "mean", "std_dev"]]

for idx, src in enumerate(srcs):
    out_lst.append({"src":src, "feature":features[idx], "stat":stats[idx]})

print(out_lst)
fp=open("preset1.json", "w")
json.dump(out_lst, fp)

fp.close()
#%% least features
out_lst=[]

srcs=["magpie", "pymatgen"]
features = [[
    "Column",
    "Row",
    "CovalentRadius",
    "Electronegativity",
    "FirstIonizationEnergy",
    "GSestBCClatcnt",
    "NValence",
    "GSvolume_pa",
    "SpaceGroupNumber",
    ],
    [
        "thermal_conductivity",
    ]
]

stats = [["minimum", "maximum", "range", "mean", "avg_dev"], ["minimum", "maximum", "range", "mean", "std_dev"]]

for idx, src in enumerate(srcs):
    out_lst.append({"src":src, "feature":features[idx], "stat":stats[idx]})

print(out_lst)
fp=open("minimal.json", "w")
json.dump(out_lst, fp)

fp.close()
#%%
fp_val=open("preset1.json")
a = json.load(fp_val)
print(len(a))
print(a)