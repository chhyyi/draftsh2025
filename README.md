# Git repo Draft 2025-08-22

**NOTE**  
* **Repo for collabs**, not ready for public share yet!
* Download Dataset from shared Google Drive!
* Changes: see [changes.md](changes.md)
    * many errors on feature generation corrected until`f394951`

## Installation for Development
### `Poetry` Virtual Environment (Recommended)
tested environment: windows 11, python==(3.11, 3.13), poetry version 2.1.4
* install `git clone https://github.com/chhyyi/draftsh2025.git`
* `cd draftsh2025`
* additional poetry config may be required: `poetry config virtualenvs.use-poetry-python true`
* `poetry install` shoulde success, include `Installing the current project: draftsh`.
* restart terminal, `cd` to repo root.
* `poetry env list` to assure you are running on the virtual environment.
* select python interpreter on the IDLE if required

### Google colab(linux) and pip
successed 2025-08-28. nowadays(2025-08-22), Colab's `python==3.12.11`
* `!git clone https://github.com/chhyyi/draftsh2025.git`
* `!pip install -e draftsh2025/` ***should success***

## Usage
### Load and Featurize in-house dataset

Load latest in-house dataset as a `pandas.DataFrame`, get featurized dataset as numpy array or pandas.DataFrame.
```python
from draftsh.dataset import Dataset
from draftsh.feature import Featurizer
import pandas as pd

dataset = Dataset(your_dataset_path, config="default.json") 

# dataset.dataframe is pandas DataFrame
assert isinstance(dataset.dataframe, pd.DataFrame)

# get featurized dataset
featurizer = Featurizer(config=r"test.json")
featurized_np = dataset.featurize_and_split(featurizer=featurizer, test_size=0.2, shuffle=False, to_numpy=True)
X_train, Y_train, X_test, Y_test = featurized_np
```

### Load additional dataset for comparison
* `draftsh.comparison.XuDataset()`: This is a mimic of [Xu 2025(paper)](https://journal.hep.com.cn/fop/EN/10.15302/frontphys.2025.014205)'s dataset. While their [repository](https://github.com/Dingfei1361/Conventional-SC-HNN-including-dataset) is not accessible, it could not be reproduced as same.
* `draftsh.comparison.StanevDataset()`: I have employed its [Supercon data](https://github.com/vstanev1/Supercon/blob/master/Supercon_data.csv) as the train data of `XuDataset`.
* `draftsh.comparison.KitagawaDataset()`: It is the validation data Xu 2025 employed, in supplements.
```python
from draftsh.comparison import XuDataset
from sklearn.metrics import r2_score
xu_dataset = comparison.XuDataset().dataframe
print(r2_score(xu_dataset["T_c(K)"], xu_dataset["pred_T_c(K)"]))
```

## Configuration Presets 
for `draftsh.Dataset()`:
* `default.json`: default preset
* `test.json`: test preset

for `draftsh.Featurizer()`:
* `compositional.json`: default preset only for compositional features. it is also compatible with `draftsh.comparison.XuDataset()`
* `test.json`

## Test
* **please add working, tested environments**
* before merge request, try this first
* `python -m unittest tests/test.py` from terminal

## (Poetry) Update and Validate Dependencies
* edit pyproject.toml
* `poetry lock`
* `poetry install` and `test` should succeess.

## Example Notebooks
example colab notebooks.
* [draftsh_test.ipynb](https://colab.research.google.com/drive/1xPBWykbPfkP2sLNI8z78uxi3i7vvs89t?usp=sharing)
    * Load in-house dataset
    * get featurized array for ML