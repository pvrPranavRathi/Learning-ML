# Data Prprocessing Template

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Importing the dataset
file_path = Path(r"YOUR_LOC")
dataset = pd.read_csv(file_path)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x) 