import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#open the out excel file
data = pd.read_csv('DataMining\dataset_mood_smartphone.csv')
df = pd.DataFrame(data)

df["time"] = pd.to_datetime(df["time"])

df['intID'] = df.id.astype('category').cat.codes

pivot_df = df.reset_index().pivot_table(values="value", index=["intID", "time"], columns="variable", aggfunc='mean')

# Split the "time" column into "date" and "time" columns
pivot_df.reset_index(inplace=True)

variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]
time_variable_list = ["screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]

#  ## LOG times
# for variable in time_variable_list:
#    pivot_df[f"{variable}log"] = np.log(pivot_df[variable])

# ## REMOVING OULTIERS USING QUANTILES
Q1 = pivot_df[variable_list].quantile(0.01)
Q3 = pivot_df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = pivot_df[~((pivot_df[variable_list] < (Q1 - 1.5 * IQR)) | (pivot_df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]

# ## GROUPING
df = (df.groupby(['intID', pd.Grouper(freq='D', key='time')]).mean().reset_index())