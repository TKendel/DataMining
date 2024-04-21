import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


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

# df = df[["intID", "time", "circumplex.arousal", "circumplex.valence", "activity", "mood"]].copy()

# ## GROUPING
df = (df.groupby(['intID', pd.Grouper(freq='D', key='time')]).mean().reset_index())

# Drop all rows that are completly empty
df = df.dropna( axis=0, how="all")
# Fill out nas using interpolation
df = df.set_index('time')

df = df.interpolate(method="time")

df = df.drop(df.index[:27])
df = df.fillna(0)

df.mood = df.mood.round()
df['circumplex.valence'] = df['circumplex.valence'].round()
df['circumplex.arousal'] = df['circumplex.arousal'].round()


lab_enc = preprocessing.LabelEncoder()
df["mood"] = lab_enc.fit_transform(df["mood"])
# df.to_excel("outputTree.xlsx")

# Split the data into features (X) and target (y)
X = df.drop('mood', axis=1)
y = df['mood']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5, )

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)



