import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

 ## LOG times
for variable in time_variable_list:
   pivot_df[f"{variable}log"] = np.log(pivot_df[variable])

# ## REMOVING OULTIERS USING QUANTILES
Q1 = pivot_df[variable_list].quantile(0.01)
Q3 = pivot_df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = pivot_df[~((pivot_df[variable_list] < (Q1 - 1.5 * IQR)) | (pivot_df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]


# ## GROUPING
df = (df.groupby(['intID', pd.Grouper(freq='D', key='time')]).mean().reset_index())


df = df.drop(['intID', 'time', 'sms', 'call'], axis=1)

# Drop all rows that are completly empty
df = df.dropna( axis=0, how="all")
# Fill out nas using interpolation
df = df.fillna(df.mean())

y = df['appCat.weather']
X = df.drop(['appCat.weather'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, shuffle=True)



# X = df.loc[:, ['Lag_1']]
# X.dropna(inplace=True)  # drop missing values in the feature set
# y = df.loc[:, 'appCat.social']  # create the target
# y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

y_pred = np.round(model.predict(X_test))
y_test = np.round(y_test)


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
print(mean_absolute_error(y_test,y_pred))
print(mean_absolute_percentage_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
