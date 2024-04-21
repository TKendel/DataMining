import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


#open the out excel file
data = pd.read_csv('dataset_no_NA_inter_fill.csv')
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df.time)

df = df.drop(['id', 'sms', 'call', 'DATE', 'TIME', 'HOUR'], axis=1)

df['Lag_1'] = df['activity'].shift(1)

## GROUPING
df = df.groupby(pd.Grouper(key='time', axis=0,  
                      freq='D', sort=True)).mean()

df = df.interpolate(method="time")

df['Lag_1'] = df['activity'].shift(1)
df = df.dropna()

df = df[['activity', 'Lag_1']]

last_row = df.tail(1)
df.drop(df.tail(1).index, inplace=True)
df.dropna(inplace=True)

# ## ROUDNING VALUES
# df['mood'] = np.round(df['mood'])
# df['circumplex.valence'] = np.round(df['circumplex.valence'])
# df['circumplex.arousal'] = np.round(df['circumplex.arousal'])

# Split the data into features (X) and target (y)
X = df.drop('Lag_1', axis=1)
y = df['Lag_1']

X, X_test, y, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

y_pred_baseline = [y.mean()] * len(y)
mae_baseline = mean_absolute_error(y, y_pred_baseline)
print("Mean Close Prices:", round(y.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2), "\n")

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)


plt.scatter(X['activity'], y,color='g')
plt.plot(X['activity'], y_pred,color='k')

plt.show()

# fig, ax = plt.subplots()
# ax.plot(X['activity'], y, '.', color='0.25')
# # ax.plot(X['activity'], y_pred)
# ax.set_ylabel('Activity')
# ax.set_xlabel('Lag_1')
# ax.set_title('Lag Plot of Activity ')
# plt.savefig('DataMining/modelGraphOutput/LR.png')
# plt.show()

training_mae = mean_absolute_error(y, model.predict(X))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print("Training MAE:", round(training_mae, 2))
print("Test MAE:", round(test_mae, 2), '\n')

print(mean_absolute_error(y, y_pred))
print(mean_absolute_percentage_error(y, y_pred))
print(mean_squared_error(y, y_pred))

f = open("modelStatOutput/LR.txt", "w")
f.write( f'MAE: {mean_absolute_error(y,y_pred)}')
f.write( f'MAPE: {mean_absolute_percentage_error(y,y_pred)}')
f.write( f'MSE: {mean_squared_error(y,y_pred)}')
