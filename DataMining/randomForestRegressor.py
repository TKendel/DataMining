import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def train_test_split(df, test_size=0.2):
    data = df.values
    feature_scaler.fit(data[:, :-1]) 
    target_scaler.fit(data[:, -1:])
    scaled_data = feature_scaler.transform(data[:, :-1])
    scaled_target = target_scaler.transform(data[:, -1:])

    data_scaled = np.concatenate((scaled_data, scaled_target), axis=1)

    n = int(len(data_scaled) * (1 - test_size))
    print(data_scaled[n:].shape[1])
    return data_scaled[:n], data_scaled[n:]

def random_forest_forecast(train, value):
    train = np.array(train)
    X, Y = train[:, :-1], train[:, -1:]
    global model
    model = RandomForestRegressor(n_estimators=1000)

    model.fit(X, Y)
    val = np.array(value).reshape(1, -1)
    prediction = model.predict(val)
    return prediction[0] 

def walk_forward_validation(data, percentage=0.2):
    # In this case  is the target column( Activity)
    train, test = train_test_split(data, percentage)
    predictions = []
    history = [x for x in train]

    for i in range(len(test)):
        test_X, test_Y = test[i, :-1], test[i, -1:]
        yhat = random_forest_forecast(history, test_X)
        predictions.append(yhat)
        history.append(test[i])

    Y_test = target_scaler.inverse_transform(test[:, -1:].reshape(1, -1))
    Y_pred = target_scaler.inverse_transform(np.array(predictions).reshape(1, -1))
    # estimate prediction error
    test_rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    return test_rmse, Y_test, Y_pred

# load the dataset
data = pd.read_csv('dataset_no_NA_inter_fill.csv')
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df.time)

df = df.drop(['id', 'sms', 'call', 'DATE', 'TIME', 'HOUR'], axis=1)

variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]

# ## REMOVING OULTIERS USING QUANTILES
Q1 = df[variable_list].quantile(0.01)
Q3 = df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = df[~((df[variable_list] < (Q1 - 1.5 * IQR)) | (df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]

variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "mood"]

# ## REMOVING OULTIERS USING QUANTILES
Q1 = df[variable_list].quantile(0.01)
Q3 = df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = df[~((df[variable_list] < (Q1 - 1.5 * IQR)) | (df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]

## GROUPING
df = df.groupby(pd.Grouper(key='time', axis=0,  
                      freq='D', sort=True)).mean()

df = df.interpolate(method="time")

df['Lag_1'] = df['activity'].shift(1)
df = df.dropna()

last_row = df.tail(1)
df.drop(df.tail(1).index, inplace=True)
df.dropna(inplace=True)

## ROUDNING VALUES
df['mood'] = np.round(df['mood'])
df['circumplex.valence'] = np.round(df['circumplex.valence'])
df['circumplex.arousal'] = np.round(df['circumplex.arousal'])

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Prediction
mae, y, yhat = walk_forward_validation(df, 0.2)
print('MAE: %.3f' % mae)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
f = open("modelStatOutput/RFR.txt", "w")
f.write(f'MAE: {mean_absolute_error(y,yhat)}')
f.write('\nRMSE: %.3f' %  np.sqrt(mean_squared_error(y, yhat)))
f.write(f'\nMSE: {mean_squared_error(y,yhat)}')

# plot expected vs predicted
plt.plot(y[0], label='Expected')
plt.plot(yhat[0], label='Predicted')
plt.xlabel("Number of expected activities")
plt.ylabel("Activity")
plt.title("Predicted vs Expected")
plt.legend()
plt.savefig('DataMining/modelGraphOutput/RandomForestRegressor.png')
plt.show()
