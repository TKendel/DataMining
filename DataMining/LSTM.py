import pandas as pd
import numpy as np

from matplotlib import pyplot
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from helperFunctions.seriesToSupervised import series_to_supervised


#open the out excel file
data = pd.read_csv('dataset_no_NA_inter_fill.csv')
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df.time)

df = df.drop(['id', 'sms', 'call', 'DATE', 'TIME', 'HOUR'], axis=1)
variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "mood"]

# ## REMOVING OULTIERS USING QUANTILES
Q1 = df[variable_list].quantile(0.01)
Q3 = df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = df[~((df[variable_list] < (Q1 - 1.5 * IQR)) | (df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]


## GROUPING
df = df.groupby(pd.Grouper(key='time', axis=0,  
                      freq='D', sort=True)).mean()

## ROUDNING VALUES
df['mood'] = np.round(df['mood'])
df['circumplex.valence'] = np.round(df['circumplex.valence'])
df['circumplex.arousal'] = np.round(df['circumplex.arousal'])

df = df.interpolate(method="time")

df = df[["circumplex.arousal", "circumplex.valence", "activity", "mood"]].copy()

values = df.values
# integer encode direction
encoder = LabelEncoder()
values[:, -1] = encoder.fit_transform(values[:, -1])

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values) 

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# This need to be changed depedning on the reframed params provided, and the len of columns
# Typically we cut of everyhing apart from the for original collumn values denoted by -1, and the single target column denoted by var/+1/+2
# depending on how far we are shifting i.e. looking into the future
reframed.drop(reframed.iloc[:, 5:8], inplace=True, axis=1)

print(reframed)

# split into train and test sets
values = reframed.values
train_size = int(len(df) * 0.8)
train = values[:train_size, :]
test = values[train_size:, :]

print(test)
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=79, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel("Number of epochs")
pyplot.ylabel("Loss")
pyplot.title("LSTM loss evaluation")
pyplot.legend()
pyplot.savefig('DataMining/modelGraphOutput/LSTM.png')
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
print(inv_yhat)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = inv_y[:,0]

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
print(mean_absolute_error(inv_y,inv_yhat))
print(mean_absolute_percentage_error(inv_y,inv_yhat))
print(mean_squared_error(inv_y,inv_yhat))

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

f = open("modelStatOutput/LSTM.txt", "w")
f.write( f'MAE: {mean_absolute_error(inv_y,inv_yhat)}')
f.write( f'MAPE: {mean_absolute_percentage_error(inv_y,inv_yhat)}')
f.write( f'MSE: {mean_squared_error(inv_y,inv_yhat)}')
f.write( f'RMSE: {rmse}')
