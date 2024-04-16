import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


#open the out excel file
data = pd.read_csv('dataset_mood_smartphone.csv')
df = pd.DataFrame(data)

df["time"] = pd.to_datetime(df["time"])

df['intID'] = df.id.astype('category').cat.codes

pivot_df = df.reset_index().pivot_table(values="value", index=["intID", "time"], columns="variable", aggfunc='mean')

# Split the "time" column into "date" and "time" columns
pivot_df.reset_index(inplace=True)
pivot_df['date'] = pivot_df['time'].dt.date
pivot_df['time'] = pivot_df['time'].dt.time

pivot_df["time"] = pd.to_datetime(df["time"])
pivot_df["time"] = pd.to_datetime(df["time"])


pivot_df["hour"] = pivot_df["time"].dt.strftime('%H:00')
pivot_df["hour"] = pd.Categorical(pivot_df["hour"], categories=sorted(pivot_df["hour"].unique()))


variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]
time_variable_list = ["screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]


# # # Data summary ##
# # Dataframe dimension
# print(pivot_df.shape[0])
# print(pivot_df.shape[1])
# # Column stats
# for column in pivot_df:
#     print(60*"-")
#     print(f"Data type of {column} is: {pivot_df[column].dtypes}")
#     print(f"Number of null values in '{column}' are: {pivot_df[column].isna().sum(axis = 0)}")
#     print("Counted items per column")
#     print(pivot_df[column].value_counts())
# print(pivot_df.describe())


# ### MOOD VARIABLE
# #plotting mood variable

# # hourly_mean_mood = pivot_df.groupby("time")["mood"].mean()
# # plt.hist(hourly_mean_mood, bins=10, color='blue', edgecolor='black')
# # plt.xlabel('Mood')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of Mood')
# # plt.show()


# # ### SCREEN VARIABLE
# # Group by hour and calculate mean value spent on screen

# plt.figure(figsize=(10, 6))
# hourly_mean_screen = pivot_df.groupby("hour")["screen"].mean()
# hourly_mean_screen.plot(marker='o', color='b', linestyle='-')


# plt.title('Mean Value Spent on Screen by Hour')
# plt.xlabel('Hour of the Day')
# plt.ylabel('Mean Value Spent on Screen')
# plt.xticks(range(len(hourly_mean_screen.index)), hourly_mean_screen.index, rotation=45) 
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ### ACTIVITY VARIABLE
# # Line plot for activity variable grouped by hour
# # Group activity variable by hour

# plt.figure(figsize=(10, 6))
# hourly_mean_activity = pivot_df.groupby("hour")["activity"].mean()
# hourly_mean_activity.plot(marker='o', color='b', linestyle='-')

# plt.title('Mean Value Spent on Activity by Hour')
# plt.xlabel('Hour of the Day')
# plt.ylabel('Mean Value Spent on Screen')
# plt.xticks(range(len(hourly_mean_activity)), hourly_mean_activity.index, rotation=45) 
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ## PROBABILITY DISTRIBUTION
# for variable in variable_list:
#     plt.hist(np.log(pivot_df[variable]), bins=30)

#     plt.xlabel('Data Values')
#     plt.ylabel('Frequency')

#     file_name = f"graphs/probabilityDistributionLog/{variable}.png"
#     plt.savefig(file_name)
#     plt.clf()

# ## BOX PLOTS
# for variable in variable_list:
#     sns.boxplot(np.log(pivot_df[variable]))

#     file_name = f"graphs/boxPlotsLog/{variable}.png"
#     plt.savefig(file_name)
#     plt.clf()
     
# ## LOWER AND UPPER BOUND
# for variable in variable_list:
#     print(30 * "*" + variable + 30 * "*")
#     upper_limit = pivot_df[variable].mean() + 3* pivot_df[variable].std() # Right from the mean
#     lower_limit = pivot_df[variable].mean() - 3* pivot_df[variable].std() # Left from the mean
#     print(upper_limit)
#     print(lower_limit)

# ## REMOVING OULTIERS USING QUANTILES
Q1 = pivot_df[variable_list].quantile(0.01)
Q3 = pivot_df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = pivot_df[~((pivot_df[variable_list] < (Q1 - 1.5 * IQR)) | (pivot_df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]


# ## REMOVING OUTLIERS 
# df = pivot_df[~(pivot_df['appCat.builtin'] < 0)]

# ## LOG times
# for variable in time_variable_list:
#     df[f"{variable}log"] = np.log(df[variable])

# pd.plotting.scatter_matrix(df[["mood", "activity", "screenlog", "appCat.builtinlog", "appCat.communicationlog"]])
# plt.show()

df = df[["time", "circumplex.arousal", "circumplex.valence", "activity", "mood"]].copy()

# ## GROUPING
##TODO Group values based on ID before using the dataframe in predictions
df = df.groupby(pd.Grouper(key='time', axis=0,  
                      freq='1d', sort=True)).mean()

# Drop all rows that are completly empty
df = df.dropna( axis=0, how="all")
# Fill out nas using interpolation
df = df.interpolate(method="time")

## Round up values
df.mood = df.mood.round()
df['circumplex.valence'] = df['circumplex.valence'].round()
df['circumplex.arousal'] = df['circumplex.arousal'].round()
# df.to_excel("outputAll.xlsx")



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = df.values
# integer encode direction
encoder = LabelEncoder()
values[:,3] = encoder.fit_transform(values[:,3])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.head())

...
# split into train and test sets
values = reframed.values
n_train_hours = 100
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
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
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# n_steps = 60
# features = 1
# # split into samples
# X_train, y_train = split_sequence(training_set_scaled, n_steps)
