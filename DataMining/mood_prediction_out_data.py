import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

#open the out excel file
data = pd.read_csv('dataset_mood_smartphone.csv')
df = pd.DataFrame(data)

df["time"] = pd.to_datetime(df["time"])


pivot_df = df.reset_index().pivot_table(values="value", index=["id", "time"], columns="variable", aggfunc='mean')


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

# # ## REMOVING OULTIERS USING QUANTILES
# Q1 = pivot_df[variable_list].quantile(0.25)
# Q3 = pivot_df[variable_list].quantile(0.75)
# IQR = Q3 - Q1

# df = pivot_df[~((pivot_df[variable_list] < (Q1 - 1.5 * IQR)) | (pivot_df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]


## REMOVING OUTLIERS 
df = pivot_df[~(pivot_df['appCat.builtin'] < 0)]

## LOG times
for variable in time_variable_list:
    df[f"{variable}log"] = np.log(df[variable])

pd.plotting.scatter_matrix(df[["mood", "activity", "screenlog", "appCat.builtinlog", "appCat.communicationlog"]])
plt.show()


# plt.hist(df['appCat.builtin'], bins=30)

# plt.xlabel('Data Values')
# plt.ylabel('Frequency')

# file_name = f"\{'appCat.builtin'}.png"
# plt.savefig(file_name)
# plt.clf()


# ## BOX PLOTS
# for variable in variable_list:
#     sns.boxplot(np.log(df[variable]))

#     file_name = f"graphs/boxPlotsNoOutliers/{variable}.png"
#     plt.savefig(file_name)
#     plt.clf()


# ## GROUPING
# g = df.groupby(pd.Grouper(key='time', freq='D')).mean()
# print(g.first())