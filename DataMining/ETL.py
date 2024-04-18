import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

## Pandas options ##
# pd.set_option('display.max_columns', None)

## Load data into data frame ##
data = pd.read_excel("out.xlsx", index_col=0)
df = pd.DataFrame(data)

variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]

# # Data summary ##
# Dataframe dimension
print(df.shape[0])
print(df.shape[1])

# Column stats
for column in df:
    print(60*"-")
    print(f"Data type of {column} is: {df[column].dtypes}")
    print(f"Number of null values in '{column}' are: {df[column].isna().sum(axis = 0)}")
    print("Counted items per column")
    print(df[column].value_counts())
print(df.describe())


### MOOD VARIABLE
#plotting mood variable

hourly_mean_mood = df.groupby("time")["mood"].mean()
plt.hist(hourly_mean_mood, bins=10, color='blue', edgecolor='black')
plt.xlabel('Mood')
plt.ylabel('Frequency')
plt.title('Histogram of Mood')
plt.show()


# ### SCREEN VARIABLE
# Group by hour and calculate mean value spent on screen

plt.figure(figsize=(10, 6))
hourly_mean_screen = df.groupby("hour")["screen"].mean()
hourly_mean_screen.plot(marker='o', color='b', linestyle='-')

plt.title('Mean Value Spent on Screen by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Mean Value Spent on Screen')
plt.xticks(range(len(hourly_mean_screen.index)), hourly_mean_screen.index, rotation=45) 
plt.grid(True)
plt.tight_layout()
plt.show()


### ACTIVITY VARIABLE
# Line plot for activity variable grouped by hour
# Group activity variable by hour

plt.figure(figsize=(10, 6))
hourly_mean_activity = df.groupby("hour")["activity"].mean()
hourly_mean_activity.plot(marker='o', color='b', linestyle='-')

plt.title('Mean Value Spent on Activity by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Mean Value Spent on Screen')
plt.xticks(range(len(hourly_mean_activity)), hourly_mean_activity.index, rotation=45) 
plt.grid(True)
plt.tight_layout()
plt.show()

## PROBABILITY DISTRIBUTION
for variable in variable_list:
    plt.hist(np.log(df[variable]), bins=30)

    plt.xlabel('Data Values')
    plt.ylabel('Frequency')

    file_name = f"graphs/probabilityDistributionLog/{variable}.png"
    plt.savefig(file_name)
    plt.clf()

## BOX PLOTS
for variable in variable_list:
    sns.boxplot(np.log(df[variable]))

    file_name = f"graphs/boxPlotsLog/{variable}.png"
    plt.savefig(file_name)
    plt.clf()

## LOWER AND UPPER BOUND
for variable in variable_list:
    print(30 * "*" + variable + 30 * "*")
    upper_limit = df[variable].mean() + 3* df[variable].std() # Right from the mean
    lower_limit = df[variable].mean() - 3* df[variable].std() # Left from the mean
    print(upper_limit)
    print(lower_limit)

# ## REMOVING OULTIERS USING QUANTILES
Q1 = df[variable_list].quantile(0.01)
Q3 = df[variable_list].quantile(0.99)
IQR = Q3 - Q1

df = df[~((df[variable_list] < (Q1 - 1.5 * IQR)) | (df[variable_list] > (Q3 + 1.5 * IQR))).any(axis=1)]


time_variable_list = ["screen", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]

## LOG times
for variable in time_variable_list:
    df[f"{variable}log"] = np.log(df[variable])

## GROUPING
df = df.groupby(pd.Grouper(key='time', axis=0,  
                      freq='1h', sort=True)).mean()