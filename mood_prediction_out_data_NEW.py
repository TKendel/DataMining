import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

#open the out excel file
data = pd.read_csv('dataset_mood_smartphone.csv')
df = pd.DataFrame(data)

df["time"] = pd.to_datetime(df["time"])
pivot_df = df.reset_index().pivot_table(values="value", index=["id", "time"], columns="variable", aggfunc='mean')



# Split the "time" column into "date" and "time" columns
pivot_df.reset_index(inplace=True)


date_col = pd.to_datetime(pivot_df["time"]).dt.date
time_col = pd.to_datetime(pivot_df["time"]).dt.time
hour_col = pd.to_datetime(pivot_df["time"]).dt.strftime('%H:00')

pivot_df.insert(2, "DATE", date_col, True)
pivot_df.insert(3, "TIME", time_col, True)
pivot_df.insert(4, "HOUR", hour_col, True)


print(pivot_df.head() )

#save pivot_df in a csv file
pivot_df.to_csv('pivot_df.csv', index=False)


##  Plot in report

plt.hist([pivot_df['circumplex.arousal'], pivot_df['circumplex.valence']], label=['Arousal', 'Valence'], color=['b', 'c'])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Arousal and Valence score distribution')
plt.legend(loc='upper left')
plt.grid(False)
plt.show()



#plot aoursal and valence score by HOUR 
plt.figure(figsize=(10, 6))
hourly_mean_arousal = pivot_df.groupby("HOUR")["circumplex.arousal"].mean()
hourly_mean_arousal.plot(marker='o', color='b', linestyle='-')

hourly_mean_valence = pivot_df.groupby("HOUR")["circumplex.valence"].mean()
hourly_mean_valence.plot(marker='o', color='c', linestyle='-')

plt.title('Mean Valence and Aoursal Score by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Mean Score')
plt.legend(['Arousal', 'Valence'])
plt.xticks(range(len(hourly_mean_valence.index)), hourly_mean_valence.index, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()










### Plots for the variables

# ### MOOD VARIABLE
# #plotting mood variable

# # hourly_mean_mood = pivot_df.groupby("time")["mood"].mean()
# # plt.hist(hourly_mean_mood, bins=10, color='blue', edgecolor='black')
# # plt.xlabel('Mood')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of Mood')
# # plt.show()


# ### SCREEN VARIABLE

# Group by hour and calculate mean value spent on screen

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




