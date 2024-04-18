import pandas as pd
import numpy as np


data = pd.read_csv("pivot_df.csv")  
data_a=data.copy(deep=True)

# Check missing values in the data
missing_values = data.isnull().sum()
# print(missing_values)


## FORWARD - BACKWARD FILL

#Filling missing values - FORWARD fill
data_a.fillna(method='ffill',inplace=True)

#Filling missing values - BACKWARD fill
#Second fill in the remaining missing values with the last valid value
data_a.fillna(method='bfill',inplace=True)

#Save the cleaned data FORWARD - BACKWARD FILL
data_a.to_csv('dataset_no_NA_for_back_fill.csv', index=False)

# print(data_a.isnull().sum())




## LINEAR INTERPOLATION
data_b=data.copy(deep=True)

data_b.interpolate(limit_direction="both",inplace=True)

# Save the cleaned data LINEAR INTERPOLATION
data_b.to_csv('dataset_no_NA_inter_fill.csv', index=False)

# print(data_b.isnull().sum())