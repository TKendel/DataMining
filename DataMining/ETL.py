import pandas as pd
import numpy as np
import seaborn

## Pandas options ##
# pd.set_option('display.max_columns', None)

## Load data into data frame ##
data = pd.read_excel("out.xlsx", index_col=0)
df = pd.DataFrame(data)

## Data summary ##
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

# Column graphing ( need some clean up)
# df.plot.hist(column="What is your stress level (0-100)?", bins=20)
# plt.xlim(-10, 200)
# plt.show()