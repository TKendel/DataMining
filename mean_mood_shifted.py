import pandas as pd
import numpy as np

# data_original = pd.read_csv('dataset_meanmood_perdate.csv')
# data = data_original.copy()

# grouped_means = data.groupby(['id'])

# grouped_means = data.groupby(['DATE'])['mood'].transform('mean')
# data['mean_mood_new'] = grouped_means

# data["DATE"] = pd.to_datetime(data["DATE"])
# data["DATE"] = pd.to_datetime(data["DATE"]).dt.date

# s = data["DATE"]
# data['new'] = s.map(data.groupby(s).mean_mood.mean().shift(-1))
# print(data.head(50))




data_original = pd.read_csv('dataset_no_NA_new.csv')
data_copy = data_original.copy()

data = data_copy[['id', 'DATE', 'mood']] 


grouped_means = data.groupby(['id'])

grouped_means = data.groupby(['DATE'])['mood'].transform('mean')
data['mean_mood'] = grouped_means

data["DATE"] = pd.to_datetime(data["DATE"])
data["DATE"] = pd.to_datetime(data["DATE"]).dt.date

s = data["DATE"]
data['mean_mood_shifted'] = s.map(data.groupby(s).mean_mood.mean().shift(-1))


to_add = data.loc[:, 'mean_mood':'mean_mood_shifted']
data_copy = pd.concat([data_copy, to_add], axis=1)

data_copy.to_csv('dataset_mood_shifted.csv', index=False)
# check if nan values are present
print(data_copy.isnull().sum())




