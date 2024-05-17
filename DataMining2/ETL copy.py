import pandas as pd
import numpy as np
import seaborn as sns
import random
import pickle


from matplotlib import pyplot as plt
# ANOVA feature selection for numeric input and categorical output
from sklearn.preprocessing import MinMaxScaler
# Models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


## Pandas options ##
# pd.set_option('display.max_columns', None)

## Load data into data frame ##
data = pd.read_csv("DataMining2/training_set_VU_DM.csv", index_col=0)
df = pd.DataFrame(data).reset_index()

# # # Data summary ##
# # Dataframe dimension
# f = open("DataMining2/columnStats.txt", "w")
# f.write(f"{df.shape[0]} \n")
# f.write(f"{df.shape[1]} \n")
print(f"Row count: {df.shape[0]} \n")
print(f"Column count: {df.shape[1]}")

# # Getting familiar
# n, bins, patches = plt.hist(df.visitor_location_country_id, 100, density = 1, facecolor='blue', alpha=0.75)
# plt.xlabel('Visitor location country Id')
# plt.title('Histogram of visitor_location_country_id')
# file_name = f"DataMining2/graphs/graphHotelSummary/histVisitorLocationCountryID.png"
# plt.savefig(file_name)

# print(df.groupby('visitor_location_country_id').size().nlargest(5))

"""
The data dfes dollars as the go to currency and a lot of the hotel searches
are in one country, out of the total 4958347 rows, 58.33% is one country
which is very likely the dfA :).
Maybe just using that part of the dataset would be better since undersampling/oversampling
for the rest migh prove time consuming.
"""

# # Remove all columns that n nan values above a certain threshold
# df = df.drop(df.columns[df.isnull().mean()>0.80], axis=1)

"""
There is additional info for each collumn in the columnStats.txt like data type
number of null values, unique items per column.
"""
# # Column stats
# for column in df.columns:
#     f.write("\n")
#     f.write(f"Data type of {column} is: {df[column].dtypes} \n")
#     f.write(f"Number of null values in {column} are: {df[column].isna().sum(axis = 0)} \n")
#     f.write("items per column")
#     f.write(f"{df[column].value_counts()} \n")
# f.write(f"{df.describe()} \n")
# f.close()

"""
Dropping all column based on the ammount of nan values and some ID columns which should not be adding much value
given we only care on users input.
"""
# Drop columns
"prop_country_id","prop_starrating","prop_brand_bool","prop_log_historical_price","orig_destination_distance","random_bool","srch_saturday_night_bool","srch_room_count","srch_children_count","srch_adults_count","srch_booking_window","srch_length_of_stay","srch_destination_id","position","prop_log_historical_price"
# dropped_variables = ["date_time","visitor_hist_adr_usd","gross_bookings_usd","visitor_hist_starrating","visitor_location_country_id","srch_query_affinity_score","srch_query_affinity_score","comp1_rate","comp1_inv","comp1_rate_percent_diff","comp2_rate","comp2_inv","comp2_rate_percent_diff","comp3_rate","comp3_inv","comp3_rate_percent_diff","comp4_rate","comp4_inv","comp4_rate_percent_diff","comp5_rate","comp5_inv","comp5_rate_percent_diff","comp6_rate","comp6_inv","comp6_rate_percent_diff","comp7_rate","comp7_inv","comp7_rate_percent_diff","comp8_rate","comp8_inv","comp8_rate_percent_diff"]
# df.drop(columns=dropped_variables, inplace=True)

# # Create clicked and booked avearge for each hotel
# df['avg_click_per_hotel'] = df.groupby(by=['prop_id']).mean()['click_bool']
# df['avg_click_per_hotel'] = df['avg_click_per_hotel'].fillna(0)
# # Create ratio
# df['ratio'] = df.groupby(by=['prop_id']).sum()['booking_bool'] / df.groupby(by=['prop_id']).sum()['click_bool']

# # Take 50% of the data
# df = df.sample(frac=0.5)

print(df.isnull().sum())

# Remove null values
df['prop_review_score'].fillna(3, inplace=True)
df['prop_review_score'] = df['prop_review_score'].replace({0:2.5})
df["prop_location_score2"].fillna(0, inplace=True)
df['srch_query_affinity_score'].fillna((df['srch_query_affinity_score'].mean()), inplace=True)
df['orig_destination_distance'].fillna((df['orig_destination_distance'].median()), inplace=True)
df["visitor_hist_adr_usd"].fillna(0, inplace=True)
df['visitor_hist_starrating_bool'] = pd.notnull(df['visitor_hist_starrating'])

print(df.isnull().sum())

for i in range(1,9):
    df['comp'+str(i)+'_rate'].fillna(0, inplace=True)
df['comp_rate_sum'] = df['comp1_rate']
for i in range(2,9):
    df['comp_rate_sum'] += df['comp'+str(i)+'_rate']

for i in range(1,9):
    df['comp'+str(i)+'_inv'].fillna(2, inplace=True)
    df['comp'+str(i)+'_inv'] = df['comp'+str(i)+'_inv'].replace({1:10, 0:1})
    df['comp'+str(i)+'_inv'] = df['comp'+str(i)+'_inv'].replace({2:0, -1:-2})
df['comp_inv_sum'] = df['comp1_inv']
for i in range(2,9):
    df['comp_inv_sum'] += df['comp'+str(i)+'_inv']

# # Drop rows depending on column name where they are null
# df = df.loc[df[column].notnull()]

# ## PROBABILITY DISTRIBUTION
# for column in df.columns:
#     plt.hist(df[column])
#     plt.xlabel('Data Values')
#     plt.ylabel('Frequency')
#     file_name = f"DataMining2/graphs/probabilityDistribution/{column}.png"
#     plt.savefig(file_name)
#     plt.clf()

# ## BOX PLOTS
# for column in df.columns:
#     sns.boxplot(df[column])
#     plt.xlabel('Data Values')
#     plt.ylabel('Frequency')
#     file_name = f"DataMining2/graphs/boxPlots/{column}.png"
#     plt.savefig(file_name)
#     plt.clf()

# corr = df.corr(method="kendall")
# sns.heatmap(corr, 
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.savefig("DataMining2/graphs/correlationMatrixKendall.png")

# corr = df.corr(method='spearman')
# sns.heatmap(corr, 
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.savefig("DataMining2/graphs/correlationMatrixSpearman.png")

# # # Clean up at asile 3
# # Remove duplicate data
# df = df.drop_duplicates().reset_index()
# # Remove completly empty rows and columns
# df = df.dropna(how='all', axis='columns')
# df = df.dropna(how='all', axis='rows')

# # # Undersampling
# # Get all clicked on hotels
# click_indices = df[df.click_bool == 1].index
# # Randomize selection between them
# random_indices = np.random.choice(click_indices, len(df.loc[df.click_bool == 1]), replace=False)
# # Get all the the regarding them to create a sample of clicked data
# click_sample = df.loc[random_indices]

# not_click = df[df.click_bool == 0].index
# # Get random rows with click_bool=0 and make a vector of size sum click_bool, or len(click_indices)
# random_indices = np.random.choice(not_click, len(click_indices), replace=False)
# not_click_sample = df.loc[random_indices]

# df_new = pd.concat([not_click_sample, click_sample], axis=0)

# # The data is split now 50/50 between clicked and not clicked
# print("Number of rows", len(df_new))

# # Undersampling
# Get all clicked on hotels
clicking_indices = df[df.click_bool == 1].index
# Randomize selection between them
random_indices = np.random.choice(clicking_indices, len(df.loc[df.click_bool == 1]), replace=False)
# Get all the the regarding them to create a sample of clicked data
clicking_sample = df.loc[random_indices]

# Non Booked hotels
not_clicked = df[df.click_bool == 0].index
random_indices = np.random.choice(not_clicked, len(clicking_indices), replace=False)
not_click_sample = df.loc[random_indices]

df_new = pd.concat([not_click_sample, clicking_sample], axis=0)

# mms = MinMaxScaler()
# df_new[['price_usd','orig_destination_distance']] = mms.fit_transform(df_new[['price_usd','orig_destination_distance']])

# #Selelcting factor featires (categorical)
# categorical = ['prop_id']
# for factor in categorical:
#      df_new[factor] = df_new[factor].astype('category')

print(f"Row count: {df_new.shape[0]} \n")
print(f"Column count: {df_new.shape[1]}")

# Save to CSV
# df.to_csv('DataMining2/output/out2.csv', index=False)
train = df_new.copy()

train.drop(columns=df_new.columns[27:54], inplace=True)
train.drop(columns=['position','date_time','srch_id', 'visitor_hist_starrating', 'visitor_hist_starrating_bool'] , inplace=True)

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)#random_state=1 *****

# #define the groups (searches)
# get_group_size = lambda df: df.reset_index().groupby("srch_id")['srch_id'].count()
# train_groups = get_group_size(x_train)
# test_groups = get_group_size(x_test)

# print('Preparing model')
# n=100    #n=100
# model = LGBMRanker(objective="lambdarank",n_estimators=n, force_row_wise=True, seed=1) #The model is seeded*****
# model.fit(x_train,y_train,group=train_groups,eval_set=[(x_test,y_test)],eval_group=[test_groups],eval_metric=['map'])

# # Save to CSV
# df.to_csv('DataMining2/output/out2.csv', index=False)

features = train.values
target = df_new['click_bool'].values

classifier = GradientBoostingClassifier(loss='log_loss', 
                                learning_rate=0.15, 
                                n_estimators=200, 
                                subsample=1.0, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_depth=3, 
                                init=None, 
                                random_state=None, 
                                max_features=None, 
                                verbose=0)
classifier.fit(features, target)

print()
with open("GBC_clicked.pkl", "wb") as f:
    pickle.dump(classifier, f)

#TODO: price gap price, current price, hisotrical prices,

# gradiant boost - decision tree, light gbm