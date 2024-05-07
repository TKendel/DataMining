import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
# ANOVA feature selection for numeric input and categorical output
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


## Pandas options ##
# pd.set_option('display.max_columns', None)

## Load data into data frame ##
data = pd.read_csv("DataMining2/training_set_VU_DM.csv", index_col=0)
test_data = pd.read_csv("DataMining2/test_set_VU_DM.csv", index_col=0)
df = pd.DataFrame(data)
df_test = pd.DataFrame(test_data)

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

print(df.groupby('visitor_location_country_id').size().nlargest(5))

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
dropped_variables = ["date_time","prop_id","visitor_hist_adr_usd","gross_bookings_usd","visitor_hist_starrating","visitor_location_country_id","srch_query_affinity_score","srch_query_affinity_score","comp1_rate","comp1_inv","comp1_rate_percent_diff","comp2_rate","comp2_inv","comp2_rate_percent_diff","comp3_rate","comp3_inv","comp3_rate_percent_diff","comp4_rate","comp4_inv","comp4_rate_percent_diff","comp5_rate","comp5_inv","comp5_rate_percent_diff","comp6_rate","comp6_inv","comp6_rate_percent_diff","comp7_rate","comp7_inv","comp7_rate_percent_diff","comp8_rate","comp8_inv","comp8_rate_percent_diff"]
df.drop(columns=dropped_variables, inplace=True)

# Shuffle the rows around adn take 50% of the data
df = df.sample(frac=0.5, random_state=99).reset_index(drop=True)

print(df.isnull().sum())

# Remove null values
df["prop_review_score"].fillna((df["prop_review_score"].mean()), inplace=True)
df["prop_location_score2"].fillna((df["prop_location_score2"].mean()), inplace=True)
df['orig_destination_distance'].fillna((df['orig_destination_distance'].mean()), inplace=True)

print(df.isnull().sum())

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

# # Clean up at asile 3
# Remove duplicate data
df = df.drop_duplicates()
# Remove completly empty rows and columns
df = df.dropna(how='all', axis='columns')
df = df.dropna(how='all', axis='rows')

# correlation = df.corr(method = 'spearman')
# plt.figure(figsize=(18, 18))
# sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
# plt.title('Correlation between different fearures')
# file_name = f"DataMining2/graphs/graphHotelSummary/correlation.png"
# plt.savefig(file_name)


# # Undersampling
# Get all clicked on hotels
click_indices = df[df.click_bool == 1].index
# Randomize selection between them
random_indices = np.random.choice(click_indices, len(df.loc[df.click_bool == 1]), replace=False)
# Get all the the regarding them to create a sample of clicked data
click_sample = df.loc[random_indices]

not_click = df[df.click_bool == 0].index
# Get random rows with click_bool=0 and make a vector of size sum click_bool, or len(click_indices)
random_indices = np.random.choice(not_click, sum(df['click_bool']), replace=False)
not_click_sample = df.loc[random_indices]

df_new = pd.concat([not_click_sample, click_sample], axis=0)

# The data is split now 50/50 between clicked and not clicked
print("Number of rows", len(df_new))

# Save to CSV
df_new.to_csv('DataMining2/output/out2.csv', index=False)

mms = MinMaxScaler()
df_new[['price_usd','orig_destination_distance']] = mms.fit_transform(df_new[['price_usd','orig_destination_distance']])

Y = df_new['click_bool']
df_new = df_new.drop(columns=['click_bool', 'booking_bool'], axis=1)
X = df_new

# Split the data into features (X) and target (y)
# X = df.drop('click_bool', axis=1)
# y = df['click_bool']
X_train,X_test, y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# # Featurer selection dfing ANOVA
# fs = SelectKBest(score_func=f_classif, k=10)
# # apply feature selection
# X_selected = fs.fit_transform(X, y)
# print(X_selected)

# # Feature selection dfing tree's
# clf = ExtraTreesClassifier(n_estimators=50)
# clf = clf.fit(X, y)
# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X)
# print(X_new.shape)
# exit()

# # Data check
# print(f"Row count: {df.shape[0]} \n")
# print(f"Column count: {df.shape[1]}")

# sns.countplot(x='booking_bool',data=df)
# file_name = f"DataMining2/graphs/graphHotelSummary/hitBooking.png"
# plt.savefig(file_name)
# print(df['booking_bool'].value_counts())

# sns.countplot(x='click_bool',data=df)
# file_name = f"DataMining2/graphs/graphHotelSummary/histClick.png"
# plt.savefig(file_name)
# print(df['click_bool'].value_counts())
rf =RandomForestClassifier(n_estimators=51,min_samples_leaf=5,min_samples_split=3)


# test_data_X = test_data.drop(columns=['click_bool'], axis=1)
# test_data_y = test_data['click_bool']

def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print('--------- Model : ', trained_model_name, ' ---------------\n')
    predicted_values = trained_model.predict(X_test)
    print(metrics.classification_report(y_test,predicted_values))
    print("Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values))
    print("---------------------------------------\n")

rf.fit(X_train, y_train)
print_evaluation_metrics(rf, "Random forest", X_test, y_test)
