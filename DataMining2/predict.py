import pandas as pd
import numpy as np
import seaborn as sns
import random
import pickle

from sklearn.preprocessing import MinMaxScaler



test_data = pd.read_csv("DataMining2/test_set_VU_DM.csv", index_col=0)
df_test = pd.DataFrame(test_data).reset_index()


print(df_test.head())
# Remove null values
def randomiseMissingData(df2):
    "randomise missing data for DataFrame (within a column)"
    df = df2.copy()
    for col in df.columns:
        data = df['prop_review_score']
        mask = data.isnull()
        samples = random.choices( data[~mask].values , k = mask.sum() )
        data[mask] = samples
    return df
df_test = randomiseMissingData(df_test)
df_test["prop_location_score2"].fillna(0, inplace=True)
df_test['srch_query_affinity_score'].fillna((df_test['srch_query_affinity_score'].mean()), inplace=True)
df_test['orig_destination_distance'].fillna((df_test['orig_destination_distance'].median()), inplace=True)
df_test["visitor_hist_adr_usd"].fillna(0, inplace=True)
# df_test['visitor_hist_starrating_bool'] = pd.notnull(df_test['visitor_hist_starrating'])


for i in range(1,9):
    df_test['comp'+str(i)+'_rate'].fillna(0, inplace=True)
df_test['comp_rate_sum'] = df_test['comp1_rate']
for i in range(2,9):
    df_test['comp_rate_sum'] += df_test['comp'+str(i)+'_rate']

for i in range(1,9):
    df_test['comp'+str(i)+'_inv'].fillna(0, inplace=True)
    df_test['comp'+str(i)+'_inv'][df_test['comp'+str(i)+'_inv']==1] = 10
    df_test['comp'+str(i)+'_inv'][df_test['comp'+str(i)+'_inv']==-1] = 1
    df_test['comp'+str(i)+'_inv'][df_test['comp'+str(i)+'_inv']==0] = -1
    df_test['comp'+str(i)+'_inv'][df_test['comp'+str(i)+'_inv']==10] = 0
df_test['comp_inv_sum'] = df_test['comp1_inv']
for i in range(2,9):
    df_test['comp_inv_sum'] += df_test['comp'+str(i)+'_inv']

# mms = MinMaxScaler()
# df_test[['price_usd','orig_destination_distance']] = mms.fit_transform(df_test[['price_usd','orig_destination_distance']])

# #Selelcting factor featires (categorical)
# categorical = ['prop_id']
# for factor in categorical:
#     df_test[factor] = df_test[factor].astype('category')

print(df_test.isnull().sum())

print(df_test.head())
test = df_test.copy()
test.drop(columns=test.columns[26:49], inplace=True)
test.drop(columns=['date_time', 'comp1_rate'], inplace=True)

with open("GBC.pkl", "rb") as f:
    GBC = pickle.load(f)

print('Making poredictions')

predictions = GBC.predict_proba(test.values)[:,1]

dic = {
     'srch_id' : test['srch_id'],
     'prop_id' : test['prop_id'],
     'score' : predictions
}


ranked = pd.DataFrame(dic)

ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print("RANKED")
print(ranked)

print('Savinf to file')
out_fearures = ['srch_id',  'prop_id']
output_df = ranked[out_fearures]
output_df.to_csv('submission.csv', index=False)

print(len(output_df))
exit()


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
# df_new.to_csv('DataMining2/output/out2.csv', index=False)

mms = MinMaxScaler()
df_new[['price_usd','orig_destination_distance']] = mms.fit_transform(df_new[['price_usd','orig_destination_distance']])

Y = df_new['click_bool']
df_new = df_new.drop(columns=['click_bool'], axis=1)
X = df_new

# Split the data into features (X) and target (y)
# X = df.drop('click_bool', axis=1)
# y = df['click_bool']
X_df,X_test, y_df,y_test = df_test_split(X, Y, test_size=0.3, random_state=1)

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
