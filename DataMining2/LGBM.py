import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import pickle as pk

from matplotlib import pyplot as plt
# ANOVA feature selection for numeric input and categorical output
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import ndcg_score



## Load data into data frame ##
data = pd.read_csv("training_set_VU_DM_cleaned.csv", index_col=0)
df = pd.DataFrame(data)


print(f"Row count: {df.shape[0]} \n")
print(f"Column count: {df.shape[1]}")



# Drop columns
# dropped_variables = ["date_time", "visitor_hist_adr_usd","gross_bookings_usd","visitor_hist_starrating","visitor_location_country_id",
#                      "srch_query_affinity_score","srch_query_affinity_score","comp1_rate","comp1_inv","comp1_rate_percent_diff","comp2_rate",
#                      "comp2_inv","comp2_rate_percent_diff","comp3_rate","comp3_inv","comp3_rate_percent_diff","comp4_rate","comp4_inv",
#                      "comp4_rate_percent_diff","comp5_rate","comp5_inv","comp5_rate_percent_diff","comp6_rate","comp6_inv",
#                      "comp6_rate_percent_diff","comp7_rate","comp7_inv","comp7_rate_percent_diff","comp8_rate","comp8_inv",
#                      "comp8_rate_percent_diff"]



dropped_variables = ["date_time","position","gross_bookings_usd","comp1_rate","comp1_inv","comp1_rate_percent_diff","comp2_rate",
                     "comp2_inv","comp2_rate_percent_diff","comp3_rate","comp3_inv","comp3_rate_percent_diff","comp4_rate","comp4_inv",
                     "comp4_rate_percent_diff","comp5_rate","comp5_inv","comp5_rate_percent_diff","comp6_rate","comp6_inv",
                     "comp6_rate_percent_diff","comp7_rate","comp7_inv","comp7_rate_percent_diff","comp8_rate","comp8_inv",
                     "comp8_rate_percent_diff"]




df.drop(columns=dropped_variables, inplace=True)


df_1 = df.copy()


# Index value counts
index_counts = df_1.index.value_counts()
print("Number of rows for each unique index value:")
print(index_counts)
print("Length of unique index values:", len(index_counts))


# Group df_1 by search_id
df_1_grouped= df_1.groupby('srch_id')


#print number of srch_id with booking_bool = 1
# print("Number of srch_id with booking_bool = 1")
# print(df_1_grouped['booking_bool'].sum().value_counts())

# Function to identify if any row in a group has booking_bool equal to 1

# def has_booking(group):
#     return group['booking_bool'].max() == 1

# # Separate groups into two datasets based on booking
# booked_data = df_1_grouped.filter(has_booking)
# not_booked_data = df_1_grouped.filter(lambda x: not has_booking(x))

# # Print the size of each dataset
# print("Number of rows in booked dataset:", len(booked_data))
# print("Number of rows in not booked dataset:", len(not_booked_data))




### LIGHT GBM ranker


# Step 0:

Y = df_1['click_bool']
df_new = df_1.drop(columns=['click_bool', 'booking_bool'], axis=1)
X = df_new



# Step 1: Normalize the data. Check the variables in X_train and X_test

# Step 2: Split training data into training and test data
          #split train test considering grouping by srch_id


splitter = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=7)
split = splitter.split(X, groups=X.index)
train_inds, test_inds = next(split)

# Further split X_train into X_train and X_val
X_train = X.iloc[train_inds]
X_test = X.iloc[test_inds]
y_train = Y.iloc[train_inds]
y_test = Y.iloc[test_inds]

# Convert labels to integers (Revelance of the search: 0 or 1, where 1 is the most relevant)
y_test = y_test.astype(int)
y_train = y_train.astype(int)


# Step 2.1.: specify the different train groups and size of each group in query_train and query_test.
    # lgm.LGBMRanker requires the size of each group in the training and test data to be specified.

query_train = X_train.index.value_counts()
query_test = X_test.index.value_counts()

# # Convert queries to a list
group_train = query_train.tolist()
group_test = query_test.tolist()



# Step 3: Define my rankmodel
    # Define a basic LGBMRanker model with default hyperparameters
def train_basic():
    gbm = lgb.LGBMRanker(objective='lambdarank',
                        n_estimators=1000,
                        random_state=42,
                        num_leaves=41,
                        learning_rate=0.002,
                        colsample_bytree=0.7,
                        n_jobs=2)
    return gbm



gbm = train_basic()




# Step 4: Hyperparameter tuning  ----> NEED TO WORK ON THIS!!!!!!!!!

# Create grid of hyperparameters
# hyperparameters = {
#     'n_estimators': [1000, 2000, 3000],
#     'num_leaves': [31, 41, 51],
#     'learning_rate': [0.002, 0.005, 0.01],
#     'colsample_bytree': [0.7, 0.8, 0.9]
# }


# use GridSearchCV to find the best hyperparameters
# from sklearn.model_selection import GridSearchCV

# # Create the GridSearchCV object
# grid_search = GridSearchCV(gbm, hyperparameters, scoring='neg_log_loss', cv=3, n_jobs=-1)

# bm = grid_search.best_estimator_

# # Fit the model
# grid_search.fit(X_train, y_train, group=group_train, eval_set=[(X_val, y_val)], eval_group=[group_val], eval_at=[1, 5, 10, 20, 38], eval_metric='ndcg')

# # Get the best hyperparameters
# best_hyperparameters = grid_search.best_params_
# print('Best hyperparameters:', best_hyperparameters)





# Step 5: Train the model
gbm.fit(X_train, y_train, group=group_train,
        eval_set=[(X_test, y_test)], eval_group=[group_test],
        eval_at=[1, 5, 10, 20, 38], eval_metric='ndcg') 

# Performance of the model at each evaluation point (Best score at each evaluation point)
# NDCG score based on the entire ranked list of items generated in each query.


print('------------------------------------------------------------ ')
print('Average nDCG across all queries in the validation at each evaluation point:')
print('------------------------------------------------------------ ')
print('Best nDCG@1:', gbm.best_score_['valid_0']['ndcg@1'])
print('Best nDCG@5:', gbm.best_score_['valid_0']['ndcg@5'])
print('Best nDCG@10:', gbm.best_score_['valid_0']['ndcg@10'])
print('Best nDCG@20:', gbm.best_score_['valid_0']['ndcg@20'])
print('Best nDCG@38:', gbm.best_score_['valid_0']['ndcg@38'])  #Higher is better


### INTERPRETATION OF RESULTS
    #The results printed represent the best NDCG (Normalized Discounted Cumulative Gain) 
    #scores achieved by the model on the validation set at different evaluation points, which
    #are specified by the eval_at parameter during model training.

    #Best nDCG@1: This is the best NDCG score achieved when evaluating the model's predictions at
    #the top position for each query (in this case, the top 1 position).
    #Best nDCG@5: This represents the best NDCG score at the top 5 positions for each query.
    #Best nDCG@10: Similarly, this is the best NDCG score at the top 10 positions for each query.

    #Conclusion: Specifically of eval_at=5
    # Calculate the NDCG@5 for each group separately, and then choose the highest NDCG@5 score among the groups
    # as the "Best nDCG@5" reported.
    #The best nDCG@5 represents the highest NDCG score achieved among all queries when evaluating the
    # model's predictions at the top 5 positions for each query.
###



# Save the model as a pickle file
pk.dump(gbm, open("gbm_model.pkl", "wb"))

# Open model from pickle file
# Open saved model using 2nd method
model = pk.load(open("gbm_model.pkl", "rb"))



# Step 6: Predict and get the NDCG scores for each group in the test set (test is validation here)


predictions = []
ndcg_scores_k10 = [] # Store the NDCG scores for each group
ndcg_scores_k30 = []

for group in np.unique(X_test.index):
    y_preds_group = model.predict(X_test[X_test.index == group])
    y_true_group = y_test[y_test.index == group].values.flatten()
    predictions.append(y_preds_group)
    ndcg_k10 = ndcg_score([y_true_group], [y_preds_group], k=10)  # Specify k as needed
    ndcg_scores_k10.append(ndcg_k10)

    ndcg_k30 = ndcg_score([y_true_group], [y_preds_group], k=30)  # Specify k as needed
    ndcg_scores_k30.append(ndcg_k30)


# Step 7: Evaluate the model

    # Compute the average NDCG score across all query groups.
average_ndcg_k10 = np.mean(ndcg_scores_k10)
average_ndcg_k30 = np.mean(ndcg_scores_k30)
print('----- # Ranking quality of the model -----')
print('Average performance of ranking algorithm across all queries - NDCG score with k=10:', average_ndcg_k10)
print('Average performance of ranking algorithm across all queries - NDCG score with k=30:', average_ndcg_k30)




# ### TEST SET ###

# Score test_set_VU_DM.csv
test_data = pd.read_csv("test_set_VU_DM.csv", index_col=0)
df_test = pd.DataFrame(test_data)

# We need same columns features as in training set
# if column name in df_test in variable dropped_variables, drop it
for column in dropped_variables:
    if column in df_test.columns:
        df_test.drop(columns=column, inplace=True)

X_test = df_test
predict = []

y_preds_group = model.predict(X_test)
predict.append(y_preds_group)



# Sort by src_id first and then the score ascending order

df_test['score'] = predict[0]
grouped = df_test.groupby('srch_id')
df_test_new = df_test.sort_values(['srch_id', 'score'], ascending=[True, False])  # Sort by 'srch_id' first
final_df = df_test_new.groupby('srch_id')['prop_id'].apply(list).reset_index()

# rop_id in final_df is a list. 
# We need to explode the list to get the final output format

final_df_exploded = final_df.explode('prop_id')
final_df_exploded['output'] = final_df_exploded.apply(lambda x: f"{x['srch_id']},{x['prop_id']}", axis=1)
output_str = "\n".join(final_df_exploded['output'])

# final dataframe with only srch_id and prop_id
final_df_two_columns = final_df_exploded[['srch_id', 'prop_id']]
print(final_df_two_columns.head())


final_df_two_columns.to_csv('submission.csv', index=False)
#print len of group by srch_id in test set



# print(final_df[final_df.index == 1]['prop_id'])

