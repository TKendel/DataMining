import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing,model_selection,metrics
from lightgbm import LGBMRanker, plot_importance

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score,coverage_error

## Pandas options ##
pd.set_option('display.max_columns', None)

print('importing train dataset')
df = pd.read_csv('DataMining2/training_set_VU_DM.csv')

#Creating target column with clicked and booked boolean columns
df['superscore'] = df['booking_bool'].apply(lambda x: x*5)
df['superscore'] = df['superscore'] + df['click_bool']


#print(list(df.columns.values))
#'srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'click_bool', 'gross_bookings_usd', 'booking_bool', 'superscore'
#I am guessing  'click_bool' & 'booking_bool' cannot be used for training


#NEW Featueres
#This line creates a new comp values
# df['comp_rate']=df['comp1_rate'].fillna(0)+df['comp2_rate'].fillna(0)+df['comp3_rate'].fillna(0)+df['comp4_rate'].fillna(0)+df['comp5_rate'].fillna(0)+df['comp6_rate'].fillna(0)+df['comp7_rate'].fillna(0)+df['comp8_rate'].fillna(0)
#df['ratio_booked_vs_clicked'] = df.groupby(by=['prop_id']).sum()['booking_bool'] / df.groupby(by=['prop_id']).sum()['click_bool']
#****add feature comparing number of rooms with number of people staying*******

for i in range(1,9):
    df['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    df['comp'+str(i)+'_rate'] = df['comp'+str(i)+'_rate'].replace({1:10, 0:1, -1:-5})
df['comp_rate_sum'] = df['comp1_rate']
for i in range(1,9):
    df['comp_rate_sum'] += df['comp'+str(i)+'_rate']

for i in range(1,9):
    df['comp'+str(i)+'_inv'].fillna(0, inplace=True)
    df['comp'+str(i)+'_inv'] = df['comp'+str(i)+'_inv'].replace({1:10, 0:1, -1:-5})
df['comp_inv_sum'] = df['comp1_inv']
for i in range(1,9):
    df['comp_inv_sum'] += df['comp'+str(i)+'_inv']

# df.drop(columns=['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 
#             'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 
#             'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 
#             'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 
#             'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'], inplace=True)

#Apply fuynction to every vvalue of column
#https://saturncloud.io/blog/how-to-create-a-new-column-based-on-the-value-of-another-column-in-pandas/#:~:text=Once%20we%20have%20had%20our,and%20return%20a%20new%20DataFrame.
def is_positive(number):
     if number <= 0:
          return 1
     return 0
#Check if input is the same
def same_country(visitor_location_country_id, prop_country_id):
     if visitor_location_country_id == prop_country_id:
          return 1
     return 0
# #rooms and guests
# df['room_minus_guests'] = df['srch_room_count']-(df['srch_adults_count']+df['srch_children_count'])
# df['room_per_person'] = df['room_minus_guests'].apply(is_positive) #is there a room per person
# df['room_guest_ratio'] = (df['srch_adults_count']+df['srch_children_count'])/ df['srch_room_count']#ratio of rooms to guests
# #Normalized price
# df['price_usd_norm'] = df['price_usd']/ df['price_usd'].abs().max()
# #Same country(?)
# #Apply function to two columns --> https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
# df['same_country'] = df.apply(lambda x: same_country(x.visitor_location_country_id, x.prop_country_id), axis=1)


#https://stackoverflow.com/questions/15723628/pandas-make-a-column-dtype-object-or-factor
#Selelcting factor featires (categorical)
# categorical = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_booking_window', 'srch_saturday_night_bool', 'random_bool'] #gross_bookings_usd was here before
categorical = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_saturday_night_bool', 'random_bool'] #gross_bookings_usd was here before

for factor in categorical:
     df[factor] = df[factor].astype('category')


# features = ['srch_id',  'prop_id', 'comp8_rate', 'comp8_inv', 'price_usd', 'prop_log_historical_price', 'prop_review_score', 'prop_starrating', 'comp']  #NOTE:  'gross_bookings_usd' is not on test set
"""
##parametters with best ourcome so far*******
#n=100
features = ['srch_id',  'prop_id', 'comp8_rate', 'comp8_inv', 'price_usd', 'prop_log_historical_price', 'prop_review_score', 'prop_starrating', 'comp']  #NOTE:  'gross_bookings_usd' is not on test set
"""

features = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 
            'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 
            'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 
            'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 
            'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 
            'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 
            'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 
            'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 
            'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 
            'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 
            'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']  

numeric_features = ['orig_destination_distance', 'prop_location_score1', 'prop_location_score2']

categorical_a = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_saturday_night_bool', 'random_bool'] #gross_bookings_usd was here before

df['prop_id'] = df['prop_id'].astype('category')

test = df.copy()
# test.drop(columns=categorical_a, inplace=True)
# test.drop(columns=['date_time'], inplace=True)
# tf = ['prop_id', 'orig_destination_distance', 'prop_location_score1', 'prop_location_score2']
# test = test[tf]
# for num_feat in numeric_features:
#     df[f'{num_feat}_avg'] = test.groupby(by=['prop_id']).mean()[num_feat]
#     # df[f'{num_feat}_std'] = test.groupby(by=['prop_id']).std()[num_feat]
#     # df[f'{num_feat}_median'] = test.groupby(by=['prop_id']).median()[num_feat]

df.drop(columns=['position','date_time', 'booking_bool', 'click_bool', 'visitor_hist_starrating', 'gross_bookings_usd'] , inplace=True)

Y = df['superscore']
X = df.drop('superscore', axis=1)

#****also try using the entire data set to train********
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle=False)#random_state=1 *****    #test_size=0.2 test_size=0.0001

#define the groups (searches)
get_group_size = lambda df: df.reset_index().groupby("srch_id")['srch_id'].count()
train_groups = get_group_size(x_train)
test_groups = get_group_size(x_test)

print('Preparing model')
model = LGBMRanker(objective="lambdarank",
                   n_estimators=200,
                   learning_rate=0.05,
                   metric='ndcg'
                   ) #The model is seeded*****   max_bin=255#max_bin=500
model.fit(x_train,y_train,group=train_groups,eval_set=[(x_test,y_test)],eval_group=[test_groups],eval_metric=['map'])

# plot_importance(model, figsize = (12,8))
# plt.show()

#getting this error 
"""
[LightGBM] [Info] Total groups: 199793, total data: 3966677
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.020097 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 409
[LightGBM] [Info] Number of data points in the train set: 3966677, number of used features: 5
[LightGBM] [Info] Total groups: 193144, total data: 991670
"""
#Seems to be memory related, Fixed by adding force_row_wise=True to the model input


#Predictions on test dataset
print("RANKED")

print('importing test dataset')
test_df = pd.read_csv('DataMining2/test_set_VU_DM.csv')
#extra fweatures
#******This was the previous model, NOTE, it is wrong, as it is using the =df['comp1_rate'] instead of the test_df
#test_df['comp']=df['comp1_rate'].fillna(0)+df['comp2_rate'].fillna(0)+df['comp3_rate'].fillna(0)+df['comp4_rate'].fillna(0)+df['comp5_rate'].fillna(0)+df['comp6_rate'].fillna(0)+df['comp7_rate'].fillna(0)+df['comp8_rate'].fillna(0)
# test_df['comp']=test_df['comp1_rate'].fillna(0)+test_df['comp2_rate'].fillna(0)+test_df['comp3_rate'].fillna(0)+test_df['comp4_rate'].fillna(0)+test_df['comp5_rate'].fillna(0)+test_df['comp6_rate'].fillna(0)+test_df['comp7_rate'].fillna(0)+test_df['comp8_rate'].fillna(0)

for i in range(1,9):
    test_df['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    test_df['comp'+str(i)+'_rate'] = test_df['comp'+str(i)+'_rate'].replace({1:10, 0:1, -1:-5})
test_df['comp_rate_sum'] = test_df['comp1_rate']
for i in range(1,9):
    test_df['comp_rate_sum'] += test_df['comp'+str(i)+'_rate']

for i in range(1,9):
    test_df['comp'+str(i)+'_inv'].fillna(0, inplace=True)
    # test_df['comp'+str(i)+'_inv'] = test_df['comp'+str(i)+'_inv'].replace({1:10, 0:1, -1:-5})
test_df['comp_inv_sum'] = test_df['comp1_inv']
for i in range(1,9):
    test_df['comp_inv_sum'] += test_df['comp'+str(i)+'_inv']

# Remove null values
test_df['prop_review_score'].fillna((test_df['prop_review_score'].mean()), inplace=True)
test_df["prop_location_score2"].fillna((test_df['prop_location_score2'].mean()), inplace=True)
test_df['orig_destination_distance'].fillna((test_df['orig_destination_distance'].median()), inplace=True)

# test_df.drop(columns=['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 
#             'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 
#             'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 
#             'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 
#             'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'], inplace=True)


# test_df['room_minus_guests'] = test_df['srch_room_count']-(test_df['srch_adults_count']+test_df['srch_children_count'])
# test_df['room_per_person'] = test_df['room_minus_guests'].apply(is_positive) #is there a room per person
# test_df['room_guest_ratio'] = (test_df['srch_adults_count']+test_df['srch_children_count'])/ test_df['srch_room_count']#ratio of rooms to guests
# test_df['price_usd_norm'] = test_df['price_usd']/ test_df['price_usd'].abs().max()
# test_df['same_country'] = test_df.apply(lambda x: same_country(x.visitor_location_country_id, x.prop_country_id), axis=1)


#maybe this is not possible, as the test dataset does not have this information
#test_df['ratio_booked_vs_clicked'] = test_df.groupby(by=['prop_id']).sum()['booking_bool'] / test_df.groupby(by=['prop_id']).sum()['click_bool']


"""
categorical = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_booking_window', 'srch_saturday_night_bool', 'random_bool', 'gross_bookings_usd']
"""
#Categorical from above (previous data frame)
# be cause 'gross_bookings_usd' ius not in other datraframe, we have to verify our categories are
for factor in categorical:
     if factor in test_df.columns.values:
        test_df[factor] = test_df[factor].astype('category')

test_df['prop_id'] = test_df['prop_id'].astype('category')

test = test_df.copy()
# test.drop(columns=categorical_a, inplace=True)
# test.drop(columns=['date_time'], inplace=True)
# tf = ['prop_id', 'orig_destination_distance', 'prop_location_score1', 'prop_location_score2']
# test = test[tf]
# for num_feat in numeric_features:
#     test_df[f'{num_feat}_avg'] = test.groupby(by=['prop_id']).mean()[num_feat]
#     # test_df[f'{num_feat}_std'] = test.groupby(by=['prop_id']).std()[num_feat]
#     # test_df[f'{num_feat}_median'] = test.groupby(by=['prop_id']).median()[num_feat]

test_df.drop(columns=['date_time', 'visitor_hist_starrating'] , inplace=True)

test_df['prop_id'] = test_df['prop_id'].astype('category')


#KeyError: "['gross_bookings_usd'] not in index"
#I think the 'gross_bookings_usd' feature is not on the test set

#features = ['srch_id',  'prop_id', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
X_t = test_df

#Full Data***
print('Making poredictions')
predictions = model.predict(X_t)

dic = {
     'srch_id' : X_t['srch_id'],
     'prop_id' : X_t['prop_id'],
     'score' : predictions
}


ranked = pd.DataFrame(dic)

ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print("RANKED")
print(ranked)

print('Saving to file')
out_fearures = ['srch_id',  'prop_id']
output_df = ranked[out_fearures]
output_df.to_csv('submission.csv', index=False)

print(len(output_df))

print('------------------------------------------------------------ ')
print('Average nDCG across all queries in the validation at each evaluation point:')
print('------------------------------------------------------------ ')
print('Best nDCG@1:', model.best_score_['valid_0']['ndcg@1'])
print('Best nDCG@5:', model.best_score_['valid_0']['ndcg@5'])
