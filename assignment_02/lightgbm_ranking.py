#!/usr/bin/python3.9

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import roc_auc_score

import matplotlib.pylab as plt

"""


#https://github.com/Ransaka/LTR-with-LIghtGBM
#********

import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing,model_selection,metrics
from lightgbm import LGBMRanker

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score,coverage_error


from tabulate import tabulate
from functools import wraps



"""
zipped_data = zipfile.ZipFile("training_set_VU_DM.zip")
zipped_data.namelist()
"""

data_df = pd.read_csv('training_set_VU_DM.csv')

#Creating target column with clicked and booked boolean columns
data_df['superscore'] = data_df['booking_bool'].apply(lambda x: x*4)
data_df['superscore'] = data_df['superscore'] + data_df['click_bool']



#https://stackoverflow.com/questions/15723628/pandas-make-a-column-dtype-object-or-factor
#Selelcting factor featires (categorical)
categorical = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_booking_window', 'srch_saturday_night_bool', 'random_bool', 'gross_bookings_usd']
for factor in categorical:
     data_df[factor] = data_df[factor].astype('category')


#print(list(data_df.columns.values))
#'srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'click_bool', 'gross_bookings_usd', 'booking_bool', 'superscore'
#I am guessing  'click_bool' & 'booking_bool' cannot be used for training

#features = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'gross_bookings_usd']

#temp
#This line creates a new comp values
data_df['comp']=data_df['comp1_rate'].fillna(0)+data_df['comp2_rate'].fillna(0)+data_df['comp3_rate'].fillna(0)+data_df['comp4_rate'].fillna(0)+data_df['comp5_rate'].fillna(0)+data_df['comp6_rate'].fillna(0)+data_df['comp7_rate'].fillna(0)+data_df['comp8_rate'].fillna(0)
features = ['srch_id',  'prop_id', 'comp8_rate', 'comp8_inv', 'price_usd', 'prop_log_historical_price', 'prop_review_score', 'prop_starrating', 'comp']  #NOTE:  'gross_bookings_usd' is not on test set
"""
##parametters with best ourcome so far*******
#n=100
features = ['srch_id',  'prop_id', 'comp8_rate', 'comp8_inv', 'price_usd', 'prop_log_historical_price', 'prop_review_score', 'prop_starrating', 'comp']  #NOTE:  'gross_bookings_usd' is not on test set
"""

features = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp']  


#paper features     - borrowed
#features = ["srch_id",  'prop_id', 'prop_country_id', 'prop_review_score', 'srch_length_of_stay', 'srch_children_count', 'srch_query_affinity_score', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_inv', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff']

#****do min max or log on price and other large features****


Y = data_df['superscore']
X = data_df[features]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)#random_state=1 *****

#define the groups (searches)
get_group_size = lambda df: df.reset_index().groupby("srch_id")['srch_id'].count()
train_groups = get_group_size(x_train)
test_groups = get_group_size(x_test)

print('Preparing model')
n=100    #n=100
model = LGBMRanker(objective="lambdarank",n_estimators=n, force_row_wise=True, seed=1) #The model is seeded*****
model.fit(x_train,y_train,group=train_groups,eval_set=[(x_test,y_test)],eval_group=[test_groups],eval_metric=['map'])

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
test_df = pd.read_csv('test_set_VU_DM.csv')
test_df['comp']=data_df['comp1_rate'].fillna(0)+data_df['comp2_rate'].fillna(0)+data_df['comp3_rate'].fillna(0)+data_df['comp4_rate'].fillna(0)+data_df['comp5_rate'].fillna(0)+data_df['comp6_rate'].fillna(0)+data_df['comp7_rate'].fillna(0)+data_df['comp8_rate'].fillna(0)
"""
categorical = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_booking_window', 'srch_saturday_night_bool', 'random_bool', 'gross_bookings_usd']
"""
#Categorical from above (previous data frame)
# be cause 'gross_bookings_usd' ius not in other datraframe, we have to verify our categories are
for factor in categorical:
     if factor in test_df.columns.values:
        test_df[factor] = test_df[factor].astype('category')

#KeyError: "['gross_bookings_usd'] not in index"
#I think the 'gross_bookings_usd' feature is not on the test set

#features = ['srch_id',  'prop_id', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
X_t = test_df[features]
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

print('Savinf to file')
out_fearures = ['srch_id',  'prop_id']
output_df = ranked[out_fearures]
output_df.to_csv('submission.csv', index=False)

print(len(output_df))








"""
print('------------------------------------------------------------ ')
print('Average nDCG across all queries in the validation at each evaluation point:')
print('------------------------------------------------------------ ')
print('Best nDCG@1:', model.best_score_['valid_0']['ndcg@1'])
print('Best nDCG@5:', model.best_score_['valid_0']['ndcg@5'])
print('Best nDCG@10:', model.best_score_['valid_0']['ndcg@10'])
print('Best nDCG@20:', model.best_score_['valid_0']['ndcg@20'])
print('Best nDCG@38:', model.best_score_['valid_0']['ndcg@38'])  #Higher is better
"""