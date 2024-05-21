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



print('Importing train dataset')
data_df = pd.read_csv('training_set_VU_DM.csv')

#Creating target column with clicked and booked boolean columns
data_df['superscore'] = data_df['booking_bool'].apply(lambda x: x*4)
data_df['superscore'] = data_df['superscore'] + data_df['click_bool']

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

print('Preparing new features')
#NEW Features
#This line creates a new comp values
data_df['comp']=data_df['comp1_rate'].fillna(0)+data_df['comp2_rate'].fillna(0)+data_df['comp3_rate'].fillna(0)+data_df['comp4_rate'].fillna(0)+data_df['comp5_rate'].fillna(0)+data_df['comp6_rate'].fillna(0)+data_df['comp7_rate'].fillna(0)+data_df['comp8_rate'].fillna(0)
data_df['inv']=data_df['comp1_inv'].fillna(0)+data_df['comp2_inv'].fillna(0)+data_df['comp3_inv'].fillna(0)+data_df['comp4_inv'].fillna(0)+data_df['comp5_inv'].fillna(0)+data_df['comp6_inv'].fillna(0)+data_df['comp7_inv'].fillna(0)+data_df['comp8_inv'].fillna(0)
#rooms and guests
data_df['room_minus_guests'] = data_df['srch_room_count']-(data_df['srch_adults_count']+data_df['srch_children_count'])
data_df['room_per_person'] = data_df['room_minus_guests'].apply(is_positive) #is there a room per person
data_df['room_guest_ratio'] = (data_df['srch_adults_count']+data_df['srch_children_count'])/ data_df['srch_room_count']#ratio of rooms to guests
#Normalized price
data_df['price_usd_norm'] = data_df['price_usd']/ data_df['price_usd'].abs().max()
#Same country(?)
#Apply function to two columns --> https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
data_df['same_country'] = data_df.apply(lambda x: same_country(x.visitor_location_country_id, x.prop_country_id), axis=1)
#https://stackoverflow.com/questions/15723628/pandas-make-a-column-dtype-object-or-factor

#New target values
#Ratios don0't work because we need labels,
#we are noirmalizing and then creating cattegories by shifting the values by multiplying by 10. as in .123 --> 1.23 and clipping the decimals with round()
#data_df['ratio_booked_vs_clicked'] = data_df.groupby(by=['prop_id']).sum()['booking_bool'] / data_df.groupby(by=['prop_id']).sum()['click_bool']
#sums to be used in the ratios
data_df['booked'] = data_df.groupby('prop_id')['booking_bool'].transform('sum') #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transform.html
data_df['booked'] =data_df['booked'].fillna(0)
data_df['clicked'] = data_df.groupby('prop_id')['click_bool'].transform('sum')
data_df['clicked'] =data_df['clicked'].fillna(0)
data_df['superscore_sum'] = data_df.groupby('prop_id')['superscore'].transform('sum')
data_df['count'] = data_df.groupby('prop_id')['prop_id'].transform('count')
#ratio_booked_vs_clicked
data_df['ratio_booked_vs_clicked'] = data_df['booked'] / data_df['clicked']     #create ratio
data_df['lbl_ratio_booked_vs_clicked'] = data_df['ratio_booked_vs_clicked']/ data_df['ratio_booked_vs_clicked'].abs().max()  #Normalize data
data_df['lbl_ratio_booked_vs_clicked'] = data_df['lbl_ratio_booked_vs_clicked'].fillna(0)  
data_df['lbl_ratio_booked_vs_clicked'] = data_df['lbl_ratio_booked_vs_clicked'].apply(lambda x: int(round(x*10,0)))     #Shift one deciml point, remove decimals, and rounf to interger
#ratio_booked_vs_count
data_df['ratio_booked_vs_count'] = data_df['booked'] / data_df['count']     #create ratio
data_df['lbl_ratio_booked_vs_count'] = data_df['ratio_booked_vs_count']/ data_df['ratio_booked_vs_count'].abs().max()  #Normalize data
data_df['lbl_ratio_booked_vs_count'] = data_df['lbl_ratio_booked_vs_count'].apply(lambda x: int(round(x*10,0)))     #Shift one deciml point, remove decimals, and rounf to interger
#ratio_clicked_vs_count
data_df['ratio_clicked_vs_count'] = data_df['clicked'] / data_df['count']     #create ratio
data_df['lbl_ratio_clicked_vs_count'] = data_df['ratio_clicked_vs_count']/ data_df['ratio_clicked_vs_count'].abs().max()  #Normalize data
data_df['lbl_ratio_clicked_vs_count'] = data_df['lbl_ratio_clicked_vs_count'].apply(lambda x: int(round(x*10,0)))     #Shift one deciml point, remove decimals, and rounf to interger
#ratio_superscore_vs_count
data_df['ratio_superscore_vs_count'] = data_df['superscore_sum'] / data_df['count']     #create ratio
data_df['lbl_ratio_superscore_vs_count'] = data_df['ratio_superscore_vs_count']/ data_df['ratio_superscore_vs_count'].abs().max()  #Normalize data
data_df['lbl_ratio_superscore_vs_count'] = data_df['lbl_ratio_superscore_vs_count'].apply(lambda x: int(round(x*10,0)))     #Shift one deciml point, remove decimals, and rounf to interger

#Selelcting factor featires (categorical)
categorical = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_brand_bool', 'promotion_flag', 'srch_destination_id', 'srch_booking_window', 'srch_saturday_night_bool', 'random_bool', 'room_per_person', 'same_country'] #gross_bookings_usd was here before
for factor in categorical:
     data_df[factor] = data_df[factor].astype('category')

#****also try using the entire data set to train********

#*****add some to have comp6_rate null = 0*******

features1 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features2 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features3 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features4 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features5 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features6 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features7 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features8 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  

features9 = ['srch_id',  'prop_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']  


print('Preparing training datasets')
#Function to get ggrgouop sizes for processing
get_group_size = lambda df: df.reset_index().groupby("srch_id")['srch_id'].count()

#Data for model 1
Y1 = data_df['superscore']
X1 = data_df[features1]
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups1 = get_group_size(x_train1)
test_groups1 = get_group_size(x_test1)
#Data for model 2
Y2 = data_df['superscore'].iloc[::-1]
X2 = data_df[features2].iloc[::-1]
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.1, shuffle=False)#random_state=1 *****
train_groups2 = get_group_size(x_train2)
test_groups2 = get_group_size(x_test2)
X2 = data_df[features2] #For the inverse lists, I need to set them to the originbal order for when cumulative is calculated
#Data for model 3
Y3 = data_df['lbl_ratio_booked_vs_clicked']
X3 = data_df[features3]
x_train3, x_test3, y_train3, y_test3 = train_test_split(X3, Y3, test_size=0.2, shuffle=False)#random_state=1 *****
train_groups3 = get_group_size(x_train3)
test_groups3 = get_group_size(x_test3)
#Data for model 4
Y4 = data_df['superscore'].iloc[::-1]    #https://stackoverflow.com/questions/20444087/right-way-to-reverse-a-pandas-dataframe
X4 = data_df[features4].iloc[::-1]  #Should invert the list
x_train4, x_test4, y_train4, y_test4 = train_test_split(X4, Y4, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups4 = get_group_size(x_train4)
test_groups4 = get_group_size(x_test4)
X4 = data_df[features4] #For the inverse lists, I need to set them to the originbal order for when cumulative is calculated
#Data for model 5
Y5 = data_df['click_bool']
X5 = data_df[features5]
x_train5, x_test5, y_train5, y_test5 = train_test_split(X5, Y5, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups5 = get_group_size(x_train5)
test_groups5 = get_group_size(x_test5)
#Data for model 6
Y6 = data_df['booking_bool']
X6 = data_df[features6]
x_train6, x_test6, y_train6, y_test6 = train_test_split(X6, Y6, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups6 = get_group_size(x_train6)
test_groups6 = get_group_size(x_test6)
#Data for model 7
Y7 = data_df['lbl_ratio_booked_vs_count'].iloc[::-1]
X7 = data_df[features7].iloc[::-1]
x_train7, x_test7, y_train7, y_test7 = train_test_split(X7, Y7, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups7 = get_group_size(x_train7)
test_groups7 = get_group_size(x_test7)
X7 = data_df[features7]  #For the inverse lists, I need to set them to the originbal order for when cumulative is calculated
#Data for model 8
Y8 = data_df['lbl_ratio_clicked_vs_count']
X8 = data_df[features8]
x_train8, x_test8, y_train8, y_test8 = train_test_split(X8, Y8, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups8 = get_group_size(x_train8)
test_groups8 = get_group_size(x_test8)
#Data for model 9
Y9 = data_df['lbl_ratio_superscore_vs_count']
X9 = data_df[features9]
x_train9, x_test9, y_train9, y_test9 = train_test_split(X9, Y9, test_size=0.001, shuffle=False)#random_state=1 *****
train_groups9 = get_group_size(x_train9)
test_groups9 = get_group_size(x_test9)


print('Preparing to train models')

seed=17
out_fearures = ['srch_id',  'prop_id']

print('------------------------------------------------------------ ')
print('Training model 1')
model1 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model1.fit(x_train1,y_train1,group=train_groups1,eval_set=[(x_test1,y_test1)],eval_group=[test_groups1],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions1=model1.predict(X1)
print('Best nDCG@5:', model1.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions1}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model1 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model1.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 2')
model2 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model2.fit(x_train2,y_train2,group=train_groups2,eval_set=[(x_test2,y_test2)],eval_group=[test_groups2],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions2=model2.predict(X2)
print('Best nDCG@5:', model2.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions2}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model2 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model2.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 3')
model3 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=255, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model3.fit(x_train3,y_train3,group=train_groups3,eval_set=[(x_test3,y_test3)],eval_group=[test_groups3],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions3=model3.predict(X3)
print('Best nDCG@5:', model3.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions3}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model3 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model3.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 4')
model4 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model4.fit(x_train4,y_train4,group=train_groups4,eval_set=[(x_test4,y_test4)],eval_group=[test_groups4],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions4=model4.predict(X4)
print('Best nDCG@5:', model4.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions4}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model4 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model4.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 5')
model5 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model5.fit(x_train5,y_train5,group=train_groups5,eval_set=[(x_test5,y_test5)],eval_group=[test_groups5],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions5=model5.predict(X5)
print('Best nDCG@5:', model5.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions5}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model5 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model5.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 6')
model6 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model6.fit(x_train6,y_train6,group=train_groups6,eval_set=[(x_test6,y_test6)],eval_group=[test_groups6],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions6=model6.predict(X6)
print('Best nDCG@5:', model6.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions6}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model6 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model6.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 7')
model7 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model7.fit(x_train7,y_train7,group=train_groups7,eval_set=[(x_test7,y_test7)],eval_group=[test_groups7],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions7=model7.predict(X4)
print('Best nDCG@5:', model7.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions7}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model7 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model7.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 8')
model8 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model8.fit(x_train8,y_train8,group=train_groups8,eval_set=[(x_test8,y_test8)],eval_group=[test_groups8],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions8=model8.predict(X5)
print('Best nDCG@5:', model8.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions8}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model8 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model8.csv', index=False)
print('------------------------------------------------------------ ')

print('Training model 9')
model9 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model9.fit(x_train9,y_train9,group=train_groups9,eval_set=[(x_test9,y_test9)],eval_group=[test_groups9],eval_metric=['map'])
print('Calculating predictions')
train_prediuctions9=model9.predict(X6)
print('Best nDCG@5:', model9.best_score_['valid_0']['ndcg@5'])
# dic = {'srch_id' : data_df['srch_id'], 'prop_id' : data_df['prop_id'], 'score' : train_prediuctions9}
# ranked = pd.DataFrame(dic)
# ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
# print('Saving model9 to file')
# output_df = ranked[out_fearures]
# output_df.to_csv('multiple_lightgbm_ranking/submission_model9.csv', index=False)
print('------------------------------------------------------------ ')



print('Creating prediction dataframe')
dic = {
     'srch_id' : data_df['srch_id'],
     'prop_id' : data_df['prop_id'],
     'predictions1' : train_prediuctions1,
     'predictions2' : train_prediuctions2,
     'predictions3' : train_prediuctions3,
     'predictions4' : train_prediuctions4,
     'predictions5' : train_prediuctions5,
     'predictions6' : train_prediuctions6,
     'predictions7' : train_prediuctions7,
     'predictions8' : train_prediuctions8,
     'predictions9' : train_prediuctions9
}
features_cum=['srch_id', 'prop_id', 'predictions1','predictions2','predictions3','predictions4','predictions5','predictions6','predictions7','predictions8','predictions9']
prediction_df = pd.DataFrame(dic)

print('Preparing cumulative training dataset')
#Data for cumulative model
Ycum = data_df['superscore']
Xcum = prediction_df[features_cum]
x_train_cum, x_test_cum, y_train_cum, y_test_cum = train_test_split(Xcum, Ycum, test_size=0.01, shuffle=False)#random_state=1 *****   # test_size=0.001
train_groups_cum = get_group_size(x_train_cum)
test_groups_cum = get_group_size(x_test_cum)

print('Training cumulative model')
print('------------------------------------------------------------ ')
model_cumulative = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model_cumulative.fit(x_train_cum,y_train_cum,group=train_groups_cum,eval_set=[(x_test_cum,y_test_cum)],eval_group=[test_groups_cum],eval_metric=['map'])
print('Best nDCG@5:', model_cumulative.best_score_['valid_0']['ndcg@5'])
print('------------------------------------------------------------ ')


features_cum2 = ['srch_id', 'prop_id', 'predictions1','predictions2','predictions3','predictions4','predictions5','predictions6','predictions7','predictions8','predictions9','visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'comp', 'inv', 'room_minus_guests', 'room_per_person', 'room_guest_ratio', 'price_usd_norm', 'same_country']


print('Concatenating data frames for training dataset 2')
#frames = [prediction_df.reset_index(), data_df.reset_index()]
#cumulative2_df = pd.concat(frames, axis=1)    #https://pandas.pydata.org/docs/user_guide/merging.html
cumulative2_df=data_df
cumulative2_df['predictions1']=prediction_df['predictions1']
cumulative2_df['predictions2']=prediction_df['predictions2']
cumulative2_df['predictions3']=prediction_df['predictions3']
cumulative2_df['predictions4']=prediction_df['predictions4']
cumulative2_df['predictions5']=prediction_df['predictions5']
cumulative2_df['predictions6']=prediction_df['predictions6']
cumulative2_df['predictions7']=prediction_df['predictions7']
cumulative2_df['predictions8']=prediction_df['predictions8']
cumulative2_df['predictions9']=prediction_df['predictions9']



print('Preparing cumulative training dataset 2')
#Data for cumulative model
Ycum2 = data_df['superscore']
Xcum2 = cumulative2_df[features_cum2]
x_train_cum2, x_test_cum2, y_train_cum2, y_test_cum2 = train_test_split(Xcum2, Ycum2, test_size=0.01, shuffle=False)#random_state=1 *****   # test_size=0.001
train_groups_cum2 = get_group_size(x_train_cum2)
test_groups_cum2 = get_group_size(x_test_cum2)

print('Training cumulative model2')
print('------------------------------------------------------------ ')
model_cumulative2 = LGBMRanker(objective="lambdarank",n_estimators=100, force_row_wise=True, seed=seed, max_bin=500, learning_rate=0.1) #The model is seeded*****   max_bin=255#max_bin=500
model_cumulative2.fit(x_train_cum2,y_train_cum2,group=train_groups_cum2,eval_set=[(x_test_cum2,y_test_cum2)],eval_group=[test_groups_cum2],eval_metric=['map'])
print('Best nDCG@5:', model_cumulative2.best_score_['valid_0']['ndcg@5'])
print('------------------------------------------------------------ ')




print('Importing test dataset')
test_df = pd.read_csv('test_set_VU_DM.csv')
print('Preparing fetures')
#extra features
test_df['comp']=test_df['comp1_rate'].fillna(0)+test_df['comp2_rate'].fillna(0)+test_df['comp3_rate'].fillna(0)+test_df['comp4_rate'].fillna(0)+test_df['comp5_rate'].fillna(0)+test_df['comp6_rate'].fillna(0)+test_df['comp7_rate'].fillna(0)+test_df['comp8_rate'].fillna(0)
test_df['inv']=test_df['comp1_inv'].fillna(0)+test_df['comp2_inv'].fillna(0)+test_df['comp3_inv'].fillna(0)+test_df['comp4_inv'].fillna(0)+test_df['comp5_inv'].fillna(0)+test_df['comp6_inv'].fillna(0)+test_df['comp7_inv'].fillna(0)+test_df['comp8_inv'].fillna(0)
test_df['room_minus_guests'] = test_df['srch_room_count']-(test_df['srch_adults_count']+test_df['srch_children_count'])
test_df['room_per_person'] = test_df['room_minus_guests'].apply(is_positive) #is there a room per person
test_df['room_guest_ratio'] = (test_df['srch_adults_count']+test_df['srch_children_count'])/ test_df['srch_room_count']#ratio of rooms to guests
test_df['price_usd_norm'] = test_df['price_usd']/ test_df['price_usd'].abs().max()
test_df['same_country'] = test_df.apply(lambda x: same_country(x.visitor_location_country_id, x.prop_country_id), axis=1)

#Categorical from above (previous data frame)
# be cause 'gross_bookings_usd' ius not in other datraframe, we have to verify our categories are
for factor in categorical:
     if factor in test_df.columns.values:
        test_df[factor] = test_df[factor].astype('category')

print('preparing test sets')
X_t1 = test_df[features1]
X_t2 = test_df[features2]
X_t3 = test_df[features3]
X_t4 = test_df[features4]
X_t5 = test_df[features5]
X_t6 = test_df[features6]
X_t7 = test_df[features7]
X_t8 = test_df[features8]
X_t9 = test_df[features9]

print('Making poredictions')
print('Predicting first layer models')
print('Predicting model 1')
train_prediuctions1=model1.predict(X_t1)
print('Predicting model 2')
train_prediuctions2=model2.predict(X_t2)
print('Predicting model 3')
train_prediuctions3=model3.predict(X_t3)
print('Predicting model 4')
train_prediuctions4=model4.predict(X_t4)
print('Predicting model 5')
train_prediuctions5=model5.predict(X_t5)
print('Predicting model 6')
train_prediuctions6=model6.predict(X_t6)
print('Predicting model 7')
train_prediuctions7=model7.predict(X_t7)
print('Predicting model 8')
train_prediuctions8=model8.predict(X_t8)
print('Predicting model 9')
train_prediuctions9=model9.predict(X_t9)

dic = {
     'srch_id' : test_df['srch_id'],
     'prop_id' : test_df['prop_id'],
     'predictions1' : train_prediuctions1,
     'predictions2' : train_prediuctions2,
     'predictions3' : train_prediuctions3,
     'predictions4' : train_prediuctions4,
     'predictions5' : train_prediuctions5,
     'predictions6' : train_prediuctions6,
     'predictions7' : train_prediuctions7,
     'predictions8' : train_prediuctions8,
     'predictions9' : train_prediuctions9
}
prediction_df = pd.DataFrame(dic)

print('Predicting cumulative model')
predictions = model_cumulative.predict(prediction_df)
dic = {
     'srch_id' : test_df['srch_id'],
     'prop_id' : test_df['prop_id'],
     'score' : predictions
}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])

#*****
print(ranked)


print('Saving cumulative model to file')
output_df = ranked[out_fearures]
print(output_df)
output_df.to_csv('submission_cumulative.csv', index=False)

print(len(output_df))
print('------------------------------------------------------------ ')
print('Best nDCG@5:', model_cumulative.best_score_['valid_0']['ndcg@5'])
print('------------------------------------------------------------ ')





print('Concatenating data frames for training dataset 2')
#frames = [prediction_df, test_df]
#cumulative2_df = pd.concat(frames)    #https://pandas.pydata.org/docs/user_guide/merging.html

cumulative2_df=test_df
cumulative2_df['predictions1']=prediction_df['predictions1']
cumulative2_df['predictions2']=prediction_df['predictions2']
cumulative2_df['predictions3']=prediction_df['predictions3']
cumulative2_df['predictions4']=prediction_df['predictions4']
cumulative2_df['predictions5']=prediction_df['predictions5']
cumulative2_df['predictions6']=prediction_df['predictions6']
cumulative2_df['predictions7']=prediction_df['predictions7']
cumulative2_df['predictions8']=prediction_df['predictions8']
cumulative2_df['predictions9']=prediction_df['predictions9']

cumulative2_df=cumulative2_df[features_cum2]

print('Predicting cumulative model 2')
predictions = model_cumulative2.predict(cumulative2_df)
dic = {
     'srch_id' : test_df['srch_id'],
     'prop_id' : test_df['prop_id'],
     'score' : predictions
}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])

#*****
print(ranked)

print('Saving cumulative model to file')
output_df = ranked[out_fearures]
print(output_df)
output_df.to_csv('submission_cumulative2.csv', index=False)

print(len(output_df))
print('------------------------------------------------------------ ')
print('Best nDCG@5:', model_cumulative2.best_score_['valid_0']['ndcg@5'])
print('------------------------------------------------------------ ')













print('Saving layer 1 models to file')
print('------------------------------------------------------------ ')
dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions1}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model1 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model1.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions2}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model2 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model2.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions3}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model3 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model3.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions4}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model4 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model4.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions5}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model5 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model5.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions6}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model6 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model6.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions7}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model7 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model7.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions8}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model8 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model8.csv', index=False)
print('------------------------------------------------------------ ')

dic = {'srch_id' : test_df['srch_id'], 'prop_id' : test_df['prop_id'], 'score' : train_prediuctions9}
ranked = pd.DataFrame(dic)
ranked = ranked.sort_values(by = ['srch_id', 'score'], ascending = [True, False])
print('Saving model9 to file')
output_df = ranked[out_fearures]
output_df.to_csv('multiple_lightgbm_ranking/submission_model9.csv', index=False)
print('------------------------------------------------------------ ')
