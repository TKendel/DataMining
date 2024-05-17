import pandas as pd
import pickle


test_data = pd.read_csv("DataMining2/test_set_VU_DM.csv", index_col=0)
df = pd.DataFrame(test_data).reset_index()

with open("GBC_clicked.pkl", "rb") as f:
    clicked_model = pickle.load(f)

with open("GBC_booking.pkl", "rb") as f:
    booked_model = pickle.load(f)

# Remove null values
df['prop_review_score'].fillna(3, inplace=True)
df['prop_review_score'] = df['prop_review_score'].replace({0:2.5})
df["prop_location_score2"].fillna(0, inplace=True)
df['srch_query_affinity_score'].fillna((df['srch_query_affinity_score'].mean()), inplace=True)
df['orig_destination_distance'].fillna((df['orig_destination_distance'].median()), inplace=True)
df["visitor_hist_adr_usd"].fillna(0, inplace=True)
df['visitor_hist_starrating_bool'] = pd.notnull(df['visitor_hist_starrating'])

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

test_click = df.copy()
test_click.drop(columns=test_click.columns[26:51], inplace=True)
test_click.drop(columns=['srch_id', 'date_time', 'visitor_hist_starrating'], inplace=True)

test_book = df.copy()
test_book.drop(columns=test_book.columns[26:50], inplace=True)
test_book.drop(columns=['srch_id', 'date_time', 'visitor_hist_starrating'], inplace=True)

c_prob = clicked_model.predict_proba(test_click.values)[:,1]
c_prob = list(-1.0*c_prob)
b_prob = booked_model.predict_proba(test_book.values)[:,1]
b_prob = list(-1.0*b_prob)


recommendations = zip(df["srch_id"], df["prop_id"], 4*b_prob+c_prob)

recommendations = list(recommendations)
df = pd.DataFrame(recommendations)

df.sort_values(by = [0, 2], ascending = [True, False])
df = df.rename(columns={0: 'srch_id', 1: 'prop_id', 2: 'score'})

df = df[['srch_id','prop_id']]
df.to_csv('submission2.csv', index=False)
