import pandas as pd
import numpy as np


def fill_NA(data):
    """
    Takes a pd dataframe
    returns two data frames, fwrd fill and interpolated
    """
    ## FORWARD - BACKWARD FILL
    data_a=data.copy(deep=False)
    #Filling missing values - FORWARD fill
    data_a.fillna(method='ffill',inplace=True)

    #Filling missing values - BACKWARD fill
    #Second fill in the remaining missing values with the last valid value
    data_a.fillna(method='bfill',inplace=True)


    ## LINEAR INTERPOLATION
    data_b=data.copy(deep=False)
    data_b.interpolate(limit_direction="both",inplace=True)

    return data_a, data_b




data = pd.read_csv("pivot_df.csv")  
data_a=pd.DataFrame(data)
data_k=data_a.copy()


data_k["time"] = pd.to_datetime(data_k["time"])

#https://technology.amis.nl/data-analytics/convert-groupby-result-on-pandas-data-frame-into-a-data-frame-using-to_frame/
#Could only do it one by one, NaN values will remain so, but because there are no NaN for time, all frames will have the same rows
mood=data_k.groupby( ['id', pd.Grouper(key='time', freq='1H')] )["mood"].mean().to_frame().reset_index()
arousal=data_k.groupby( ['id', pd.Grouper(key='time', freq='1H')] )["circumplex.arousal"].mean().to_frame().reset_index()
valence=data_k.groupby( ['id', pd.Grouper(key='time', freq='1H')] )["circumplex.valence"].mean().to_frame().reset_index()
activity=data_k.groupby( ['id', pd.Grouper(key='time', freq='1H')] )["activity"].mean().to_frame().reset_index()
#Attach all info into one data frame
dataframe=mood.copy()
dataframe["circumplex.arousal"]=arousal["circumplex.arousal"]
dataframe["circumplex.valence"]=valence["circumplex.valence"]
dataframe["activity"]=activity["activity"]


#Generates a list of average mood per day
next_mood=data_k.groupby( ['id', pd.Grouper(key='time', freq='24H')] )["mood"].mean().to_frame().reset_index()
#Shifts all moods one day behind (so they become the mood of the next day)
next_mood["time"] = next_mood["time"] + pd.Timedelta('-1 day')







#dataframe["next_mood"] = data_k[data_k['id'] == id]
#print(dataframe.loc[dataframe['time'].dt.date==next_mood["time"]])
#print("HEEERE")

#dataframe["next_mood"]= next_mood["activity"] if dataframe.loc[dataframe['time'].dt.date==next_mood["activity"]] else pass




dataframe['next_mood'] = np.nan
print("iterating for matches in date and id to add next_mood" )
for  index, row in dataframe.iterrows():
    #print(row['time'].date() )
    for index_, row_ in next_mood.iterrows():
        #print(row_["time"].date() )
        if(row_["time"].date()==row['time'].date())and(row_["id"]==row['id']):
            print("YESSSS")
            print(row_["time"].date(), row['time'].date(), row_["id"], row['id'] )
            dataframe['next_mood'][index] = row_["mood"]
            print(row_["mood"])
            break   #Break out of the second for-loop, we only need one match, as ther should not be two entries with same ID and Date


#dataframe["next_mood"]=next_mood["mood"]
"""
dataframe.reset_index(drop=True)
next_mood.reset_index(drop=True)
dataframe['next_mood']= np.where((dataframe['time'].dt.date == next_mood['time'].dt.date) & ( dataframe['id'] == next_mood['if']), next_mood['mood'], np.nan)
"""

#print(len(dataframe))
#print(dataframe.head())




    #Raw data
#Print example
print("Dataframe Head")
print(dataframe.head())
print("The dataframe contains", len(dataframe), "data points")


    #Interpollated data
#Interpolate results
print("Interpolating dataset")
fwrd_fill, interpolated = fill_NA(dataframe)
#Print example
print("fwrd_fill Dataframe Head")
print(fwrd_fill.head())
print("The dataframe contains", len(fwrd_fill), "data points")



    #Drop Raw
    #Remove lines with all NaN in all 'mood', "circumplex.arousal", "circumplex.valence", "activity"
print("Removing rows with all NaN values (excluding time and id)")
#'mood', "circumplex.arousal", "circumplex.valence", "activity"
    #Needs to be copied before sorting or dropna will not work
drop = dataframe.copy()
#https://stackoverflow.com/questions/39128856/python-drop-row-if-two-columns-are-nan
drop.dropna(subset=['mood', "circumplex.arousal", "circumplex.valence", "activity"], how='all', inplace=True)
#Print example
print("Dataframe Head")
print(drop.head())
print("The dataframe contains", len(drop), "data points")


    #Drop Interpollated
#Interpolate results
print("Interpolating dataset")
drop_fwrd_fill, drop_interpolated = fill_NA(drop)
#Print example
print("drop_fwrd_fill Dataframe Head")
print(drop_fwrd_fill.head())
print("The dataframe contains", len(drop_fwrd_fill), "data points")

print("dataframe raw null values: ", dataframe.isnull().sum())
print("dataframe fwrd_fill null values: ", fwrd_fill.isnull().sum())
print("dataframe interpolated null values: ", interpolated.isnull().sum())
print("dataframe drop null values: ", drop.isnull().sum())
print("dataframe drop_fwrd_fill null values: ", drop_fwrd_fill.isnull().sum())
print("dataframe drop_interpolated null values: ", drop_interpolated.isnull().sum())


#Save raw results
print("Saving data to CSV")
dataframe.sort_values(by=['time'], inplace=True), dataframe.to_csv('dataframe_raw_per_hour.csv', index=False)
print("Data saved succesfully")

#Save interpolated results
print("Saving data to CSV")
fwrd_fill.sort_values(by=['time'], inplace=True), fwrd_fill.to_csv('dataframe_fwrd_fill_per_hour.csv', index=False)
interpolated.sort_values(by=['time'], inplace=True), interpolated.to_csv('dataframe_interpolated_per_hour.csv', index=False)
print("Data saved succesfully")

#Save raw drop results
print("Saving data to CSV")
drop.sort_values(by=['time'], inplace=True), drop.to_csv('dataframe_raw_drop_per_hour.csv', index=False)
print("Data saved succesfully")

#Save interpolated results
print("Saving data to CSV")
drop_fwrd_fill.sort_values(by=['time'], inplace=True), drop_fwrd_fill.to_csv('dataframe_drop_fwrd_fill_per_hour.csv', index=False)
drop_interpolated.sort_values(by=['time'], inplace=True), drop_interpolated.to_csv('dataframe_drop_interpolated_per_hour.csv', index=False)
print("Data saved succesfully")
