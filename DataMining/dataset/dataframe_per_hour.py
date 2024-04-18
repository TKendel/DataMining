import pandas as pd



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

#Print example
print("Dataframe Head")
print(dataframe.head())
print("The dataframe contains", len(dataframe), "data points")

#Save raw results
print("Saving data to CSV")
dataframe.to_csv('dataframe_raw_per_hour.csv', index=False)
print("Data saved succesfully")

#Interpolate results
print("Interpolating dataset")
fwrd_fill, interpolated = fill_NA(dataframe)

#Print example
print("fwrd_fill Dataframe Head")
print(fwrd_fill.head())
print("The dataframe contains", len(fwrd_fill), "data points")

#Save interpolated results
print("Saving data to CSV")
fwrd_fill.to_csv('dataframe_fwrd_fill_per_hour.csv', index=False)
interpolated.to_csv('dataframe_interpolated_per_hour.csv', index=False)
print("Data saved succesfully")



    #Remove lines with all NaN in all 'mood', "circumplex.arousal", "circumplex.valence", "activity"
print("Removing rows with all NaN values (excluding time and id)")
#'mood', "circumplex.arousal", "circumplex.valence", "activity"
drop = dataframe.copy()
#https://stackoverflow.com/questions/39128856/python-drop-row-if-two-columns-are-nan
drop.dropna(subset=['mood', "circumplex.arousal", "circumplex.valence", "activity"], how='all', inplace=True)


#Print example
print("Dataframe Head")
print(drop.head())
print("The dataframe contains", len(drop), "data points")

#Save raw results
print("Saving data to CSV")
drop.to_csv('dataframe_raw_drop_per_hour.csv', index=False)
print("Data saved succesfully")

#Interpolate results
print("Interpolating dataset")
drop_fwrd_fill, drop_interpolated = fill_NA(drop)

#Print example
print("drop_fwrd_fill Dataframe Head")
print(drop_fwrd_fill.head())
print("The dataframe contains", len(drop_fwrd_fill), "data points")

#Save interpolated results
print("Saving data to CSV")
drop_fwrd_fill.to_csv('dataframe_drop_fwrd_fill_per_hour.csv', index=False)
drop_interpolated.to_csv('dataframe_drop_interpolated_per_hour.csv', index=False)
print("Data saved succesfully")
