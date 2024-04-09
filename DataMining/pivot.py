import pandas as pd
import openpyxl


## Pandas options ##
pd.set_option('display.max_columns', None)

## Load data into data frame ##
data = pd.read_csv("dataset_mood_smartphone.csv", index_col=0)
df = pd.DataFrame(data)
df["time"] = pd.to_datetime(df["time"])

variable_list = ["circumplex.arousal", "circumplex.valence", "activity", "screen", "call", "sms", "appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]

df = df.reset_index().pivot_table(values="value", index=["id", "time"], columns="variable", aggfunc='mean')
df.to_excel('out.xlsx')