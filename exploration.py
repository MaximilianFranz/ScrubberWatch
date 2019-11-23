import pandas as pd


df = pd.read_csv('data/q3.csv')

df['slot-datetime'] = pd.to_datetime(df['IOTERMIN'], infer_datetime_format=True)
df['slot-date'] = df['slot-datetime'].dt.date

counts = df.groupby("slot-date").size()

onedate = df.loc[df['slot-date'] == "2019-01-07"]

print(df.head())

