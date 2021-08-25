import pandas as pd

df = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)
df = df.drop(['taster_name', 'taster_twitter_handle', 'region_1',
             'region_2', 'province', 'winery'], axis=1)
df = df.dropna()
df.to_csv('winemag-data_custom.csv', index=False)
