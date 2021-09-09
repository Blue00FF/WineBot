import pandas as pd
import nltk

df = pd.read_csv('../data/winemag-data-130k-v2.csv', index_col=0)
df = df.drop(['taster_name', 'taster_twitter_handle', 'region_1',
             'region_2', 'province', 'winery'], axis=1)
df = df.dropna()
df.to_csv('../data/winemag-data_custom.csv', index=False)
