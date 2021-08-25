import pandas as pd

df = pd.read_csv('../data/winemag-data_custom.csv')

pd.set_option('display.max_columns', None)

print(df)
