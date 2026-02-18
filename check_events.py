import pandas as pd

# Read event list
df = pd.read_excel('intial/event_list.xlsx')

print(f'Total events: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'\nFirst 5 rows:')
print(df.head())
print(f'\nLast 5 rows:')
print(df.tail())
print(f'\nStations: {df["Stasiun"].unique()}')
print(f'\nStation counts:')
print(df["Stasiun"].value_counts())
print(f'\nDate range: {df["Tanggal"].min()} to {df["Tanggal"].max()}')
