import pandas as pd
import os

csv_count = len(pd.read_csv('quiet_days.csv'))
print(f'Total records in quiet_days.csv: {csv_count}')

metadata_path = 'dataset_normal/metadata/dataset_metadata.csv'
if os.path.exists(metadata_path):
    generated_count = len(pd.read_csv(metadata_path))
    print(f'Already generated: {generated_count}')
    print(f'Remaining: {csv_count - generated_count}')
    print(f'Progress: {generated_count/csv_count*100:.1f}%')
else:
    print('No data generated yet')
    print(f'Need to generate: {csv_count} records')
