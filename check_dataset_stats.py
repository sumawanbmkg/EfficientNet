import pandas as pd

df = pd.read_csv('dataset_unified/metadata/unified_metadata.csv')
print(f'Total samples: {len(df)}')
print(f'\nMagnitude distribution:')
print(df['magnitude_class'].value_counts())
print(f'\nAzimuth distribution:')
print(df['azimuth_class'].value_counts())
