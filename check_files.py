import pandas as pd
from pathlib import Path

df = pd.read_csv('dataset_unified/metadata/train_split.csv')
spec_dir = Path('dataset_spectrogram_ssh_v22/spectrograms')

missing = []
for f in df['spectrogram_file'].head(20):
    if not (spec_dir / f).exists():
        missing.append(f)

print(f'Missing files: {len(missing)}/{20}')
if missing:
    print('Examples:', missing[:5])
