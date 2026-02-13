import pandas as pd
from pathlib import Path

def consolidate_metadata():
    base_path = Path('d:/multi/dataset_consolidation/metadata')
    files = ['split_train.csv', 'split_val.csv', 'split_test.csv']
    
    dfs = []
    for f in files:
        file_path = base_path / f
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Add a 'split' column to distinguish them
            df['split'] = f.split('_')[1].split('.')[0]
            dfs.append(df)
            print(f"Loaded {f}: {len(df)} samples")
    
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        # Ensure common columns match
        output_path = base_path / 'metadata_final_phase21.csv'
        final_df.to_csv(output_path, index=False)
        print(f"Consolidated into {output_path}: {len(final_df)} samples total")
        return True
    return False

if __name__ == "__main__":
    consolidate_metadata()
