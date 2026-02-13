import pandas as pd
import os
import shutil
from tqdm import tqdm

def consolidate():
    output_dir = 'dataset_experiment_3'
    img_dir = os.path.join(output_dir, 'spectrograms')
    os.makedirs(img_dir, exist_ok=True)
    
    final_dfs = []
    cols = ['filename', 'station', 'date', 'hour', 'magnitude', 'magnitude_class']

    # 1. Normal
    try:
        df = pd.read_csv('dataset_normal_new/metadata.csv')
        print(f"Loading Normal: {len(df)}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying Normal"):
            src = os.path.join('dataset_normal_new', row['filepath'])
            dst = os.path.join(img_dir, row['filename'])
            if os.path.exists(src):
                shutil.copy2(src, dst)
        final_dfs.append(df[cols].copy())
    except Exception as e:
        print(f"Error Normal: {e}")

    # 2. Moderate
    try:
        df = pd.read_csv('dataset_moderate/metadata.csv')
        print(f"Loading Moderate: {len(df)}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying Moderate"):
            src = os.path.join('dataset_moderate', row['filepath'])
            dst = os.path.join(img_dir, row['filename'])
            if os.path.exists(src):
                shutil.copy2(src, dst)
        final_dfs.append(df[cols].copy())
    except Exception as e:
        print(f"Error Moderate: {e}")

    # 3. Medium
    try:
        df = pd.read_csv('dataset_medium_new/metadata.csv')
        print(f"Loading Medium: {len(df)}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying Medium"):
            src = os.path.join('dataset_medium_new', row['filepath'])
            dst = os.path.join(img_dir, row['filename'])
            if os.path.exists(src):
                shutil.copy2(src, dst)
        final_dfs.append(df[cols].copy())
    except Exception as e:
        print(f"Error Medium: {e}")

    # 4. Load Large (Historical)
    try:
        df_21 = pd.read_csv('dataset_consolidation/metadata/metadata_final_phase21.csv')
        df_l = df_21[df_21['magnitude_class'] == 'Large'].copy()
        
        # Phase 2.1 doesn't have 'filename', it has 'consolidation_path'
        # We need to extract filename and ensure it has other columns
        df_l['filename'] = df_l['consolidation_path'].apply(lambda x: os.path.basename(x))
        
        print(f"Loading Large: {len(df_l)}")
        for _, row in tqdm(df_l.iterrows(), total=len(df_l), desc="Copying Large"):
            # Src for Large is in dataset_consolidation/spectrograms/
            src = os.path.join('dataset_consolidation/spectrograms', row['filename'])
            dst = os.path.join(img_dir, row['filename'])
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                # Try fallback if consolidation_path has it differently
                src_fallback = os.path.join('dataset_consolidation', row['consolidation_path'])
                if os.path.exists(src_fallback):
                    shutil.copy2(src_fallback, dst)

        final_dfs.append(df_l[cols].copy())
    except Exception as e:
        print(f"Error Large: {e}")
        import traceback
        traceback.print_exc()

    # Combine
    if final_dfs:
        final_df = pd.concat(final_dfs, ignore_index=True)
        final_df['filepath'] = final_df['filename'].apply(lambda x: f"spectrograms/{x}")
        output_meta = os.path.join(output_dir, 'metadata_raw_exp3.csv')
        final_df.to_csv(output_meta, index=False)
        print("\nFinal Counts:")
        print(final_df['magnitude_class'].value_counts())
        print(f"Total: {len(final_df)}")
    else:
        print("No data consolidated.")

if __name__ == "__main__":
    consolidate()
