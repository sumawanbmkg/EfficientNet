import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_golden_dataset():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir.parent / "dataset_fix"
    metadata_path = data_dir / "train.csv"
    
    print(f"Reading main metadata: {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"Total samples: {len(df)}")
    
    # Stratified Split (To ensure golden set has balanced classes)
    # We stratify by magnitude_class if possible, or just shuffle
    
    # 20% for Golden Set (Fixed forever)
    train_active, golden_test = train_test_split(
        df, test_size=0.20, random_state=2026, stratify=df['magnitude_class']
    )
    
    algo_dir = base_dir / "data"
    algo_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the splits
    train_active_path = algo_dir / "unified_metadata.csv" # This is the "Active" training set
    golden_test_path = algo_dir / "golden_test.csv"       # This is the "Holy" test set
    
    train_active.to_csv(train_active_path, index=False)
    golden_test.to_csv(golden_test_path, index=False)
    
    print("-" * 50)
    print(f"✅ GOLDEN DATASET CREATED (Frozen for Evaluation)")
    print(f"   Golden Test Set: {len(golden_test)} samples -> {golden_test_path}")
    print(f"   Breakdown: {golden_test['magnitude_class'].value_counts().to_dict()}")
    print("-" * 50)
    print(f"✅ ACTIVE TRAINING SET (Evolving)")
    print(f"   Active Train Set: {len(train_active)} samples -> {train_active_path}")
    print("-" * 50)
    print("Next Step: Update pipeline_config.yaml to point to 'unified_metadata.csv' for training")
    print("           and use 'golden_test.csv' ONLY for No-Harm Validation calculation.")

if __name__ == "__main__":
    create_golden_dataset()
