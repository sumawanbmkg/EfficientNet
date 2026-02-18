#!/usr/bin/env python3
"""
Cleanup Old Files Script
Remove old dashboards, data, and experiments to avoid confusion
Keep only production (v2.0) and latest training (experiments_fixed)
"""

import os
import shutil
from pathlib import Path

print("="*70)
print("CLEANUP OLD FILES - REMOVE CONFUSION")
print("="*70)
print("\n⚠️  This will DELETE old files that use outdated data")
print("✅ Keep only: production/ and experiments_fixed/ (v2.0, 98.68%)")
print("\n" + "="*70)

# Ask for confirmation
response = input("\n🔴 Are you sure you want to cleanup? (yes/no): ")

if response.lower() != 'yes':
    print("\n❌ Cleanup cancelled.")
    exit(0)

print("\n✅ Starting cleanup...\n")

# Files and folders to delete
to_delete = [
    # Old dashboard
    'project_dashboard.py',
    
    # Old dashboard data
    'dashboard_data',
    
    # Old experiments (before fix)
    'experiments',
    'experiments_v3',
    'experiments_v4',
    'experiments_v5',
    'experiments_v6',
    
    # Old training scripts
    'train_earthquake_v3.py',
    'train_with_improvements_v4.py',
    'train_hierarchical_v5.py',
    'train_separate_models_v6.py',
    'train_multi_task.py',
    
    # Old test scripts
    'test_separate_models_v6.py',
]

deleted_count = 0
skipped_count = 0

for item in to_delete:
    path = Path(item)
    
    if path.exists():
        try:
            if path.is_file():
                path.unlink()
                print(f"✅ Deleted file: {item}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"✅ Deleted folder: {item}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Error deleting {item}: {e}")
    else:
        print(f"⏭️  Skipped (not found): {item}")
        skipped_count += 1

print("\n" + "="*70)
print("CLEANUP COMPLETE")
print("="*70)
print(f"\n✅ Deleted: {deleted_count} items")
print(f"⏭️  Skipped: {skipped_count} items")

# Verify production files still exist
print("\n" + "="*70)
print("VERIFYING PRODUCTION FILES")
print("="*70)

production_files = [
    'production/models/earthquake_model.pth',
    'production/config/production_config.json',
    'production/scripts/web_dashboard.py',
    'experiments_fixed/exp_fixed_20260202_163643/best_model.pth',
    'train_fixed_split_pytorch.py'
]

all_good = True
for file in production_files:
    path = Path(file)
    if path.exists():
        print(f"✅ {file}")
    else:
        print(f"❌ MISSING: {file}")
        all_good = False

print("\n" + "="*70)
if all_good:
    print("✅ ALL PRODUCTION FILES INTACT")
    print("="*70)
    print("\n🎉 Cleanup successful!")
    print("\n📊 Now use ONLY:")
    print("   python production/scripts/web_dashboard.py")
    print("   http://localhost:5000")
    print("\n✅ Single source of truth: production/ (v2.0, 98.68%)")
else:
    print("❌ SOME PRODUCTION FILES MISSING!")
    print("="*70)
    print("\n⚠️  Please check production files!")

print("\n" + "="*70)
