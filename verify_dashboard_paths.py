#!/usr/bin/env python3
"""
Verify Dashboard Paths
Check if all paths are correct and files exist
"""

from pathlib import Path
import json

print("="*70)
print("VERIFYING DASHBOARD PATHS")
print("="*70)

# Simulate web_dashboard.py path resolution
script_dir = Path(__file__).parent / 'production' / 'scripts'
print(f"\nScript directory: {script_dir}")

# Check model file
model_file = script_dir.parent / 'models' / 'earthquake_model.pth'
print(f"\n1. Model File:")
print(f"   Path: {model_file}")
print(f"   Exists: {model_file.exists()}")
if model_file.exists():
    size_mb = model_file.stat().st_size / (1024*1024)
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Status: ✅ FOUND")
else:
    print(f"   Status: ❌ NOT FOUND")

# Check config file
config_file = script_dir.parent / 'config' / 'production_config.json'
print(f"\n2. Config File:")
print(f"   Path: {config_file}")
print(f"   Exists: {config_file.exists()}")
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"   Version: {config.get('model_version', 'N/A')}")
    print(f"   Status: {config.get('deployment_status', 'N/A')}")
    perf = config.get('performance_metrics', {})
    if perf:
        print(f"   Accuracy: {perf.get('test_magnitude_accuracy', 'N/A')}%")
    print(f"   Status: ✅ FOUND")
else:
    print(f"   Status: ❌ NOT FOUND")

# Check monitoring directory
monitoring_dir = script_dir.parent / 'monitoring'
print(f"\n3. Monitoring Directory:")
print(f"   Path: {monitoring_dir}")
print(f"   Exists: {monitoring_dir.exists()}")
if monitoring_dir.exists():
    predictions_file = monitoring_dir / 'predictions_log.csv'
    print(f"   Predictions log: {predictions_file.exists()}")
    print(f"   Status: ✅ FOUND")
else:
    print(f"   Status: ⚠️  Will be created automatically")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

# Summary
all_good = model_file.exists() and config_file.exists()
if all_good:
    print("\n✅ All critical files found!")
    print("✅ Dashboard should work correctly")
    print("\nNext step:")
    print("1. Stop current server (Ctrl+C)")
    print("2. Run: python production/scripts/web_dashboard.py")
    print("3. Open: http://localhost:5000")
else:
    print("\n❌ Some files missing!")
    print("Please check the paths above")
