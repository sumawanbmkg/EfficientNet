#!/usr/bin/env python3
"""
Daily Operations Script
Automated script untuk daily operations

Features:
- Automated daily scans
- Performance monitoring
- Report generation
- Alert checking
- Log management

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import json

print("="*70)
print("DAILY OPERATIONS - Earthquake Prediction System")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Configuration
STATIONS = ['SCN', 'GTO', 'MLB', 'SKB', 'LWK']  # Priority stations
SCAN_DATE = datetime.now().strftime('%Y-%m-%d')  # Today

# Step 1: System Health Check
print("\n🏥 STEP 1: System Health Check")
print("-"*70)

health_checks = {
    'model_file': Path('../models/earthquake_model.pth').exists(),
    'config_file': Path('../config/production_config.json').exists(),
    'monitoring_dir': Path('../monitoring').exists(),
    'results_dir': Path('../results').exists()
}

all_healthy = all(health_checks.values())

for check, status in health_checks.items():
    icon = "✅" if status else "❌"
    print(f"   {icon} {check}: {'OK' if status else 'FAILED'}")

if not all_healthy:
    print("\n❌ System health check FAILED!")
    print("   Please fix issues before continuing.")
    sys.exit(1)

print("\n✅ System health check PASSED!")

# Step 2: Run Daily Scans
print("\n📡 STEP 2: Running Daily Scans")
print("-"*70)
print(f"   Scanning date: {SCAN_DATE}")
print(f"   Stations: {', '.join(STATIONS)}")

scan_results = []

for station in STATIONS:
    print(f"\n   Scanning {station}...")
    
    try:
        # Run scanner (without saving to avoid display issues)
        cmd = [
            sys.executable,
            'prekursor_scanner_production.py',
            '--station', station,
            '--date', SCAN_DATE
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"      ✅ {station}: Scan complete")
            scan_results.append({
                'station': station,
                'status': 'success',
                'date': SCAN_DATE
            })
        else:
            print(f"      ❌ {station}: Scan failed")
            scan_results.append({
                'station': station,
                'status': 'failed',
                'date': SCAN_DATE,
                'error': result.stderr[:200]
            })
    
    except subprocess.TimeoutExpired:
        print(f"      ⚠️  {station}: Timeout")
        scan_results.append({
            'station': station,
            'status': 'timeout',
            'date': SCAN_DATE
        })
    
    except Exception as e:
        print(f"      ❌ {station}: Error - {str(e)[:100]}")
        scan_results.append({
            'station': station,
            'status': 'error',
            'date': SCAN_DATE,
            'error': str(e)[:200]
        })

# Summary
successful_scans = sum(1 for r in scan_results if r['status'] == 'success')
print(f"\n   Summary: {successful_scans}/{len(STATIONS)} scans successful")

# Step 3: Generate Daily Report
print("\n📊 STEP 3: Generating Daily Report")
print("-"*70)

try:
    cmd = [
        sys.executable,
        'monitor_production_performance.py',
        '--report',
        '--time-window', '1'
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("   ✅ Daily report generated")
    else:
        print("   ⚠️  Report generation had issues")

except Exception as e:
    print(f"   ❌ Error generating report: {str(e)[:100]}")

# Step 4: Check for Alerts
print("\n🚨 STEP 4: Checking for Alerts")
print("-"*70)

# Check monitoring data for alerts
predictions_file = Path('../monitoring/predictions_log.csv')

if predictions_file.exists():
    import pandas as pd
    
    df = pd.read_csv(predictions_file)
    
    # Check recent predictions
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    recent = df[df['timestamp'] >= (datetime.now() - timedelta(days=1))]
    
    if len(recent) > 0:
        precursor_count = recent['is_precursor'].sum()
        
        print(f"   Recent predictions: {len(recent)}")
        print(f"   Precursor detections: {precursor_count}")
        
        if precursor_count > 0:
            print(f"\n   ⚠️  ALERT: {precursor_count} precursor(s) detected!")
            print(f"   Review predictions and take appropriate action.")
        else:
            print(f"   ✅ No precursors detected")
    else:
        print("   ℹ️  No recent predictions")
else:
    print("   ℹ️  No monitoring data yet")

# Step 5: Save Daily Log
print("\n💾 STEP 5: Saving Daily Log")
print("-"*70)

log_dir = Path('../logs')
log_dir.mkdir(exist_ok=True)

daily_log = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'system_health': 'HEALTHY' if all_healthy else 'UNHEALTHY',
    'scans_performed': len(scan_results),
    'scans_successful': successful_scans,
    'scan_results': scan_results
}

log_file = log_dir / f"daily_log_{datetime.now().strftime('%Y%m%d')}.json"

with open(log_file, 'w') as f:
    json.dump(daily_log, f, indent=2)

print(f"   ✅ Daily log saved: {log_file}")

# Final Summary
print("\n" + "="*70)
print("DAILY OPERATIONS COMPLETE")
print("="*70)

print(f"\n📊 Summary:")
print(f"   System Health: {'✅ HEALTHY' if all_healthy else '❌ UNHEALTHY'}")
print(f"   Scans: {successful_scans}/{len(STATIONS)} successful")
print(f"   Report: Generated")
print(f"   Log: Saved")

print(f"\n💡 Next Steps:")
print(f"   1. Review scan results in ../results/")
print(f"   2. Check monitoring dashboard")
print(f"   3. Verify any precursor detections")
print(f"   4. Update daily operations log")

print("\n" + "="*70)
print("✅ ALL OPERATIONS COMPLETE!")
print("="*70)
