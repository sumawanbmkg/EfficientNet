#!/usr/bin/env python3
"""
Test API Directly
Test if the API endpoints are returning correct data
"""

import requests
import json

print("="*70)
print("TESTING DASHBOARD API ENDPOINTS")
print("="*70)

base_url = "http://localhost:5000"

# Test system health
print("\n1. Testing /api/system_health...")
try:
    response = requests.get(f"{base_url}/api/system_health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Model Version: {data.get('model_version', 'N/A')}")
        print(f"   Issues: {data.get('issues', [])}")
        
        if data.get('status') == 'HEALTHY':
            print(f"   Result: ✅ HEALTHY")
        else:
            print(f"   Result: ⚠️  {data.get('status')}")
            if data.get('issues'):
                for issue in data['issues']:
                    print(f"      - {issue}")
    else:
        print(f"   Error: Status {response.status_code}")
except Exception as e:
    print(f"   Error: {e}")

# Test model info
print("\n2. Testing /api/model_info...")
try:
    response = requests.get(f"{base_url}/api/model_info", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   Version: {data.get('version', 'N/A')}")
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Confidence: {data.get('confidence_level', 'N/A')}%")
        
        perf = data.get('performance', {})
        if perf:
            print(f"   Test Accuracy: {perf.get('test_magnitude_accuracy', 'N/A')}%")
            print(f"   Detection Rate: {perf.get('validation_detection_rate', 'N/A')}%")
        
        if data.get('version') == '2.0':
            print(f"   Result: ✅ CORRECT VERSION")
        else:
            print(f"   Result: ⚠️  Wrong version")
    else:
        print(f"   Error: Status {response.status_code}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

print("\n📊 DIAGNOSIS:")
print("\nIf you see:")
print("  - Status: HEALTHY ✅")
print("  - Version: 2.0 ✅")
print("  - Test Accuracy: 98.68% ✅")
print("\nThen the API is working correctly!")
print("\nIf you still see 'undefined' in browser:")
print("  1. Hard refresh: Ctrl+F5")
print("  2. Clear browser cache")
print("  3. Try different browser")
print("\nIf you see 'Model file not found':")
print("  1. Stop server: Ctrl+C")
print("  2. Restart: python production/scripts/web_dashboard.py")
print("  3. Refresh browser")
