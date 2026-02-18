#!/usr/bin/env python3
"""
Test Web Dashboard
Quick test to verify dashboard is working
"""

import requests
import time

def test_dashboard():
    """Test dashboard endpoints"""
    base_url = "http://localhost:5000"
    
    print("="*70)
    print("TESTING WEB DASHBOARD")
    print("="*70)
    
    # Wait for server to start
    print("\n⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Test endpoints
    endpoints = [
        '/api/system_health',
        '/api/model_info',
        '/api/statistics',
        '/api/performance_metrics',
        '/api/recent_predictions'
    ]
    
    print("\n📊 Testing API Endpoints:\n")
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK")
                data = response.json()
                
                # Show key info
                if endpoint == '/api/model_info':
                    print(f"   Version: {data.get('version', 'N/A')}")
                    print(f"   Status: {data.get('status', 'N/A')}")
                    perf = data.get('performance', {})
                    if perf:
                        print(f"   Test Accuracy: {perf.get('test_magnitude_accuracy', 'N/A')}%")
                
                elif endpoint == '/api/system_health':
                    print(f"   Status: {data.get('status', 'N/A')}")
                    print(f"   Model Version: {data.get('model_version', 'N/A')}")
                
                elif endpoint == '/api/statistics':
                    print(f"   Total Predictions: {data.get('total_predictions', 0)}")
                    print(f"   Precursor Detections: {data.get('precursor_detections', 0)}")
                
            else:
                print(f"⚠️  {endpoint}: Status {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Server not running")
            print("\n⚠️  Please start the server first:")
            print("   python production/scripts/web_dashboard.py")
            return False
        
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
    
    print("\n" + "="*70)
    print("✅ DASHBOARD TEST COMPLETE!")
    print("="*70)
    print("\n📊 Dashboard URL: http://localhost:5000")
    print("   Open this URL in your browser to see the dashboard\n")
    
    return True


if __name__ == '__main__':
    test_dashboard()
