"""
Simple test untuk precursor system
"""

import os
import sys
import numpy as np
from datetime import datetime
import logging

# Add intial to path
sys.path.insert(0, 'intial')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple():
    """Simple test"""
    print("="*60)
    print("SIMPLE PRECURSOR TEST")
    print("="*60)
    
    try:
        from geomagnetic_fetcher import GeomagneticDataFetcher
        print("✅ GeomagneticDataFetcher imported successfully")
        
        # Test connection
        with GeomagneticDataFetcher() as fetcher:
            print("✅ SSH connection established")
            
            # Test data fetch
            date = datetime(2018, 1, 17)
            station = 'SCN'
            
            print(f"Testing data fetch: {date.strftime('%Y-%m-%d')} from {station}")
            
            data = fetcher.fetch_data(date, station)
            
            if data is not None:
                print(f"✅ Data fetched successfully")
                print(f"   H samples: {len(data['Hcomp'])}")
                print(f"   D samples: {len(data['Dcomp'])}")
                print(f"   Z samples: {len(data['Zcomp'])}")
                
                # Test hour extraction
                hour = 19
                start_idx = hour * 3600
                end_idx = start_idx + 3600
                
                h_hour = data['Hcomp'][start_idx:end_idx]
                d_hour = data['Dcomp'][start_idx:end_idx]
                z_hour = data['Zcomp'][start_idx:end_idx]
                
                print(f"✅ Hour {hour} data extracted")
                print(f"   H hour samples: {len(h_hour)}")
                print(f"   H mean: {np.mean(h_hour):.1f} nT")
                print(f"   H std: {np.std(h_hour):.1f} nT")
                
                # Basic analysis
                anomaly_score = 0
                if np.std(h_hour) > 2000:
                    anomaly_score += 0.25
                if np.std(d_hour) > 2000:
                    anomaly_score += 0.25
                if np.std(z_hour) > 2000:
                    anomaly_score += 0.25
                
                print(f"✅ Basic anomaly analysis")
                print(f"   Anomaly score: {anomaly_score:.3f}")
                print(f"   Anomaly level: {'High' if anomaly_score > 0.5 else 'Medium' if anomaly_score > 0.25 else 'Low'}")
                
            else:
                print("❌ Failed to fetch data")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == '__main__':
    test_simple()