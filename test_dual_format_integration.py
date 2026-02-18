#!/usr/bin/env python3
"""
Test Dual Format Integration
Test the updated geomagnetic_fetcher.py with dual format support
"""

import sys
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dual_format_integration():
    """Test the integrated dual format support"""
    
    print("="*80)
    print("TESTING INTEGRATED DUAL FORMAT SUPPORT")
    print("="*80)
    
    # Test 1: Default behavior (prefer compressed)
    print("\n--- Test 1: Default Behavior (Prefer Compressed) ---")
    with GeomagneticDataFetcher() as fetcher:
        data = fetcher.fetch_data('2018-01-17', 'SCN')
        if data:
            print(f"✅ Success!")
            if 'file_info' in data:
                print(f"   Format: {data['file_info']['format']}")
                print(f"   Description: {data['file_info']['description']}")
                print(f"   File size: {data['file_info']['compressed_size']:,} bytes")
                if data['file_info']['format'] == 'gz':
                    print(f"   Compression ratio: {data['file_info']['compression_ratio']:.1f}x")
            print(f"   Coverage: {data['stats']['coverage']:.1f}%")
            print(f"   Valid samples: {data['stats']['valid_samples']:,}")
        else:
            print("❌ Failed")
    
    # Test 2: Prefer uncompressed
    print("\n--- Test 2: Prefer Uncompressed Files ---")
    with GeomagneticDataFetcher(prefer_compressed=False) as fetcher:
        data = fetcher.fetch_data('2018-01-17', 'SCN')
        if data:
            print(f"✅ Success!")
            if 'file_info' in data:
                print(f"   Format: {data['file_info']['format']}")
                print(f"   Description: {data['file_info']['description']}")
                print(f"   File size: {data['file_info']['compressed_size']:,} bytes")
            print(f"   Coverage: {data['stats']['coverage']:.1f}%")
            print(f"   Valid samples: {data['stats']['valid_samples']:,}")
        else:
            print("❌ Failed")
    
    # Test 3: Test with precursor system
    print("\n--- Test 3: Integration with Precursor System ---")
    try:
        from test_precursor_integrated import IntegratedPrecursorTester
        
        # Create tester with dual format fetcher
        tester = IntegratedPrecursorTester(device='cpu')
        
        # Test a single hour
        print("Testing precursor system with dual format support...")
        results = tester.test_date('2018-01-17', 'SCN', [0])
        
        if results and results['summary']['status'] == 'SUCCESS':
            print("✅ Precursor system integration successful!")
            print(f"   Hours analyzed: {results['summary']['successful_tests']}")
            print(f"   Coverage: {results['summary']['coverage']:.1f}%")
        else:
            print("❌ Precursor system integration failed")
            
    except Exception as e:
        print(f"❌ Error testing precursor integration: {e}")
    
    print("\n" + "="*80)
    print("DUAL FORMAT INTEGRATION TEST COMPLETED")
    print("="*80)

if __name__ == '__main__':
    test_dual_format_integration()