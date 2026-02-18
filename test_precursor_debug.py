"""
Debug version of precursor test
"""

print("Starting precursor test debug...")

try:
    import sys
    print("✅ sys imported")
    
    import os
    print("✅ os imported")
    
    import argparse
    print("✅ argparse imported")
    
    # Test argparse
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--date', required=True)
    parser.add_argument('--station', required=True)
    parser.add_argument('--hours', default=None)
    
    print("✅ argparse configured")
    
    args = parser.parse_args()
    print(f"✅ args parsed: date={args.date}, station={args.station}, hours={args.hours}")
    
    # Test imports
    sys.path.insert(0, 'intial')
    from geomagnetic_fetcher import GeomagneticDataFetcher
    print("✅ GeomagneticDataFetcher imported")
    
    # Test basic functionality
    print(f"Testing with date: {args.date}, station: {args.station}")
    
    # Parse hours
    if args.hours:
        if '-' in args.hours:
            start, end = map(int, args.hours.split('-'))
            hour_range = list(range(start, end + 1))
        else:
            hour_range = [int(h) for h in args.hours.split(',')]
    else:
        hour_range = list(range(24))
    
    print(f"✅ Hours parsed: {hour_range}")
    
    print("✅ All basic functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()