#!/usr/bin/env python3
"""
Direct Test of Precursor System with CNN
Bypasses command line parsing to test core functionality
"""

import sys
import os
from datetime import datetime

# Add intial to path
sys.path.insert(0, 'intial')

# Import the tester class directly
from test_precursor import EarthquakePrecursorTester

def test_precursor_with_cnn():
    """Test precursor system with CNN integration"""
    
    print("="*80)
    print("DIRECT PRECURSOR TEST WITH CNN")
    print("="*80)
    
    try:
        # Initialize tester
        print("1. Initializing tester...")
        tester = EarthquakePrecursorTester(device='cpu')
        print("✅ Tester initialized")
        
        # Load model
        print("\n2. Loading CNN model...")
        model_path = "experiments/exp_20260129_160807/best_model.pth"
        tester.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Test date
        print("\n3. Testing precursor analysis...")
        test_date = "2018-01-17"
        station = "SCN"
        hour_range = [0, 1]  # Test first 2 hours
        
        print(f"Date: {test_date}")
        print(f"Station: {station}")
        print(f"Hours: {hour_range}")
        
        # Run test
        results = tester.test_date(test_date, station, hour_range)
        
        # Generate report
        print("\n4. Generating report...")
        output_file = f"precursor_cnn_report_{station}_{test_date.replace('-', '')}.txt"
        report = tester.generate_report(results, output_file)
        
        print("✅ Report generated")
        print(f"📄 Report saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        summary = results['summary']
        print(f"Status: {summary['status']}")
        print(f"Successful tests: {summary['successful_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        
        if 'cnn_prediction' in summary:
            cnn = summary['cnn_prediction']
            print(f"\n🎯 CNN PREDICTIONS:")
            print(f"Magnitude: {cnn['magnitude']['predicted_class']} "
                  f"(~{cnn['magnitude']['predicted_value']}) "
                  f"[Confidence: {cnn['magnitude']['confidence']:.3f}]")
            print(f"Azimuth: {cnn['azimuth']['predicted_class']} "
                  f"({cnn['azimuth']['predicted_degrees']}°) "
                  f"[Confidence: {cnn['azimuth']['confidence']:.3f}]")
        
        if 'anomaly_analysis' in summary:
            anom = summary['anomaly_analysis']
            print(f"\n📊 ANOMALY ANALYSIS:")
            print(f"Precursor Likelihood: {anom['precursor_likelihood']}")
            print(f"Anomaly Level: {anom['anomaly_level']}")
            print(f"Average Anomaly Score: {anom['average_anomaly_score']:.3f}")
        
        print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_precursor_with_cnn()
    sys.exit(0 if success else 1)