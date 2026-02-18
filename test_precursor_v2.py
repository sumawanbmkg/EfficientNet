"""
Test Precursor v2 - Simplified Earthquake Prediction System
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from datetime import datetime
import logging
import argparse

# Add intial to path
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_pc3_filter(data, sampling_rate=1.0, pc3_low=0.01, pc3_high=0.045):
    """Apply PC3 bandpass filter"""
    if len(data) < 100:
        return data
    
    data_clean = np.nan_to_num(data, nan=np.nanmean(data))
    
    nyquist = sampling_rate / 2
    low = max(0.001, min(pc3_low / nyquist, 0.999))
    high = max(0.001, min(pc3_high / nyquist, 0.999))
    
    if low >= high:
        return data_clean
    
    try:
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data_clean)
        return filtered
    except:
        return data_clean


def analyze_precursor_patterns(h_data, d_data, z_data):
    """Analyze precursor patterns"""
    # Basic statistics
    stats = {
        'H': {'mean': np.mean(h_data), 'std': np.std(h_data), 'range': np.max(h_data) - np.min(h_data)},
        'D': {'mean': np.mean(d_data), 'std': np.std(d_data), 'range': np.max(d_data) - np.min(d_data)},
        'Z': {'mean': np.mean(z_data), 'std': np.std(z_data), 'range': np.max(z_data) - np.min(z_data)}
    }
    
    # PC3 analysis
    h_pc3 = apply_pc3_filter(h_data)
    d_pc3 = apply_pc3_filter(d_data)
    z_pc3 = apply_pc3_filter(z_data)
    
    pc3_stats = {
        'H_pc3_std': np.std(h_pc3),
        'D_pc3_std': np.std(d_pc3),
        'Z_pc3_std': np.std(z_pc3),
        'total_pc3_energy': np.std(h_pc3) + np.std(d_pc3) + np.std(z_pc3)
    }
    
    # Anomaly detection
    anomaly_score = 0
    if stats['H']['std'] > 2000:
        anomaly_score += 0.25
    if stats['D']['std'] > 2000:
        anomaly_score += 0.25
    if stats['Z']['std'] > 2000:
        anomaly_score += 0.25
    if pc3_stats['total_pc3_energy'] > 1000:
        anomaly_score += 0.25
    
    anomaly_level = 'High' if anomaly_score > 0.5 else 'Medium' if anomaly_score > 0.25 else 'Low'
    
    return {
        'basic_stats': stats,
        'pc3_stats': pc3_stats,
        'anomaly_score': anomaly_score,
        'anomaly_level': anomaly_level
    }


def test_precursor_date(test_date, station, hour_range):
    """Test precursor untuk tanggal tertentu"""
    
    if isinstance(test_date, str):
        test_date = datetime.strptime(test_date, '%Y-%m-%d')
    
    print("="*80)
    print("EARTHQUAKE PRECURSOR TEST")
    print("="*80)
    print(f"Date: {test_date.strftime('%Y-%m-%d')}")
    print(f"Station: {station}")
    print(f"Hours: {hour_range}")
    print("="*80)
    
    results = []
    successful_tests = 0
    failed_tests = 0
    
    with GeomagneticDataFetcher() as fetcher:
        print("[OK] SSH Connection established!")
        
        for hour in hour_range:
            print(f"\n[HOUR {hour:02d}] Testing {test_date.strftime('%Y-%m-%d')} {hour:02d}:00...")
            
            try:
                # Fetch data
                date = test_date
                data = fetcher.fetch_data(date, station)
                
                if data is None:
                    print(f"[HOUR {hour:02d}] ❌ Failed to fetch data")
                    failed_tests += 1
                    continue
                
                # Extract hour data
                start_idx = hour * 3600
                end_idx = start_idx + 3600
                
                h_full = data['Hcomp']
                d_full = data['Dcomp']
                z_full = data['Zcomp']
                
                if end_idx > len(h_full):
                    end_idx = min(start_idx + 3600, len(h_full))
                
                if start_idx >= len(h_full):
                    print(f"[HOUR {hour:02d}] ❌ Hour data not available")
                    failed_tests += 1
                    continue
                
                h_hour = h_full[start_idx:end_idx]
                d_hour = d_full[start_idx:end_idx]
                z_hour = z_full[start_idx:end_idx]
                
                if len(h_hour) < 100:
                    print(f"[HOUR {hour:02d}] ❌ Insufficient data ({len(h_hour)} samples)")
                    failed_tests += 1
                    continue
                
                # Analyze patterns
                analysis = analyze_precursor_patterns(h_hour, d_hour, z_hour)
                
                hour_result = {
                    'hour': hour,
                    'samples': len(h_hour),
                    'coverage': len(h_hour) / 3600 * 100,
                    'analysis': analysis
                }
                
                results.append(hour_result)
                successful_tests += 1
                
                print(f"[HOUR {hour:02d}] ✅ Analysis complete")
                print(f"   Samples: {len(h_hour)}")
                print(f"   Anomaly Score: {analysis['anomaly_score']:.3f}")
                print(f"   Anomaly Level: {analysis['anomaly_level']}")
                print(f"   PC3 Energy: {analysis['pc3_stats']['total_pc3_energy']:.1f}")
                
            except Exception as e:
                print(f"[HOUR {hour:02d}] ❌ Error: {e}")
                failed_tests += 1
    
    # Generate summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Successful Tests: {successful_tests}")
    print(f"Failed Tests: {failed_tests}")
    print(f"Coverage: {successful_tests/(successful_tests+failed_tests)*100:.1f}%")
    
    if results:
        # Aggregate results
        anomaly_scores = [r['analysis']['anomaly_score'] for r in results]
        avg_anomaly = np.mean(anomaly_scores)
        max_anomaly = np.max(anomaly_scores)
        
        print(f"\nANOMALY ANALYSIS:")
        print(f"  Average Anomaly Score: {avg_anomaly:.3f}")
        print(f"  Maximum Anomaly Score: {max_anomaly:.3f}")
        print(f"  Overall Anomaly Level: {'High' if avg_anomaly > 0.5 else 'Medium' if avg_anomaly > 0.25 else 'Low'}")
        print(f"  Precursor Likelihood: {'High' if max_anomaly > 0.75 else 'Medium' if max_anomaly > 0.5 else 'Low'}")
        
        # PC3 energy analysis
        pc3_energies = [r['analysis']['pc3_stats']['total_pc3_energy'] for r in results]
        avg_pc3 = np.mean(pc3_energies)
        max_pc3 = np.max(pc3_energies)
        
        print(f"\nPC3 ENERGY ANALYSIS:")
        print(f"  Average PC3 Energy: {avg_pc3:.1f}")
        print(f"  Maximum PC3 Energy: {max_pc3:.1f}")
        print(f"  PC3 Activity Level: {'High' if avg_pc3 > 1000 else 'Medium' if avg_pc3 > 500 else 'Low'}")
        
        # Final assessment
        print(f"\nFINAL ASSESSMENT:")
        if max_anomaly > 0.75 or avg_pc3 > 1000:
            assessment = "HIGH PRECURSOR ACTIVITY - Potential earthquake risk"
        elif max_anomaly > 0.5 or avg_pc3 > 500:
            assessment = "MEDIUM PRECURSOR ACTIVITY - Moderate earthquake potential"
        else:
            assessment = "LOW PRECURSOR ACTIVITY - Normal geomagnetic conditions"
        
        print(f"  {assessment}")
        
        # Generate report
        report_file = f"precursor_report_{station}_{test_date.strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EARTHQUAKE PRECURSOR ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TEST INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Date: {test_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Station: {station}\n")
            f.write(f"Hours Tested: {hour_range}\n")
            f.write(f"Successful Tests: {successful_tests}\n")
            f.write(f"Failed Tests: {failed_tests}\n")
            f.write(f"Coverage: {successful_tests/(successful_tests+failed_tests)*100:.1f}%\n\n")
            
            f.write("ANOMALY ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Average Anomaly Score: {avg_anomaly:.3f}\n")
            f.write(f"Maximum Anomaly Score: {max_anomaly:.3f}\n")
            f.write(f"Overall Anomaly Level: {'High' if avg_anomaly > 0.5 else 'Medium' if avg_anomaly > 0.25 else 'Low'}\n")
            f.write(f"Precursor Likelihood: {'High' if max_anomaly > 0.75 else 'Medium' if max_anomaly > 0.5 else 'Low'}\n\n")
            
            f.write("PC3 ENERGY ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Average PC3 Energy: {avg_pc3:.1f}\n")
            f.write(f"Maximum PC3 Energy: {max_pc3:.1f}\n")
            f.write(f"PC3 Activity Level: {'High' if avg_pc3 > 1000 else 'Medium' if avg_pc3 > 500 else 'Low'}\n\n")
            
            f.write("FINAL ASSESSMENT\n")
            f.write("-"*40 + "\n")
            f.write(f"{assessment}\n\n")
            
            f.write("HOURLY DETAILS\n")
            f.write("-"*40 + "\n")
            for result in results:
                f.write(f"Hour {result['hour']:02d}: ")
                f.write(f"Anomaly={result['analysis']['anomaly_score']:.3f}, ")
                f.write(f"PC3={result['analysis']['pc3_stats']['total_pc3_energy']:.1f}, ")
                f.write(f"Level={result['analysis']['anomaly_level']}\n")
        
        print(f"\n📄 Report saved to: {report_file}")
    
    else:
        print("\n❌ No successful tests - cannot generate assessment")
    
    print("="*80)
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test earthquake precursor untuk tanggal tertentu'
    )
    parser.add_argument('--date', required=True,
                       help='Test date (YYYY-MM-DD)')
    parser.add_argument('--station', required=True,
                       help='Station code (e.g., SCN, MLB, ALR)')
    parser.add_argument('--hours', default=None,
                       help='Hours to test (e.g., "0,1,2" or "0-23"), default: all hours')
    
    args = parser.parse_args()
    
    # Parse hours
    if args.hours:
        if '-' in args.hours:
            start, end = map(int, args.hours.split('-'))
            hour_range = list(range(start, end + 1))
        else:
            hour_range = [int(h) for h in args.hours.split(',')]
    else:
        hour_range = list(range(24))  # All hours
    
    # Test precursor
    results = test_precursor_date(args.date, args.station, hour_range)
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)