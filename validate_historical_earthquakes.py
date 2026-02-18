#!/usr/bin/env python3
"""
Historical Earthquake Validation
Validasi model dengan semua earthquake events dari dataset

Tujuan:
- Test model dengan semua earthquake events historis
- Hitung detection rate (berapa % terdeteksi)
- Analyze false positives/negatives
- Generate comprehensive report

Author: Earthquake Prediction Research Team
Date: 2 February 2026
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

# Import production scanner
from prekursor_scanner_production import PrekursorScannerProduction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_historical.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalValidator:
    """
    Validator untuk test model dengan historical earthquake data
    """
    
    def __init__(self):
        """Initialize validator"""
        logger.info("="*70)
        logger.info("HISTORICAL EARTHQUAKE VALIDATION")
        logger.info("="*70)
        
        # Initialize scanner
        logger.info("Initializing production scanner...")
        self.scanner = PrekursorScannerProduction()
        
        # Load earthquake events
        logger.info("Loading earthquake events...")
        self.events = self._load_earthquake_events()
        logger.info(f"✅ Loaded {len(self.events)} earthquake events")
        
        # Results storage
        self.results = []
        
    def _load_earthquake_events(self):
        """Load earthquake events from dataset metadata"""
        # Load from unified dataset
        metadata_file = 'dataset_unified/metadata/train_split.csv'
        
        if not os.path.exists(metadata_file):
            # Try test split
            metadata_file = 'dataset_unified/metadata/test_split.csv'
        
        if not os.path.exists(metadata_file):
            logger.error("❌ Metadata file not found!")
            return pd.DataFrame()
        
        df = pd.read_csv(metadata_file)
        
        # Filter earthquake events only (not Normal)
        earthquake_events = df[df['magnitude_class'] != 'Normal'].copy()
        
        # Get unique events (by station + date)
        earthquake_events['date_only'] = pd.to_datetime(earthquake_events['date']).dt.date
        unique_events = earthquake_events.groupby(['station', 'date_only']).first().reset_index()
        
        logger.info(f"Found {len(unique_events)} unique earthquake events")
        logger.info(f"Stations: {unique_events['station'].nunique()}")
        logger.info(f"Date range: {unique_events['date_only'].min()} to {unique_events['date_only'].max()}")
        
        return unique_events
    
    def validate_event(self, station, date, true_magnitude, true_azimuth):
        """
        Validate single earthquake event
        
        Args:
            station: Station code
            date: Date object
            true_magnitude: True magnitude class
            true_azimuth: True azimuth class
            
        Returns:
            dict with validation results
        """
        date_str = date.strftime('%Y-%m-%d')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Validating: {station} - {date_str}")
        logger.info(f"True: Magnitude={true_magnitude}, Azimuth={true_azimuth}")
        logger.info(f"{'='*70}")
        
        try:
            # Run scanner
            scan_results = self.scanner.scan(date_str, station, save_results=False)
            
            if scan_results is None:
                logger.warning(f"⚠️  Failed to scan {station} {date_str}")
                return {
                    'station': station,
                    'date': date_str,
                    'status': 'failed',
                    'error': 'scan_failed'
                }
            
            # Extract predictions
            predictions = scan_results['predictions']
            pred_magnitude = predictions['magnitude']['class_name']
            pred_azimuth = predictions['azimuth']['class_name']
            pred_mag_conf = predictions['magnitude']['confidence']
            pred_azi_conf = predictions['azimuth']['confidence']
            is_precursor = predictions['is_precursor']
            
            # Check if detected
            detected = is_precursor  # Precursor detected = earthquake detected
            
            # Check if magnitude matches
            mag_correct = (pred_magnitude == true_magnitude)
            
            # Check if azimuth matches
            azi_correct = (pred_azimuth == true_azimuth)
            
            # Overall correctness
            correct = detected and mag_correct
            
            result = {
                'station': station,
                'date': date_str,
                'status': 'success',
                'true_magnitude': true_magnitude,
                'true_azimuth': true_azimuth,
                'pred_magnitude': pred_magnitude,
                'pred_azimuth': pred_azimuth,
                'mag_confidence': pred_mag_conf,
                'azi_confidence': pred_azi_conf,
                'detected': detected,
                'mag_correct': mag_correct,
                'azi_correct': azi_correct,
                'correct': correct,
                'data_coverage': scan_results['data_quality']['coverage']
            }
            
            # Log result
            if detected:
                logger.info(f"✅ DETECTED: {pred_magnitude} ({pred_mag_conf:.1f}%), {pred_azimuth} ({pred_azi_conf:.1f}%)")
                if correct:
                    logger.info(f"✅ CORRECT prediction!")
                else:
                    logger.warning(f"⚠️  INCORRECT: Expected {true_magnitude}, got {pred_magnitude}")
            else:
                logger.warning(f"❌ NOT DETECTED (predicted Normal)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error validating {station} {date_str}: {e}")
            return {
                'station': station,
                'date': date_str,
                'status': 'error',
                'error': str(e)
            }
    
    def validate_all(self, max_events=None, sample_random=False):
        """
        Validate all earthquake events
        
        Args:
            max_events: Maximum number of events to test (None = all)
            sample_random: If True, sample randomly; if False, take first N
        """
        logger.info(f"\n{'='*70}")
        logger.info("STARTING VALIDATION")
        logger.info(f"{'='*70}")
        
        events_to_test = self.events.copy()
        
        if max_events and max_events < len(events_to_test):
            if sample_random:
                events_to_test = events_to_test.sample(n=max_events, random_state=42)
                logger.info(f"Randomly sampling {max_events} events")
            else:
                events_to_test = events_to_test.head(max_events)
                logger.info(f"Testing first {max_events} events")
        
        logger.info(f"Total events to validate: {len(events_to_test)}")
        
        # Validate each event
        for idx, row in tqdm(events_to_test.iterrows(), total=len(events_to_test), desc="Validating"):
            result = self.validate_event(
                station=row['station'],
                date=row['date_only'],
                true_magnitude=row['magnitude_class'],
                true_azimuth=row['azimuth_class']
            )
            
            self.results.append(result)
            
            # Save intermediate results every 10 events
            if len(self.results) % 10 == 0:
                self._save_intermediate_results()
        
        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION COMPLETE")
        logger.info(f"{'='*70}")
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        output_dir = Path('validation_results')
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / 'intermediate_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING REPORT")
        logger.info(f"{'='*70}")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter successful scans
        successful = df[df['status'] == 'success']
        
        if len(successful) == 0:
            logger.error("❌ No successful validations!")
            return
        
        # Calculate metrics
        total_events = len(successful)
        detected = successful['detected'].sum()
        not_detected = total_events - detected
        detection_rate = (detected / total_events) * 100
        
        mag_correct = successful['mag_correct'].sum()
        mag_accuracy = (mag_correct / total_events) * 100
        
        azi_correct = successful['azi_correct'].sum()
        azi_accuracy = (azi_correct / total_events) * 100
        
        overall_correct = successful['correct'].sum()
        overall_accuracy = (overall_correct / total_events) * 100
        
        # Average confidence
        avg_mag_conf = successful['mag_confidence'].mean()
        avg_azi_conf = successful['azi_confidence'].mean()
        
        # Print report
        print("\n" + "="*70)
        print("HISTORICAL VALIDATION REPORT")
        print("="*70)
        
        print(f"\n📊 SUMMARY")
        print(f"   Total events tested: {total_events}")
        print(f"   Successful scans: {len(successful)}")
        print(f"   Failed scans: {len(df) - len(successful)}")
        
        print(f"\n🎯 DETECTION RATE")
        print(f"   Detected: {detected}/{total_events} ({detection_rate:.1f}%)")
        print(f"   Not detected: {not_detected}/{total_events} ({(not_detected/total_events)*100:.1f}%)")
        
        if detection_rate >= 90:
            print(f"   ✅ EXCELLENT detection rate!")
        elif detection_rate >= 70:
            print(f"   ✅ GOOD detection rate")
        elif detection_rate >= 50:
            print(f"   ⚠️  MODERATE detection rate")
        else:
            print(f"   ❌ LOW detection rate - needs improvement")
        
        print(f"\n📏 MAGNITUDE ACCURACY")
        print(f"   Correct: {mag_correct}/{total_events} ({mag_accuracy:.1f}%)")
        print(f"   Average confidence: {avg_mag_conf:.1f}%")
        
        print(f"\n🧭 AZIMUTH ACCURACY")
        print(f"   Correct: {azi_correct}/{total_events} ({azi_accuracy:.1f}%)")
        print(f"   Average confidence: {avg_azi_conf:.1f}%")
        
        print(f"\n⚡ OVERALL ACCURACY")
        print(f"   Correct (detected + magnitude): {overall_correct}/{total_events} ({overall_accuracy:.1f}%)")
        
        # Per-station analysis
        print(f"\n📍 PER-STATION ANALYSIS")
        station_stats = successful.groupby('station').agg({
            'detected': ['count', 'sum'],
            'mag_correct': 'sum',
            'azi_correct': 'sum'
        })
        
        for station in station_stats.index:
            count = station_stats.loc[station, ('detected', 'count')]
            detected_count = station_stats.loc[station, ('detected', 'sum')]
            det_rate = (detected_count / count) * 100
            print(f"   {station}: {detected_count}/{count} detected ({det_rate:.1f}%)")
        
        # Per-magnitude analysis
        print(f"\n📊 PER-MAGNITUDE ANALYSIS")
        mag_stats = successful.groupby('true_magnitude').agg({
            'detected': ['count', 'sum'],
            'mag_correct': 'sum'
        })
        
        for mag_class in mag_stats.index:
            count = mag_stats.loc[mag_class, ('detected', 'count')]
            detected_count = mag_stats.loc[mag_class, ('detected', 'sum')]
            correct_count = mag_stats.loc[mag_class, ('mag_correct', 'sum')]
            det_rate = (detected_count / count) * 100
            acc_rate = (correct_count / count) * 100
            print(f"   {mag_class}: {detected_count}/{count} detected ({det_rate:.1f}%), {correct_count}/{count} correct ({acc_rate:.1f}%)")
        
        # Save detailed results
        output_dir = Path('validation_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON
        results_file = output_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"💾 Results saved to: {results_file}")
        
        # Save CSV
        csv_file = output_dir / 'validation_results.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 CSV saved to: {csv_file}")
        
        # Save summary report
        report_file = output_dir / 'validation_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HISTORICAL VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"SUMMARY\n")
            f.write(f"Total events tested: {total_events}\n")
            f.write(f"Detection rate: {detection_rate:.1f}%\n")
            f.write(f"Magnitude accuracy: {mag_accuracy:.1f}%\n")
            f.write(f"Azimuth accuracy: {azi_accuracy:.1f}%\n")
            f.write(f"Overall accuracy: {overall_accuracy:.1f}%\n")
        logger.info(f"💾 Report saved to: {report_file}")
        
        print(f"\n{'='*70}")
        print("✅ VALIDATION COMPLETE!")
        print(f"{'='*70}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Historical Earthquake Validation'
    )
    parser.add_argument(
        '--max-events', '-n',
        type=int,
        default=None,
        help='Maximum number of events to test (default: all)'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Sample events randomly'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = HistoricalValidator()
    
    # Validate all events
    validator.validate_all(
        max_events=args.max_events,
        sample_random=args.random
    )
    
    # Generate report
    validator.generate_report()


if __name__ == '__main__':
    main()
