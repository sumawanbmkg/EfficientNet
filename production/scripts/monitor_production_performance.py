#!/usr/bin/env python3
"""
Production Performance Monitoring System
Monitor kinerja model di production secara real-time

Features:
- Track predictions over time
- Calculate accuracy metrics
- Detect performance degradation
- Generate monitoring reports
- Alert system for anomalies

Author: Earthquake Prediction Research Team
Date: 2 February 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionMonitor:
    """
    Monitor untuk track kinerja model di production
    """
    
    def __init__(self, monitoring_dir='production_monitoring'):
        """
        Initialize monitor
        
        Args:
            monitoring_dir: Directory untuk simpan monitoring data
        """
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Predictions log file
        self.predictions_file = self.monitoring_dir / 'predictions_log.csv'
        
        # Initialize predictions log if not exists
        if not self.predictions_file.exists():
            self._initialize_predictions_log()
        
        logger.info("✅ Production Monitor initialized")
        logger.info(f"   Monitoring directory: {self.monitoring_dir}")
    
    def _initialize_predictions_log(self):
        """Initialize predictions log file"""
        df = pd.DataFrame(columns=[
            'timestamp',
            'station',
            'date',
            'pred_magnitude',
            'pred_azimuth',
            'mag_confidence',
            'azi_confidence',
            'is_precursor',
            'data_coverage',
            'true_magnitude',
            'true_azimuth',
            'earthquake_occurred',
            'days_to_earthquake',
            'verified'
        ])
        df.to_csv(self.predictions_file, index=False)
        logger.info(f"✅ Initialized predictions log: {self.predictions_file}")
    
    def log_prediction(self, station, date, predictions, data_quality):
        """
        Log prediction ke monitoring system
        
        Args:
            station: Station code
            date: Date string
            predictions: Prediction results dict
            data_quality: Data quality dict
        """
        # Create log entry
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'station': station,
            'date': date,
            'pred_magnitude': predictions['magnitude']['class_name'],
            'pred_azimuth': predictions['azimuth']['class_name'],
            'mag_confidence': predictions['magnitude']['confidence'],
            'azi_confidence': predictions['azimuth']['confidence'],
            'is_precursor': predictions['is_precursor'],
            'data_coverage': data_quality['coverage'],
            'true_magnitude': '',  # To be filled later
            'true_azimuth': '',
            'earthquake_occurred': False,
            'days_to_earthquake': np.nan,
            'verified': False
        }
        
        # Append to log
        df = pd.read_csv(self.predictions_file)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.predictions_file, index=False)
        
        logger.info(f"✅ Logged prediction: {station} {date}")
    
    def verify_prediction(self, station, date, true_magnitude, true_azimuth, 
                         earthquake_occurred, days_to_earthquake=None):
        """
        Verify prediction dengan actual earthquake data
        
        Args:
            station: Station code
            date: Date string
            true_magnitude: True magnitude class
            true_azimuth: True azimuth class
            earthquake_occurred: Boolean
            days_to_earthquake: Days until earthquake (if occurred)
        """
        df = pd.read_csv(self.predictions_file)
        
        # Find prediction
        mask = (df['station'] == station) & (df['date'] == date)
        
        if not mask.any():
            logger.warning(f"⚠️  Prediction not found: {station} {date}")
            return
        
        # Update with true values
        df.loc[mask, 'true_magnitude'] = true_magnitude
        df.loc[mask, 'true_azimuth'] = true_azimuth
        df.loc[mask, 'earthquake_occurred'] = earthquake_occurred
        if days_to_earthquake is not None:
            df.loc[mask, 'days_to_earthquake'] = days_to_earthquake
        df.loc[mask, 'verified'] = True
        
        # Save
        df.to_csv(self.predictions_file, index=False)
        
        logger.info(f"✅ Verified prediction: {station} {date}")
    
    def calculate_metrics(self, time_window_days=30):
        """
        Calculate performance metrics
        
        Args:
            time_window_days: Time window untuk calculate metrics (days)
            
        Returns:
            dict with metrics
        """
        df = pd.read_csv(self.predictions_file)
        
        # Filter verified predictions
        verified = df[df['verified'] == True].copy()
        
        if len(verified) == 0:
            logger.warning("⚠️  No verified predictions yet")
            return None
        
        # Filter by time window
        verified['timestamp'] = pd.to_datetime(verified['timestamp'])
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent = verified[verified['timestamp'] >= cutoff_date]
        
        if len(recent) == 0:
            logger.warning(f"⚠️  No verified predictions in last {time_window_days} days")
            recent = verified  # Use all data
        
        # Calculate metrics
        total = len(recent)
        
        # Detection metrics
        precursor_detected = recent['is_precursor'].sum()
        earthquake_occurred = recent['earthquake_occurred'].sum()
        
        # True positives: predicted precursor AND earthquake occurred
        true_positives = ((recent['is_precursor'] == True) & 
                         (recent['earthquake_occurred'] == True)).sum()
        
        # False positives: predicted precursor BUT no earthquake
        false_positives = ((recent['is_precursor'] == True) & 
                          (recent['earthquake_occurred'] == False)).sum()
        
        # False negatives: predicted normal BUT earthquake occurred
        false_negatives = ((recent['is_precursor'] == False) & 
                          (recent['earthquake_occurred'] == True)).sum()
        
        # True negatives: predicted normal AND no earthquake
        true_negatives = ((recent['is_precursor'] == False) & 
                         (recent['earthquake_occurred'] == False)).sum()
        
        # Calculate rates
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        accuracy = (true_positives + true_negatives) / total
        
        # Magnitude accuracy (for earthquakes only)
        earthquakes = recent[recent['earthquake_occurred'] == True]
        if len(earthquakes) > 0:
            mag_correct = (earthquakes['pred_magnitude'] == earthquakes['true_magnitude']).sum()
            mag_accuracy = mag_correct / len(earthquakes)
        else:
            mag_accuracy = 0.0
        
        # Azimuth accuracy (for earthquakes only)
        if len(earthquakes) > 0:
            azi_correct = (earthquakes['pred_azimuth'] == earthquakes['true_azimuth']).sum()
            azi_accuracy = azi_correct / len(earthquakes)
        else:
            azi_accuracy = 0.0
        
        # Average confidence
        avg_mag_conf = recent['mag_confidence'].mean()
        avg_azi_conf = recent['azi_confidence'].mean()
        
        # Lead time (days before earthquake)
        if len(earthquakes) > 0:
            avg_lead_time = earthquakes['days_to_earthquake'].mean()
        else:
            avg_lead_time = np.nan
        
        metrics = {
            'time_window_days': time_window_days,
            'total_predictions': total,
            'verified_predictions': len(recent),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'magnitude_accuracy': mag_accuracy,
            'azimuth_accuracy': azi_accuracy,
            'avg_mag_confidence': avg_mag_conf,
            'avg_azi_confidence': avg_azi_conf,
            'avg_lead_time_days': avg_lead_time
        }
        
        return metrics
    
    def generate_report(self, time_window_days=30):
        """
        Generate monitoring report
        
        Args:
            time_window_days: Time window untuk report (days)
        """
        logger.info(f"\n{'='*70}")
        logger.info("PRODUCTION MONITORING REPORT")
        logger.info(f"{'='*70}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(time_window_days)
        
        if metrics is None:
            logger.warning("⚠️  Cannot generate report - no verified predictions")
            return
        
        # Print report
        print(f"\n📊 MONITORING PERIOD")
        print(f"   Time window: Last {time_window_days} days")
        print(f"   Total predictions: {metrics['total_predictions']}")
        print(f"   Verified predictions: {metrics['verified_predictions']}")
        
        print(f"\n🎯 DETECTION PERFORMANCE")
        print(f"   True Positives: {metrics['true_positives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        print(f"   True Negatives: {metrics['true_negatives']}")
        
        print(f"\n📈 METRICS")
        print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"   Precision: {metrics['precision']*100:.1f}%")
        print(f"   Recall: {metrics['recall']*100:.1f}%")
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        
        print(f"\n📏 CLASSIFICATION ACCURACY")
        print(f"   Magnitude: {metrics['magnitude_accuracy']*100:.1f}%")
        print(f"   Azimuth: {metrics['azimuth_accuracy']*100:.1f}%")
        
        print(f"\n🔮 CONFIDENCE SCORES")
        print(f"   Magnitude: {metrics['avg_mag_confidence']:.1f}%")
        print(f"   Azimuth: {metrics['avg_azi_confidence']:.1f}%")
        
        if not np.isnan(metrics['avg_lead_time_days']):
            print(f"\n⏰ LEAD TIME")
            print(f"   Average: {metrics['avg_lead_time_days']:.1f} days before earthquake")
        
        # Performance assessment
        print(f"\n⚡ OVERALL ASSESSMENT")
        if metrics['accuracy'] >= 0.9:
            print(f"   ✅ EXCELLENT performance!")
        elif metrics['accuracy'] >= 0.7:
            print(f"   ✅ GOOD performance")
        elif metrics['accuracy'] >= 0.5:
            print(f"   ⚠️  MODERATE performance")
        else:
            print(f"   ❌ LOW performance - needs attention")
        
        # Alerts
        alerts = []
        if metrics['false_positives'] > metrics['true_positives']:
            alerts.append("⚠️  High false positive rate")
        if metrics['false_negatives'] > 0:
            alerts.append("⚠️  Missed earthquake events (false negatives)")
        if metrics['precision'] < 0.5:
            alerts.append("⚠️  Low precision - too many false alarms")
        if metrics['recall'] < 0.7:
            alerts.append("⚠️  Low recall - missing earthquake events")
        
        if alerts:
            print(f"\n🚨 ALERTS")
            for alert in alerts:
                print(f"   {alert}")
        
        # Save report
        report_file = self.monitoring_dir / f'monitoring_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\n💾 Report saved to: {report_file}")
        
        print(f"\n{'='*70}")
        print("✅ REPORT COMPLETE!")
        print(f"{'='*70}\n")
    
    def plot_performance_trends(self, save_path=None):
        """
        Plot performance trends over time
        
        Args:
            save_path: Path to save plot
        """
        df = pd.read_csv(self.predictions_file)
        verified = df[df['verified'] == True].copy()
        
        if len(verified) == 0:
            logger.warning("⚠️  No verified predictions to plot")
            return
        
        verified['timestamp'] = pd.to_datetime(verified['timestamp'])
        verified = verified.sort_values('timestamp')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Production Performance Trends', fontsize=16, fontweight='bold')
        
        # 1. Predictions over time
        ax1 = axes[0, 0]
        verified['date_only'] = verified['timestamp'].dt.date
        daily_counts = verified.groupby('date_only').size()
        ax1.plot(daily_counts.index, daily_counts.values, marker='o')
        ax1.set_title('Predictions per Day')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Detection rate over time
        ax2 = axes[0, 1]
        daily_detection = verified.groupby('date_only')['is_precursor'].mean() * 100
        ax2.plot(daily_detection.index, daily_detection.values, marker='o', color='orange')
        ax2.set_title('Detection Rate over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Detection Rate (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Confidence scores over time
        ax3 = axes[1, 0]
        ax3.plot(verified['timestamp'], verified['mag_confidence'], 
                label='Magnitude', alpha=0.7)
        ax3.plot(verified['timestamp'], verified['azi_confidence'], 
                label='Azimuth', alpha=0.7)
        ax3.set_title('Confidence Scores over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Confidence (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Confusion matrix (latest 30 days)
        ax4 = axes[1, 1]
        recent = verified[verified['timestamp'] >= (datetime.now() - timedelta(days=30))]
        
        tp = ((recent['is_precursor'] == True) & (recent['earthquake_occurred'] == True)).sum()
        fp = ((recent['is_precursor'] == True) & (recent['earthquake_occurred'] == False)).sum()
        fn = ((recent['is_precursor'] == False) & (recent['earthquake_occurred'] == True)).sum()
        tn = ((recent['is_precursor'] == False) & (recent['earthquake_occurred'] == False)).sum()
        
        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        im = ax4.imshow(confusion_matrix, cmap='Blues')
        
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Earthquake', 'Normal'])
        ax4.set_yticklabels(['Predicted\nEarthquake', 'Predicted\nNormal'])
        ax4.set_title('Confusion Matrix (Last 30 Days)')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="black", fontsize=14)
        
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production Performance Monitoring'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate monitoring report'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot performance trends'
    )
    parser.add_argument(
        '--time-window',
        type=int,
        default=30,
        help='Time window in days (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ProductionMonitor()
    
    if args.report:
        # Generate report
        monitor.generate_report(time_window_days=args.time_window)
    
    if args.plot:
        # Plot trends
        plot_path = monitor.monitoring_dir / f'performance_trends_{datetime.now().strftime("%Y%m%d")}.png'
        monitor.plot_performance_trends(save_path=plot_path)
    
    if not args.report and not args.plot:
        print("\nProduction Monitor initialized!")
        print("Use --report to generate monitoring report")
        print("Use --plot to plot performance trends")


if __name__ == '__main__':
    main()
