#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard
Dashboard untuk monitor kinerja model secara real-time

Features:
- Live performance metrics
- Recent predictions display
- Alert system
- Performance trends
- System health check

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from monitor_production_performance import ProductionMonitor


class RealtimeMonitoringDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self):
        self.monitor = ProductionMonitor(monitoring_dir='../monitoring')
        self.refresh_interval = 5  # seconds
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_recent_predictions(self, hours=24):
        """Get recent predictions"""
        predictions_file = Path('../monitoring/predictions_log.csv')
        
        if not predictions_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(predictions_file)
        
        if len(df) == 0:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = df[df['timestamp'] >= cutoff]
        
        return recent.sort_values('timestamp', ascending=False)
    
    def get_system_health(self):
        """Check system health"""
        health = {
            'status': 'HEALTHY',
            'issues': []
        }
        
        # Check model file
        model_file = Path('../models/earthquake_model.pth')
        if not model_file.exists():
            health['status'] = 'ERROR'
            health['issues'].append('Model file not found')
        
        # Check config
        config_file = Path('../config/production_config.json')
        if not config_file.exists():
            health['status'] = 'WARNING'
            health['issues'].append('Config file not found')
        
        # Check monitoring data
        predictions_file = Path('../monitoring/predictions_log.csv')
        if not predictions_file.exists():
            health['status'] = 'WARNING'
            health['issues'].append('No predictions logged yet')
        
        return health
    
    def display_dashboard(self):
        """Display dashboard"""
        self.clear_screen()
        
        print("="*80)
        print(" "*20 + "EARTHQUAKE PREDICTION SYSTEM")
        print(" "*20 + "Real-time Monitoring Dashboard")
        print("="*80)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # System Health
        health = self.get_system_health()
        status_icon = "✅" if health['status'] == 'HEALTHY' else "⚠️" if health['status'] == 'WARNING' else "❌"
        
        print(f"\n🏥 SYSTEM HEALTH: {status_icon} {health['status']}")
        if health['issues']:
            for issue in health['issues']:
                print(f"   ⚠️  {issue}")
        
        # Recent Predictions
        recent = self.get_recent_predictions(hours=24)
        
        print(f"\n📊 RECENT PREDICTIONS (Last 24 hours)")
        print("-"*80)
        
        if len(recent) == 0:
            print("   No predictions in last 24 hours")
        else:
            print(f"   Total: {len(recent)} predictions")
            
            # Count by status
            precursor_count = recent['is_precursor'].sum()
            normal_count = len(recent) - precursor_count
            
            print(f"   Precursor Detected: {precursor_count}")
            print(f"   Normal: {normal_count}")
            
            # Show last 5 predictions
            print(f"\n   Last 5 Predictions:")
            for idx, row in recent.head(5).iterrows():
                timestamp = row['timestamp']
                station = row['station']
                date = row['date']
                mag = row['pred_magnitude']
                conf = row['mag_confidence']
                status = "⚠️ PRECURSOR" if row['is_precursor'] else "✅ NORMAL"
                
                print(f"   {timestamp} | {station} {date} | {mag} ({conf:.1f}%) | {status}")
        
        # Performance Metrics (if verified predictions exist)
        metrics = self.monitor.calculate_metrics(time_window_days=30)
        
        if metrics:
            print(f"\n📈 PERFORMANCE METRICS (Last 30 days)")
            print("-"*80)
            print(f"   Verified Predictions: {metrics['verified_predictions']}")
            print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"   Precision: {metrics['precision']*100:.1f}%")
            print(f"   Recall: {metrics['recall']*100:.1f}%")
            print(f"   F1 Score: {metrics['f1_score']:.3f}")
            
            # Alerts
            alerts = []
            if metrics['false_positives'] > metrics['true_positives']:
                alerts.append("⚠️  High false positive rate")
            if metrics['false_negatives'] > 0:
                alerts.append("⚠️  Missed earthquake events")
            if metrics['precision'] < 0.5:
                alerts.append("⚠️  Low precision")
            if metrics['recall'] < 0.7:
                alerts.append("⚠️  Low recall")
            
            if alerts:
                print(f"\n🚨 ALERTS")
                for alert in alerts:
                    print(f"   {alert}")
        else:
            print(f"\n📈 PERFORMANCE METRICS")
            print("-"*80)
            print("   No verified predictions yet")
            print("   Run predictions and verify them to see metrics")
        
        # Model Info
        print(f"\n🤖 MODEL INFORMATION")
        print("-"*80)
        print(f"   Version: 2.1 (Final)")
        print(f"   Test Magnitude Accuracy: 98.94%")
        print(f"   Test Azimuth Accuracy: 83.92%")
        print(f"   Test Normal Detection: 100.00%")
        print(f"   LOEO Validation: 97.53% (Magnitude)")
        print(f"   LOSO Validation: 97.57% (Magnitude)")
        print(f"   Status: PRODUCTION")
        
        # Instructions
        print(f"\n💡 QUICK ACTIONS")
        print("-"*80)
        print(f"   Run Scanner: python prekursor_scanner_production.py --station SCN --date 2018-01-17")
        print(f"   Generate Report: python monitor_production_performance.py --report")
        print(f"   Plot Trends: python monitor_production_performance.py --plot")
        
        print("\n" + "="*80)
        print(f"Press Ctrl+C to exit | Refreshing every {self.refresh_interval} seconds")
        print("="*80)
    
    def run(self):
        """Run dashboard"""
        print("Starting Real-time Monitoring Dashboard...")
        print("Press Ctrl+C to exit")
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            print("Thank you for using Earthquake Prediction System!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time Monitoring Dashboard'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Refresh interval in seconds (default: 5)'
    )
    
    args = parser.parse_args()
    
    dashboard = RealtimeMonitoringDashboard()
    dashboard.refresh_interval = args.refresh
    dashboard.run()


if __name__ == '__main__':
    main()
