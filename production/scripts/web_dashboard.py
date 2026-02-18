#!/usr/bin/env python3
"""
Web Dashboard - Earthquake Prediction System
Dashboard berbasis web untuk monitoring sistem

Features:
- Web-based interface
- Real-time updates
- Performance charts
- System status
- Recent predictions

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
from flask import Flask, render_template, jsonify

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from monitor_production_performance import ProductionMonitor

app = Flask(__name__)

# Initialize monitor with absolute path
try:
    script_dir = Path(__file__).parent
    monitoring_dir = script_dir.parent / 'monitoring'
    monitor = ProductionMonitor(monitoring_dir=str(monitoring_dir))
except Exception as e:
    print(f"Warning: Could not initialize monitor: {e}")
    monitor = None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/system_health')
def system_health():
    """Get system health status"""
    health = {
        'status': 'HEALTHY',
        'issues': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    model_file = script_dir.parent / 'models' / 'earthquake_model.pth'
    config_file = script_dir.parent / 'config' / 'production_config.json'
    
    # Check model file
    if not model_file.exists():
        health['status'] = 'ERROR'
        health['issues'].append(f'Model file not found: {model_file}')
    
    # Check config
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            health['model_version'] = config.get('model_version', 'Unknown')
            health['deployment_date'] = config.get('deployment_date', 'Unknown')
    else:
        health['status'] = 'WARNING'
        health['issues'].append(f'Config file not found: {config_file}')
    
    return jsonify(health)


@app.route('/api/model_info')
def model_info():
    """Get model information"""
    script_dir = Path(__file__).parent
    config_file = script_dir.parent / 'config' / 'production_config.json'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        info = {
            'version': config.get('model_version', '2.0'),
            'deployment_date': config.get('deployment_date', '2026-02-03'),
            'status': config.get('deployment_status', 'PRODUCTION'),
            'confidence_level': config.get('confidence_level', 99),
            'performance': config.get('performance_metrics', {})
        }
        
        return jsonify(info)
    
    return jsonify({'error': 'Config not found'}), 404


@app.route('/api/recent_predictions')
def recent_predictions():
    """Get recent predictions"""
    try:
        script_dir = Path(__file__).parent
        predictions_file = script_dir.parent / 'monitoring' / 'predictions_log.csv'
        
        if not predictions_file.exists():
            return jsonify({'predictions': [], 'count': 0})
        
        df = pd.read_csv(predictions_file)
        
        if len(df) == 0:
            return jsonify({'predictions': [], 'count': 0})
        
        # Get last 24 hours
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent = df[df['timestamp'] >= (datetime.now() - timedelta(hours=24))]
        
        # Convert to dict
        predictions = recent.sort_values('timestamp', ascending=False).head(10).to_dict('records')
        
        # Format timestamps
        for pred in predictions:
            if isinstance(pred['timestamp'], pd.Timestamp):
                pred['timestamp'] = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'predictions': predictions,
            'count': len(recent)
        })
    except Exception as e:
        return jsonify({'predictions': [], 'count': 0, 'error': str(e)})


@app.route('/api/performance_metrics')
def performance_metrics():
    """Get performance metrics"""
    try:
        if monitor is None:
            return jsonify({
                'status': 'no_data',
                'message': 'Monitor not initialized'
            })
        
        metrics = monitor.calculate_metrics(time_window_days=30)
        
        if metrics is None:
            return jsonify({
                'status': 'no_data',
                'message': 'No verified predictions yet'
            })
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/api/statistics')
def statistics():
    """Get overall statistics"""
    try:
        script_dir = Path(__file__).parent
        predictions_file = script_dir.parent / 'monitoring' / 'predictions_log.csv'
        
        if not predictions_file.exists():
            return jsonify({
                'total_predictions': 0,
                'precursor_detections': 0,
                'normal_predictions': 0,
                'verified_predictions': 0
            })
        
        df = pd.read_csv(predictions_file)
        
        if len(df) == 0:
            return jsonify({
                'total_predictions': 0,
                'precursor_detections': 0,
                'normal_predictions': 0,
                'verified_predictions': 0
            })
        
        stats = {
            'total_predictions': len(df),
            'precursor_detections': int(df['is_precursor'].sum()) if 'is_precursor' in df.columns else 0,
            'normal_predictions': int((~df['is_precursor']).sum()) if 'is_precursor' in df.columns else 0,
            'verified_predictions': int(df['verified'].sum()) if 'verified' in df.columns else 0
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'total_predictions': 0,
            'precursor_detections': 0,
            'normal_predictions': 0,
            'verified_predictions': 0,
            'error': str(e)
        })


def create_html_template():
    """Create HTML template for dashboard"""
    template_dir = Path('templates')
    template_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Earthquake Prediction System - Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .status.healthy {
            background: #10b981;
            color: white;
        }
        
        .status.warning {
            background: #f59e0b;
            color: white;
        }
        
        .status.error {
            background: #ef4444;
            color: white;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #666;
        }
        
        .metric-value {
            font-weight: bold;
            color: #667eea;
        }
        
        .prediction-item {
            padding: 15px;
            background: #f9fafb;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .prediction-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .prediction-station {
            font-weight: bold;
            color: #667eea;
        }
        
        .prediction-time {
            color: #999;
            font-size: 0.9em;
        }
        
        .prediction-result {
            color: #666;
        }
        
        .precursor-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .precursor-badge.yes {
            background: #fef3c7;
            color: #92400e;
        }
        
        .precursor-badge.no {
            background: #d1fae5;
            color: #065f46;
        }
        
        .refresh-info {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 Earthquake Prediction System</h1>
            <p>Real-time Monitoring Dashboard - Version 2.0</p>
            <p id="last-update">Last Updated: Loading...</p>
        </div>
        
        <div class="grid">
            <!-- System Health -->
            <div class="card">
                <h2>🏥 System Health</h2>
                <div id="system-health">
                    <p>Loading...</p>
                </div>
            </div>
            
            <!-- Model Info -->
            <div class="card">
                <h2>🤖 Model Information</h2>
                <div id="model-info">
                    <p>Loading...</p>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="card">
                <h2>📊 Statistics</h2>
                <div id="statistics">
                    <p>Loading...</p>
                </div>
            </div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>📈 Performance Metrics (Last 30 Days)</h2>
            <div id="performance-metrics">
                <p>Loading...</p>
            </div>
        </div>
        
        <!-- Recent Predictions -->
        <div class="card">
            <h2>🔮 Recent Predictions (Last 24 Hours)</h2>
            <div id="recent-predictions">
                <p>Loading...</p>
            </div>
        </div>
        
        <div class="refresh-info">
            Auto-refresh every 10 seconds
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            // Update timestamp
            document.getElementById('last-update').textContent = 
                'Last Updated: ' + new Date().toLocaleString();
            
            // Fetch system health
            fetch('/api/system_health')
                .then(response => response.json())
                .then(data => {
                    let statusClass = data.status.toLowerCase();
                    let html = `
                        <div class="metric">
                            <span class="metric-label">Status:</span>
                            <span class="status ${statusClass}">${data.status}</span>
                        </div>
                    `;
                    
                    if (data.model_version) {
                        html += `
                            <div class="metric">
                                <span class="metric-label">Model Version:</span>
                                <span class="metric-value">${data.model_version}</span>
                            </div>
                        `;
                    }
                    
                    if (data.issues.length > 0) {
                        html += '<div style="margin-top: 10px; color: #ef4444;">';
                        data.issues.forEach(issue => {
                            html += `<p>⚠️ ${issue}</p>`;
                        });
                        html += '</div>';
                    }
                    
                    document.getElementById('system-health').innerHTML = html;
                });
            
            // Fetch model info
            fetch('/api/model_info')
                .then(response => response.json())
                .then(data => {
                    let html = `
                        <div class="metric">
                            <span class="metric-label">Version:</span>
                            <span class="metric-value">${data.version}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Status:</span>
                            <span class="metric-value">${data.status}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Confidence:</span>
                            <span class="metric-value">${data.confidence_level}%</span>
                        </div>
                    `;
                    
                    if (data.performance) {
                        html += `
                            <div class="metric">
                                <span class="metric-label">Test Accuracy:</span>
                                <span class="metric-value">${data.performance.test_magnitude_accuracy}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Detection Rate:</span>
                                <span class="metric-value">${data.performance.validation_detection_rate}%</span>
                            </div>
                        `;
                    }
                    
                    document.getElementById('model-info').innerHTML = html;
                });
            
            // Fetch statistics
            fetch('/api/statistics')
                .then(response => response.json())
                .then(data => {
                    let html = `
                        <div class="metric">
                            <span class="metric-label">Total Predictions:</span>
                            <span class="metric-value">${data.total_predictions}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Precursor Detections:</span>
                            <span class="metric-value">${data.precursor_detections}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Normal Predictions:</span>
                            <span class="metric-value">${data.normal_predictions}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Verified:</span>
                            <span class="metric-value">${data.verified_predictions}</span>
                        </div>
                    `;
                    
                    document.getElementById('statistics').innerHTML = html;
                });
            
            // Fetch performance metrics
            fetch('/api/performance_metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'no_data') {
                        document.getElementById('performance-metrics').innerHTML = 
                            '<p style="color: #999;">No verified predictions yet. Run predictions and verify them to see metrics.</p>';
                        return;
                    }
                    
                    let html = `
                        <div class="grid">
                            <div class="metric">
                                <span class="metric-label">Accuracy:</span>
                                <span class="metric-value">${(data.accuracy * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Precision:</span>
                                <span class="metric-value">${(data.precision * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Recall:</span>
                                <span class="metric-value">${(data.recall * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">F1 Score:</span>
                                <span class="metric-value">${data.f1_score.toFixed(3)}</span>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('performance-metrics').innerHTML = html;
                });
            
            // Fetch recent predictions
            fetch('/api/recent_predictions')
                .then(response => response.json())
                .then(data => {
                    if (data.predictions.length === 0) {
                        document.getElementById('recent-predictions').innerHTML = 
                            '<p style="color: #999;">No predictions in last 24 hours</p>';
                        return;
                    }
                    
                    let html = '';
                    data.predictions.forEach(pred => {
                        let precursorClass = pred.is_precursor ? 'yes' : 'no';
                        let precursorText = pred.is_precursor ? 'PRECURSOR' : 'NORMAL';
                        
                        html += `
                            <div class="prediction-item">
                                <div class="prediction-header">
                                    <span class="prediction-station">${pred.station} - ${pred.date}</span>
                                    <span class="prediction-time">${pred.timestamp}</span>
                                </div>
                                <div class="prediction-result">
                                    Magnitude: ${pred.pred_magnitude} (${pred.mag_confidence.toFixed(1)}%)
                                    <span class="precursor-badge ${precursorClass}">${precursorText}</span>
                                </div>
                            </div>
                        `;
                    });
                    
                    document.getElementById('recent-predictions').innerHTML = html;
                });
        }
        
        // Initial load
        updateDashboard();
        
        // Auto-refresh every 10 seconds
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>
    """
    
    template_file = template_dir / 'dashboard.html'
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML template created: {template_file}")


if __name__ == '__main__':
    print("="*70)
    print("WEB DASHBOARD - Earthquake Prediction System")
    print("="*70)
    
    # Create HTML template
    create_html_template()
    
    print("\n🌐 Starting web server...")
    print("📊 Dashboard URL: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
