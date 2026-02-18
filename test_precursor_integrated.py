#!/usr/bin/env python3
"""
Integrated Precursor Test with CNN
Complete earthquake precursor testing system with CNN predictions
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
import torch
import torch.nn.functional as F
from PIL import Image
import argparse

# Add intial to path
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher

# Import model components
from multi_task_cnn_model import create_model
from earthquake_dataset import get_transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedPrecursorTester:
    """Integrated earthquake precursor tester with CNN predictions"""
    
    def __init__(self, model_path=None, device='cpu'):
        """Initialize tester"""
        self.device = torch.device(device)
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0
        self.image_size = 224
        
        # Class mappings
        self.magnitude_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
        self.azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        self.magnitude_to_value = {
            'Small': 3.5, 'Moderate': 4.5, 'Medium': 5.5, 
            'Large': 6.5, 'Major': 7.5
        }
        
        self.azimuth_to_degrees = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Transform for preprocessing
        self.transform = get_transforms(augment=False)
        
        logger.info(f"IntegratedPrecursorTester initialized on {self.device}")
    
    def load_model(self, model_path):
        """Load trained CNN model"""
        try:
            # Model config
            config = {
                'backbone': 'resnet50',
                'pretrained': True,
                'num_magnitude_classes': 5,
                'num_azimuth_classes': 8,
                'dropout_rate': 0.5,
                'learn_weights': True
            }
            
            # Create model
            self.model, _ = create_model(config)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Model loaded from checkpoint")
                if 'best_val_loss' in checkpoint:
                    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"Model state dict loaded")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"CNN model loaded successfully: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise
    
    def apply_pc3_filter(self, data):
        """Apply PC3 bandpass filter"""
        if len(data) < 100:
            return data
        
        data_clean = np.nan_to_num(data, nan=np.nanmean(data))
        
        nyquist = self.sampling_rate / 2
        low = max(0.001, min(self.pc3_low / nyquist, 0.999))
        high = max(0.001, min(self.pc3_high / nyquist, 0.999))
        
        if low >= high:
            return data_clean
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, data_clean)
            return filtered
        except:
            return data_clean
    
    def generate_spectrogram(self, data):
        """Generate spectrogram"""
        nperseg = min(256, len(data) // 4)
        noverlap = nperseg // 2
        
        f, t, Sxx = spectrogram(
            data, fs=self.sampling_rate,
            window='hann', nperseg=nperseg,
            noverlap=noverlap, scaling='density'
        )
        
        return f, t, Sxx
    
    def create_spectrogram_image(self, f, t, Sxx_h, Sxx_d, Sxx_z):
        """Create spectrogram image for CNN (224×224 RGB, no axis)"""
        fig_size = self.image_size / 100.0
        fig, axes = plt.subplots(3, 1, figsize=(fig_size, fig_size))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        
        # Limit to PC3 range
        freq_mask = (f >= self.pc3_low) & (f <= self.pc3_high)
        f_pc3 = f[freq_mask]
        Sxx_h_pc3 = Sxx_h[freq_mask, :]
        Sxx_d_pc3 = Sxx_d[freq_mask, :]
        Sxx_z_pc3 = Sxx_z[freq_mask, :]
        
        # Convert to dB
        Sxx_h_db = 10 * np.log10(Sxx_h_pc3 + 1e-10)
        Sxx_d_db = 10 * np.log10(Sxx_d_pc3 + 1e-10)
        Sxx_z_db = 10 * np.log10(Sxx_z_pc3 + 1e-10)
        
        # Plot (NO axis, NO text)
        axes[0].pcolormesh(t, f_pc3, Sxx_h_db, shading='gouraud', cmap='jet')
        axes[0].axis('off')
        
        axes[1].pcolormesh(t, f_pc3, Sxx_d_db, shading='gouraud', cmap='jet')
        axes[1].axis('off')
        
        axes[2].pcolormesh(t, f_pc3, Sxx_z_db, shading='gouraud', cmap='jet')
        axes[2].axis('off')
        
        # Save to temporary file
        temp_path = 'temp_spectrogram.png'
        plt.savefig(temp_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load and resize
        img = Image.open(temp_path)
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return img
    
    def predict_earthquake(self, image):
        """Predict earthquake from spectrogram image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image).unsqueeze(0)
        else:
            image_tensor = image.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            mag_logits, az_logits = self.model(image_tensor)
            
            # Get probabilities
            mag_probs = F.softmax(mag_logits, dim=1)
            az_probs = F.softmax(az_logits, dim=1)
            
            # Get predictions
            mag_pred = mag_logits.argmax(dim=1).item()
            az_pred = az_logits.argmax(dim=1).item()
            
            # Get confidence scores
            mag_confidence = mag_probs[0, mag_pred].item()
            az_confidence = az_probs[0, az_pred].item()
            
            # Get class names
            mag_class = self.magnitude_classes[mag_pred]
            az_class = self.azimuth_classes[az_pred]
            
            # Convert to actual values
            mag_value = self.magnitude_to_value[mag_class]
            az_degrees = self.azimuth_to_degrees[az_class]
            
            results = {
                'magnitude': {
                    'class': mag_class,
                    'value': mag_value,
                    'confidence': mag_confidence,
                    'probabilities': mag_probs[0].cpu().numpy()
                },
                'azimuth': {
                    'class': az_class,
                    'degrees': az_degrees,
                    'confidence': az_confidence,
                    'probabilities': az_probs[0].cpu().numpy()
                }
            }
            
            return results
    
    def analyze_precursor_patterns(self, h_data, d_data, z_data):
        """Analyze precursor patterns"""
        # Basic statistics
        stats = {
            'H': {'mean': np.mean(h_data), 'std': np.std(h_data), 'range': np.max(h_data) - np.min(h_data)},
            'D': {'mean': np.mean(d_data), 'std': np.std(d_data), 'range': np.max(d_data) - np.min(d_data)},
            'Z': {'mean': np.mean(z_data), 'std': np.std(z_data), 'range': np.max(z_data) - np.min(z_data)}
        }
        
        # PC3 analysis
        h_pc3 = self.apply_pc3_filter(h_data)
        d_pc3 = self.apply_pc3_filter(d_data)
        z_pc3 = self.apply_pc3_filter(z_data)
        
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
    
    def test_date(self, test_date, station, hour_range):
        """Test specific date for precursor activity"""
        if isinstance(test_date, str):
            test_date = datetime.strptime(test_date, '%Y-%m-%d')
        
        logger.info("="*80)
        logger.info("INTEGRATED EARTHQUAKE PRECURSOR TEST")
        logger.info("="*80)
        logger.info(f"Date: {test_date.strftime('%Y-%m-%d')}")
        logger.info(f"Station: {station}")
        logger.info(f"Hours: {hour_range}")
        logger.info(f"CNN Model: {'Loaded' if self.model else 'Not loaded'}")
        logger.info("="*80)
        
        results = {
            'test_info': {
                'date': test_date.strftime('%Y-%m-%d'),
                'station': station,
                'hours_tested': hour_range,
                'model_used': self.model is not None
            },
            'hourly_results': [],
            'summary': {}
        }
        
        successful_tests = 0
        failed_tests = 0
        
        with GeomagneticDataFetcher() as fetcher:
            logger.info("[OK] SSH Connection established!")
            
            for hour in hour_range:
                logger.info(f"\n[HOUR {hour:02d}] Testing {test_date.strftime('%Y-%m-%d')} {hour:02d}:00...")
                
                try:
                    # Fetch data
                    date = test_date
                    data = fetcher.fetch_data(date, station)
                    
                    if data is None:
                        logger.warning(f"[HOUR {hour:02d}] Failed to fetch data")
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
                        logger.warning(f"[HOUR {hour:02d}] Hour data not available")
                        failed_tests += 1
                        continue
                    
                    h_hour = h_full[start_idx:end_idx]
                    d_hour = d_full[start_idx:end_idx]
                    z_hour = z_full[start_idx:end_idx]
                    
                    if len(h_hour) < 100:
                        logger.warning(f"[HOUR {hour:02d}] Insufficient data ({len(h_hour)} samples)")
                        failed_tests += 1
                        continue
                    
                    # Pattern analysis
                    pattern_analysis = self.analyze_precursor_patterns(h_hour, d_hour, z_hour)
                    
                    hour_result = {
                        'hour': hour,
                        'data_quality': {
                            'samples': len(h_hour),
                            'coverage': len(h_hour) / 3600 * 100
                        },
                        'pattern_analysis': pattern_analysis
                    }
                    
                    # CNN Prediction (if model available)
                    if self.model is not None:
                        try:
                            # Apply PC3 filter
                            h_pc3 = self.apply_pc3_filter(h_hour)
                            d_pc3 = self.apply_pc3_filter(d_hour)
                            z_pc3 = self.apply_pc3_filter(z_hour)
                            
                            # Generate spectrograms
                            f_h, t_h, Sxx_h = self.generate_spectrogram(h_pc3)
                            f_d, t_d, Sxx_d = self.generate_spectrogram(d_pc3)
                            f_z, t_z, Sxx_z = self.generate_spectrogram(z_pc3)
                            
                            # Create image
                            spec_image = self.create_spectrogram_image(f_h, t_h, Sxx_h, Sxx_d, Sxx_z)
                            
                            # Predict
                            prediction = self.predict_earthquake(spec_image)
                            hour_result['cnn_prediction'] = prediction
                            
                            logger.info(f"[HOUR {hour:02d}] CNN Prediction: Mag {prediction['magnitude']['class']} "
                                      f"({prediction['magnitude']['confidence']:.3f}), "
                                      f"Az {prediction['azimuth']['class']} "
                                      f"({prediction['azimuth']['confidence']:.3f})")
                            
                        except Exception as e:
                            logger.error(f"[HOUR {hour:02d}] CNN prediction failed: {e}")
                            hour_result['cnn_prediction'] = None
                    
                    results['hourly_results'].append(hour_result)
                    successful_tests += 1
                    
                    logger.info(f"[HOUR {hour:02d}] ✅ Analysis complete")
                    logger.info(f"   Samples: {len(h_hour)}")
                    logger.info(f"   Anomaly Score: {pattern_analysis['anomaly_score']:.3f}")
                    logger.info(f"   Anomaly Level: {pattern_analysis['anomaly_level']}")
                    
                except Exception as e:
                    logger.error(f"[HOUR {hour:02d}] ❌ Failed: {e}")
                    failed_tests += 1
        
        # Generate summary
        results['summary'] = self._generate_summary(results, successful_tests, failed_tests)
        
        return results
    
    def _generate_summary(self, results, successful_tests, failed_tests):
        """Generate summary from test results"""
        hourly_results = results['hourly_results']
        
        if not hourly_results:
            return {
                'status': 'FAILED',
                'message': 'No successful tests',
                'successful_tests': 0,
                'failed_tests': failed_tests
            }
        
        # Aggregate predictions
        mag_predictions = []
        az_predictions = []
        mag_confidences = []
        az_confidences = []
        anomaly_scores = []
        
        for hour_result in hourly_results:
            if 'cnn_prediction' in hour_result and hour_result['cnn_prediction']:
                pred = hour_result['cnn_prediction']
                mag_predictions.append(pred['magnitude']['class'])
                az_predictions.append(pred['azimuth']['class'])
                mag_confidences.append(pred['magnitude']['confidence'])
                az_confidences.append(pred['azimuth']['confidence'])
            
            if 'pattern_analysis' in hour_result:
                anomaly_scores.append(hour_result['pattern_analysis']['anomaly_score'])
        
        summary = {
            'status': 'SUCCESS',
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'coverage': successful_tests / (successful_tests + failed_tests) * 100
        }
        
        # CNN prediction summary
        if mag_predictions:
            from collections import Counter
            
            mag_counter = Counter(mag_predictions)
            az_counter = Counter(az_predictions)
            
            most_common_mag = mag_counter.most_common(1)[0]
            most_common_az = az_counter.most_common(1)[0]
            
            avg_mag_confidence = np.mean(mag_confidences)
            avg_az_confidence = np.mean(az_confidences)
            
            summary['cnn_prediction'] = {
                'magnitude': {
                    'predicted_class': most_common_mag[0],
                    'predicted_value': self.magnitude_to_value[most_common_mag[0]],
                    'frequency': most_common_mag[1],
                    'confidence': avg_mag_confidence,
                    'all_predictions': dict(mag_counter)
                },
                'azimuth': {
                    'predicted_class': most_common_az[0],
                    'predicted_degrees': self.azimuth_to_degrees[most_common_az[0]],
                    'frequency': most_common_az[1],
                    'confidence': avg_az_confidence,
                    'all_predictions': dict(az_counter)
                }
            }
        
        # Anomaly analysis summary
        if anomaly_scores:
            avg_anomaly_score = np.mean(anomaly_scores)
            max_anomaly_score = np.max(anomaly_scores)
            
            summary['anomaly_analysis'] = {
                'average_anomaly_score': avg_anomaly_score,
                'maximum_anomaly_score': max_anomaly_score,
                'anomaly_level': 'High' if avg_anomaly_score > 0.5 else 'Medium' if avg_anomaly_score > 0.25 else 'Low',
                'precursor_likelihood': 'High' if max_anomaly_score > 0.75 else 'Medium' if max_anomaly_score > 0.5 else 'Low'
            }
        
        return summary
    
    def generate_report(self, results, output_file=None):
        """Generate detailed report"""
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("INTEGRATED EARTHQUAKE PRECURSOR TEST REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Test Info
        test_info = results['test_info']
        report_lines.append("TEST INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Date Tested: {test_info['date']}")
        report_lines.append(f"Station: {test_info['station']}")
        report_lines.append(f"Hours Tested: {len(test_info['hours_tested'])}")
        report_lines.append(f"CNN Model Used: {'Yes' if test_info['model_used'] else 'No'}")
        report_lines.append("")
        
        # Summary
        summary = results['summary']
        report_lines.append("SUMMARY RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(f"Status: {summary['status']}")
        report_lines.append(f"Successful Tests: {summary['successful_tests']}")
        report_lines.append(f"Failed Tests: {summary['failed_tests']}")
        report_lines.append(f"Coverage: {summary.get('coverage', 0):.1f}%")
        report_lines.append("")
        
        # CNN Prediction Results
        if 'cnn_prediction' in summary:
            cnn = summary['cnn_prediction']
            report_lines.append("CNN PREDICTION RESULTS")
            report_lines.append("-" * 40)
            
            # Magnitude
            mag = cnn['magnitude']
            report_lines.append(f"MAGNITUDE PREDICTION:")
            report_lines.append(f"  Predicted Class: {mag['predicted_class']}")
            report_lines.append(f"  Predicted Value: {mag['predicted_value']}")
            report_lines.append(f"  Confidence: {mag['confidence']:.3f} ({mag['confidence']*100:.1f}%)")
            report_lines.append(f"  Frequency: {mag['frequency']}/{summary['successful_tests']} hours")
            report_lines.append("")
            
            # Azimuth
            az = cnn['azimuth']
            report_lines.append(f"AZIMUTH PREDICTION:")
            report_lines.append(f"  Predicted Class: {az['predicted_class']}")
            report_lines.append(f"  Predicted Degrees: {az['predicted_degrees']}°")
            report_lines.append(f"  Confidence: {az['confidence']:.3f} ({az['confidence']*100:.1f}%)")
            report_lines.append(f"  Frequency: {az['frequency']}/{summary['successful_tests']} hours")
            report_lines.append("")
        
        # Anomaly Analysis
        if 'anomaly_analysis' in summary:
            anom = summary['anomaly_analysis']
            report_lines.append("ANOMALY ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Average Anomaly Score: {anom['average_anomaly_score']:.3f}")
            report_lines.append(f"Maximum Anomaly Score: {anom['maximum_anomaly_score']:.3f}")
            report_lines.append(f"Anomaly Level: {anom['anomaly_level']}")
            report_lines.append(f"Precursor Likelihood: {anom['precursor_likelihood']}")
            report_lines.append("")
        
        # Final Assessment
        report_lines.append("FINAL ASSESSMENT")
        report_lines.append("-" * 40)
        
        if 'cnn_prediction' in summary and 'anomaly_analysis' in summary:
            mag_class = summary['cnn_prediction']['magnitude']['predicted_class']
            mag_conf = summary['cnn_prediction']['magnitude']['confidence']
            az_class = summary['cnn_prediction']['azimuth']['predicted_class']
            az_conf = summary['cnn_prediction']['azimuth']['confidence']
            precursor_likelihood = summary['anomaly_analysis']['precursor_likelihood']
            
            overall_conf = (mag_conf + az_conf) / 2
            
            report_lines.append(f"EARTHQUAKE POTENTIAL: {'HIGH' if precursor_likelihood == 'High' else 'MEDIUM' if precursor_likelihood == 'Medium' else 'LOW'}")
            report_lines.append(f"PREDICTED MAGNITUDE: {mag_class} (~{summary['cnn_prediction']['magnitude']['predicted_value']})")
            report_lines.append(f"PREDICTED AZIMUTH: {az_class} ({summary['cnn_prediction']['azimuth']['predicted_degrees']}°)")
            report_lines.append(f"OVERALL CONFIDENCE: {overall_conf:.3f} ({overall_conf*100:.1f}%)")
            report_lines.append("")
            
            error_estimate = 1 - overall_conf
            report_lines.append(f"ESTIMATED ERROR: {error_estimate:.3f} ({error_estimate*100:.1f}%)")
            
            if overall_conf > 0.8:
                reliability = "HIGH"
            elif overall_conf > 0.6:
                reliability = "MEDIUM"
            else:
                reliability = "LOW"
            
            report_lines.append(f"PREDICTION RELIABILITY: {reliability}")
            
        else:
            report_lines.append("INSUFFICIENT DATA FOR RELIABLE PREDICTION")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Join report
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_file}")
        
        return report_text


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Integrated earthquake precursor test with CNN predictions'
    )
    parser.add_argument('--date', required=True,
                       help='Test date (YYYY-MM-DD)')
    parser.add_argument('--station', required=True,
                       help='Station code (e.g., SCN, MLB, ALR)')
    parser.add_argument('--hours', default=None,
                       help='Hours to test (e.g., "0,1,2" or "0-23"), default: all hours')
    parser.add_argument('--model', default=None,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', default=None,
                       help='Output report file')
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("INTEGRATED EARTHQUAKE PRECURSOR TEST")
    print("="*80)
    print(f"Date: {args.date}")
    print(f"Station: {args.station}")
    print(f"Hours: {args.hours or 'all'}")
    print(f"Model: {args.model or 'none'}")
    print("="*80)
    
    try:
        # Parse hours
        if args.hours:
            if '-' in args.hours:
                start, end = map(int, args.hours.split('-'))
                hour_range = list(range(start, end + 1))
            else:
                hour_range = [int(h) for h in args.hours.split(',')]
        else:
            hour_range = list(range(24))  # All hours
        
        # Create tester
        tester = IntegratedPrecursorTester(model_path=args.model, device=args.device)
        
        # Test date
        results = tester.test_date(args.date, args.station, hour_range)
        
        # Generate report
        output_file = args.output or f"integrated_precursor_report_{args.station}_{args.date.replace('-', '')}.txt"
        report = tester.generate_report(results, output_file)
        
        # Print summary
        print("\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)
        
        summary = results['summary']
        if summary['status'] == 'SUCCESS':
            print(f"✅ Test successful: {summary['successful_tests']} hours analyzed")
            
            if 'cnn_prediction' in summary:
                cnn = summary['cnn_prediction']
                print(f"\n🎯 CNN PREDICTIONS:")
                print(f"   Magnitude: {cnn['magnitude']['predicted_class']} "
                      f"(~{cnn['magnitude']['predicted_value']}) "
                      f"[Confidence: {cnn['magnitude']['confidence']:.3f}]")
                print(f"   Azimuth: {cnn['azimuth']['predicted_class']} "
                      f"({cnn['azimuth']['predicted_degrees']}°) "
                      f"[Confidence: {cnn['azimuth']['confidence']:.3f}]")
            
            if 'anomaly_analysis' in summary:
                anom = summary['anomaly_analysis']
                print(f"\n📊 ANOMALY ANALYSIS:")
                print(f"   Precursor Likelihood: {anom['precursor_likelihood']}")
                print(f"   Anomaly Level: {anom['anomaly_level']}")
        else:
            print(f"❌ Test failed: {summary.get('message', 'Unknown error')}")
        
        print(f"\n📄 Full report saved to: {output_file}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)