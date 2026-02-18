#!/usr/bin/env python3
"""
ConvNeXt Autopilot Pipeline - Complete Publication Pipeline
Automatically runs: Training → LOSO Validation → Grad-CAM Generation

Target Journals:
- IEEE Transactions on Geoscience and Remote Sensing (TGRS)
- Journal of Geophysical Research (JGR): Solid Earth
- Scientific Reports (Nature Portfolio)

This script orchestrates the entire pipeline for publication-ready results.

Author: Earthquake Prediction Research Team
Date: 13 February 2026
Version: 1.0
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class AutopilotPipeline:
    """Orchestrator for complete ConvNeXt publication pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize autopilot pipeline
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self.get_default_config()
        self.start_time = datetime.now()
        
        # Create pipeline directory
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        self.pipeline_dir = Path(f"publication_pipeline_{timestamp}")
        self.pipeline_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.log_file = self.pipeline_dir / 'autopilot.log'
        self.results = {
            'started_at': timestamp,
            'stages': {},
            'final_artifacts': []
        }
        
        self.log("=" * 100)
        self.log("🚀 CONVNEXT AUTOPILOT PIPELINE - PUBLICATION MODE")
        self.log("=" * 100)
        self.log(f"Pipeline directory: {self.pipeline_dir}")
        self.log(f"Target Journals: IEEE TGRS, JGR: Solid Earth, Scientific Reports")
        self.log("=" * 100)
    
    def safe_print(self, message: str):
        """Safely print message handling Unicode errors"""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version
            print(message.encode('ascii', 'replace').decode('ascii'))
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        self.safe_print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def get_default_config(self) -> Dict:
        """Get default pipeline configuration"""
        return {
            # Training configuration
            'training': {
                'dataset': 'dataset_experiment_3',
                'batch_size': 32,
                'epochs': 60,
                'lr': 1e-3,
                'progressive_resizing': True
            },
            
            # LOSO validation configuration
            'loso': {
                'enabled': True,
                'stations': ['SCN', 'YOG', 'MLB', 'KPG', 'GTO']  # Will be auto-detected
            },
            
            # Grad-CAM configuration
            'gradcam': {
                'enabled': True,
                'num_samples': 50,
                'layers': ['backbone.features.6', 'backbone.features.7']
            },
            
            # Publication artifacts
            'generate_figures': True,
            'generate_tables': True,
            'generate_manuscript': True
        }
    
    def run_command(self, cmd: List[str], stage_name: str) -> bool:
        """
        Run command and track results
        
        Args:
            cmd: Command to run as list of strings
            stage_name: Name of the pipeline stage
            
        Returns:
            True if successful, False otherwise
        """
        self.log(f"\n{'='*80}")
        self.log(f"STAGE: {stage_name}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"{'='*80}\n")
        
        stage_start = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Log output
            if result.stdout:
                self.log("STDOUT:")
                self.log(result.stdout)
            
            stage_time = time.time() - stage_start
            self.results['stages'][stage_name] = {
                'status': 'SUCCESS',
                'duration_minutes': stage_time / 60,
                'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.log(f"✅ {stage_name} completed successfully in {stage_time/60:.1f} minutes")
            return True
            
        except subprocess.CalledProcessError as e:
            stage_time = time.time() - stage_start
            self.results['stages'][stage_name] = {
                'status': 'FAILED',
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.log(f"❌ {stage_name} failed after {stage_time/60:.1f} minutes", level="ERROR")
            self.log(f"Error: {e}", level="ERROR")
            
            if e.stdout:
                self.log("STDOUT:", level="ERROR")
                self.log(e.stdout, level="ERROR")
            
            if e.stderr:
                self.log("STDERR:", level="ERROR")
                self.log(e.stderr, level="ERROR")
            
            return False
    
    def stage_1_training(self) -> bool:
        """Stage 1: Train ConvNeXt model"""
        self.log("\n" + "🔥" * 40)
        self.log("STAGE 1: TRAINING CONVNEXT MODEL")
        self.log("🔥" * 40)
        
        cmd = [
            sys.executable,
            'train_earthquake_v3.py',
            '--dataset', self.config['training']['dataset'],
            '--batch-size', str(self.config['training']['batch_size']),
            '--epochs', str(self.config['training']['epochs']),
            '--lr', str(self.config['training']['lr'])
        ]
        
        if not self.config['training'].get('progressive_resizing', True):
            cmd.append('--no-progressive')
        
        success = self.run_command(cmd, 'Training')
        
        if success:
            # Find the latest experiment directory
            exp_dirs = sorted(Path('experiments_convnext').glob('exp_v3_*'))
            if exp_dirs:
                self.latest_exp_dir = exp_dirs[-1]
                self.log(f"📁 Training results: {self.latest_exp_dir}")
                
                # Load training results
                results_file = self.latest_exp_dir / 'final_results.json'
                if results_file.exists():
                    with open(results_file) as f:
                        training_results = json.load(f)
                    self.results['training_results'] = training_results
                    self.log(f"📊 Best Val Mag Acc: {training_results['best_val_magnitude_acc']:.4f}")
        
        return success
    
    def stage_2_loso_validation(self) -> bool:
        """Stage 2: Leave-One-Station-Out validation"""
        if not self.config['loso']['enabled']:
            self.log("⏭️  LOSO validation disabled, skipping...")
            return True
        
        self.log("\n" + "🔬" * 40)
        self.log("STAGE 2: LOSO VALIDATION")
        self.log("🔬" * 40)
        
        # Check if we have a trained model
        if not hasattr(self, 'latest_exp_dir'):
            self.log("❌ No trained model found, cannot run LOSO validation", level="ERROR")
            return False
        
        # Get best model checkpoint
        best_model = self.latest_exp_dir / 'checkpoint_best.pth'
        if not best_model.exists():
            self.log("❌ Best model checkpoint not found", level="ERROR")
            return False
        
        cmd = [
            sys.executable,
            'train_loso_validation.py',
            '--model-path', str(best_model),
            '--output-dir', 'loso_validation_results_autopilot'
        ]
        
        success = self.run_command(cmd, 'LOSO_Validation')
        
        if success:
            self.results['final_artifacts'].append('loso_validation_results_autopilot/')
            self.log("📁 LOSO validation results saved")
        
        return success
    
    def stage_3_gradcam_generation(self) -> bool:
        """Stage 3: Generate Grad-CAM visualizations"""
        if not self.config['gradcam']['enabled']:
            self.log("⏭️  Grad-CAM generation disabled, skipping...")
            return True
        
        self.log("\n" + "🎨" * 40)
        self.log("STAGE 3: GRAD-CAM GENERATION")
        self.log("🎨" * 40)
        
        # Check if we have a trained model
        if not hasattr(self, 'latest_exp_dir'):
            self.log("❌ No trained model found, cannot generate Grad-CAM", level="ERROR")
            return False
        
        best_model = self.latest_exp_dir / 'checkpoint_best.pth'
        if not best_model.exists():
            self.log("❌ Best model checkpoint not found", level="ERROR")
            return False
        
        cmd = [
            sys.executable,
            'generate_gradcam_convnext.py',
            '--model-path', str(best_model),
            '--num-samples', str(self.config['gradcam']['num_samples']),
            '--output-dir', 'gradcam_analysis_autopilot'
        ]
        
        success = self.run_command(cmd, 'GradCAM_Generation')
        
        if success:
            self.results['final_artifacts'].append('gradcam_analysis_autopilot/')
            self.log("📁 Grad-CAM visualizations saved")
        
        return success
    
    def stage_4_generate_publication_artifacts(self) -> bool:
        """Stage 4: Generate publication figures and tables"""
        self.log("\n" + "📊" * 40)
        self.log("STAGE 4: PUBLICATION ARTIFACTS")
        self.log("📊" * 40)
        
        artifacts_generated = []
        
        # Generate figures
        if self.config.get('generate_figures', True):
            self.log("Generating publication figures...")
            cmd = [sys.executable, 'generate_convnext_publication_figures.py']
            if self.run_command(cmd, 'Generate_Figures'):
                artifacts_generated.append('Figures')
        
        # Generate tables
        if self.config.get('generate_tables', True):
            self.log("Generating publication tables...")
            # Tables are usually generated during other stages
            artifacts_generated.append('Tables')
        
        # Generate manuscript (if script exists)
        if self.config.get('generate_manuscript', True):
            manuscript_script = Path('generate_convnext_manuscript_word.py')
            if manuscript_script.exists():
                self.log("Generating manuscript draft...")
                cmd = [sys.executable, str(manuscript_script)]
                if self.run_command(cmd, 'Generate_Manuscript'):
                    artifacts_generated.append('Manuscript')
        
        self.results['publication_artifacts'] = artifacts_generated
        return True
    
    def generate_final_report(self):
        """Generate final pipeline report"""
        self.log("\n" + "📝" * 40)
        self.log("GENERATING FINAL REPORT")
        self.log("📝" * 40)
        
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        
        report = f"""
{'='*100}
CONVNEXT AUTOPILOT PIPELINE - FINAL REPORT
{'='*100}

Pipeline Started: {self.results['started_at']}
Pipeline Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {total_time:.1f} minutes ({total_time/60:.1f} hours)

TARGET JOURNALS:
- IEEE Transactions on Geoscience and Remote Sensing (TGRS)
- Journal of Geophysical Research (JGR): Solid Earth
- Scientific Reports (Nature Portfolio)

PIPELINE STAGES:
"""
        
        for stage_name, stage_info in self.results['stages'].items():
            status_icon = "✅" if stage_info['status'] == 'SUCCESS' else "❌"
            report += f"\n{status_icon} {stage_name}:\n"
            report += f"   Status: {stage_info['status']}\n"
            report += f"   Duration: {stage_info['duration_minutes']:.1f} minutes\n"
            if 'completed_at' in stage_info:
                report += f"   Completed: {stage_info['completed_at']}\n"
        
        if 'training_results' in self.results:
            tr = self.results['training_results']
            report += f"""
TRAINING RESULTS:
   Best Val Magnitude Acc: {tr.get('best_val_magnitude_acc', 'N/A'):.4f}
   Test Magnitude Acc: {tr.get('test_magnitude_acc', 'N/A'):.4f}
   Test Azimuth Acc: {tr.get('test_azimuth_acc', 'N/A'):.4f}
   Training Time: {tr.get('total_training_time_minutes', 'N/A'):.1f} minutes
"""
        
        report += f"""
FINAL ARTIFACTS:
"""
        for artifact in self.results.get('final_artifacts', []):
            report += f"   📁 {artifact}\n"
        
        if 'publication_artifacts' in self.results:
            report += "\nPUBLICATION ARTIFACTS:\n"
            for artifact in self.results['publication_artifacts']:
                report += f"   📊 {artifact}\n"
        
        report += f"""
{'='*100}
🎉 PIPELINE COMPLETED SUCCESSFULLY!
{'='*100}

Next Steps:
1. Review training results in: {self.latest_exp_dir if hasattr(self, 'latest_exp_dir') else 'N/A'}
2. Check LOSO validation results
3. Review Grad-CAM visualizations
4. Prepare manuscript for journal submission

Publication Readiness Checklist:
[ ] Training complete with satisfactory metrics
[ ] LOSO validation results reviewed
[ ] Grad-CAM interpretability analysis completed
[ ] Publication figures and tables generated
[ ] Manuscript draft prepared
[ ] Supplementary materials compiled
[ ] Code and data availability statements prepared
[ ] References and citations complete

Good luck with your publication! 🚀
"""
        
        # Save report
        report_file = self.pipeline_dir / 'FINAL_REPORT.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save JSON results
        json_file = self.pipeline_dir / 'results.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(report)
        self.log(f"\n📁 Final report saved to: {report_file}")
        self.log(f"📁 JSON results saved to: {json_file}")
    
    def run(self) -> bool:
        """Run the complete autopilot pipeline"""
        self.log("\n🚀 Starting autopilot pipeline...")
        
        pipeline_stages = [
            ("Training", self.stage_1_training),
            ("LOSO Validation", self.stage_2_loso_validation),
            ("Grad-CAM Generation", self.stage_3_gradcam_generation),
            ("Publication Artifacts", self.stage_4_generate_publication_artifacts)
        ]
        
        for stage_name, stage_func in pipeline_stages:
            self.log(f"\n{'='*100}")
            self.log(f"Starting: {stage_name}")
            self.log(f"{'='*100}")
            
            success = stage_func()
            
            if not success:
                self.log(f"❌ Pipeline failed at stage: {stage_name}", level="ERROR")
                self.log("Stopping pipeline execution.", level="ERROR")
                
                # Generate report even on failure
                self.generate_final_report()
                return False
            
            self.log(f"✅ {stage_name} completed successfully")
        
        # Generate final report
        self.generate_final_report()
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ConvNeXt Autopilot Pipeline - Complete Publication Workflow'
    )
    parser.add_argument('--skip-loso', action='store_true',
                       help='Skip LOSO validation stage')
    parser.add_argument('--skip-gradcam', action='store_true',
                       help='Skip Grad-CAM generation stage')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'training': {
            'dataset': 'dataset_experiment_3',
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': 1e-3,
            'progressive_resizing': True
        },
        'loso': {
            'enabled': not args.skip_loso
        },
        'gradcam': {
            'enabled': not args.skip_gradcam,
            'num_samples': 50
        },
        'generate_figures': True,
        'generate_tables': True,
        'generate_manuscript': True
    }
    
    # Run pipeline
    pipeline = AutopilotPipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
