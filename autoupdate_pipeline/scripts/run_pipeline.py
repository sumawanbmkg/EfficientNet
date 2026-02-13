#!/usr/bin/env python
"""
Main Pipeline Runner

Orchestrates the complete auto-update pipeline:
1. Check trigger conditions
2. Train new challenger model
3. Evaluate and compare with champion
4. Deploy if challenger wins
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_registry, save_registry, log_pipeline_event
from src.data_validator import DataValidator
from src.data_ingestion import DataIngestion
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.model_comparator import ModelComparator
from src.deployer import ModelDeployer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'pipeline.log')
    ]
)
logger = logging.getLogger("pipeline_runner")


class PipelineRunner:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = None, quick_test: bool = False):
        """
        Initialize pipeline components.
        
        Args:
            config_path: Path to config file
            quick_test: Enable quick test mode for faster testing
        """
        self.config = load_config(config_path)
        self.trigger_config = self.config.get('triggers', {})
        self.quick_test = quick_test
        
        # Initialize components
        self.validator = DataValidator(self.config)
        self.ingestion = DataIngestion(self.config)
        self.trainer = ModelTrainer(self.config, quick_test=quick_test)
        self.evaluator = ModelEvaluator(self.config)
        self.comparator = ModelComparator(self.config)
        self.deployer = ModelDeployer(self.config)
        
        if quick_test:
            logger.info("Pipeline initialized in QUICK TEST MODE")
        else:
            logger.info("Pipeline initialized")
    
    def check_trigger_conditions(self) -> dict:
        """
        Check if pipeline should be triggered.
        
        Returns:
            dict with trigger status and reason
        """
        registry = load_registry()
        
        # Check 1: Minimum new events
        validated_count = len(registry.get('validated_events', []))
        min_events = self.trigger_config.get('min_new_events', 20)
        
        if validated_count >= min_events:
            return {
                "should_trigger": True,
                "reason": f"Minimum events reached ({validated_count}/{min_events})"
            }
        
        # Check 2: Maximum days since last training
        champion = registry.get('champion', {})
        last_deployed = champion.get('deployed_at')
        
        if last_deployed:
            last_date = datetime.fromisoformat(last_deployed)
            days_since = (datetime.now() - last_date).days
            max_days = self.trigger_config.get('max_days_between_training', 90)
            
            if days_since >= max_days:
                return {
                    "should_trigger": True,
                    "reason": f"Maximum days reached ({days_since}/{max_days} days)"
                }
        
        # Check 3: Performance degradation (if monitoring enabled)
        # TODO: Implement performance monitoring
        
        return {
            "should_trigger": False,
            "reason": f"Conditions not met. Events: {validated_count}/{min_events}",
            "validated_events": validated_count,
            "min_events_required": min_events
        }
    
    def run(self, force: bool = False, auto_deploy: bool = False) -> dict:
        """
        Run the complete pipeline.
        
        Args:
            force: Force run even if trigger conditions not met
            auto_deploy: Auto-deploy if challenger wins
            
        Returns:
            Pipeline execution results
        """
        logger.info("=" * 70)
        logger.info("EARTHQUAKE MODEL AUTO-UPDATE PIPELINE")
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info("=" * 70)
        
        results = {
            "success": False,
            "started_at": datetime.now().isoformat(),
            "stages": {}
        }
        
        log_pipeline_event("pipeline_started", {"force": force, "auto_deploy": auto_deploy})
        
        try:
            # Stage 1: Check trigger conditions
            logger.info("\n[STAGE 1] Checking trigger conditions...")
            trigger_status = self.check_trigger_conditions()
            results["stages"]["trigger_check"] = trigger_status
            
            if not trigger_status["should_trigger"] and not force:
                results["message"] = f"Pipeline not triggered: {trigger_status['reason']}"
                logger.info(results["message"])
                return results
            
            logger.info(f"Trigger: {trigger_status['reason']}")
            
            # Stage 2: Train new model
            logger.info("\n[STAGE 2] Training challenger model...")
            training_results = self.trainer.train_model(include_new_events=True)
            results["stages"]["training"] = training_results
            
            if not training_results.get("success"):
                results["message"] = "Training failed"
                logger.error(results["message"])
                return results
            
            logger.info(f"Training complete. Model ID: {training_results['model_id']}")
            
            # Stage 3: Evaluate challenger
            logger.info("\n[STAGE 3] Evaluating challenger model...")
            eval_results = self.evaluator.evaluate_challenger()
            results["stages"]["evaluation"] = eval_results
            
            if not eval_results.get("success"):
                results["message"] = "Evaluation failed"
                logger.error(results["message"])
                return results
            
            logger.info(f"Challenger Mag Acc: {eval_results['results']['magnitude_accuracy']:.2f}%")
            logger.info(f"Challenger Azi Acc: {eval_results['results']['azimuth_accuracy']:.2f}%")
            
            # Stage 4: Compare with champion
            logger.info("\n[STAGE 4] Comparing with champion...")
            comparison_results = self.comparator.compare()
            results["stages"]["comparison"] = comparison_results
            
            if not comparison_results.get("success"):
                results["message"] = "Comparison failed"
                logger.error(results["message"])
                return results
            
            decision = comparison_results["decision"]
            logger.info(f"Decision: {'PROMOTE CHALLENGER' if decision['promote_challenger'] else 'KEEP CHAMPION'}")
            logger.info(f"Reason: {decision['reason']}")
            
            # Stage 5: Deploy if challenger wins
            if decision["promote_challenger"]:
                logger.info("\n[STAGE 5] Deploying challenger...")
                
                deploy_force = auto_deploy or self.config.get('deployment', {}).get('auto_deploy', False)
                deploy_results = self.deployer.deploy_challenger(comparison_results, force=deploy_force)
                results["stages"]["deployment"] = deploy_results
                
                if deploy_results["success"]:
                    results["success"] = True
                    results["message"] = f"Pipeline complete! New model deployed: {deploy_results['new_champion']['version']}"
                else:
                    results["message"] = f"Deployment pending: {deploy_results['message']}"
            else:
                results["success"] = True
                results["message"] = "Pipeline complete. Champion retained (challenger did not improve)."
            
            results["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            results["message"] = f"Pipeline error: {str(e)}"
            logger.error(results["message"], exc_info=True)
            log_pipeline_event("pipeline_error", {"error": str(e)})
        
        logger.info("\n" + "=" * 70)
        logger.info(f"PIPELINE RESULT: {results['message']}")
        logger.info("=" * 70)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run earthquake model auto-update pipeline")
    parser.add_argument("--force", action="store_true", help="Force run even if conditions not met")
    parser.add_argument("--auto-deploy", action="store_true", help="Auto-deploy if challenger wins")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--check-only", action="store_true", help="Only check trigger conditions")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Quick test mode: reduced dataset (200 samples) and epochs (3)")
    
    args = parser.parse_args()
    
    runner = PipelineRunner(args.config, quick_test=args.quick_test)
    
    if args.check_only:
        status = runner.check_trigger_conditions()
        print(f"\nTrigger Status: {'READY' if status['should_trigger'] else 'NOT READY'}")
        print(f"Reason: {status['reason']}")
        return
    
    results = runner.run(force=args.force, auto_deploy=args.auto_deploy)
    
    # Exit code based on success
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
