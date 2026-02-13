"""
Model Comparator Module

Compares champion and challenger models using the Champion-Challenger pattern.
Integrates enhanced evaluation metrics with research-based decision criteria.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

from .utils import (
    load_config, load_registry, save_registry,
    calculate_composite_score, log_pipeline_event
)
from .evaluator import ModelEvaluator
from .enhanced_comparator import EnhancedModelComparator

logger = logging.getLogger("autoupdate_pipeline.comparator")


class ModelComparator:
    """
    Compares champion and challenger models.
    
    Decision criteria (research-based):
    1. Composite score comparison (weighted metrics)
    2. Statistical significance testing (McNemar/Bootstrap)
    3. No regression in critical metrics (No-Harm Principle)
    
    References:
    - Chicco & Jurman (2020) - MCC: DOI 10.1186/s12864-019-6413-7
    - McNemar (1947): DOI 10.1007/BF02295996
    - Dietterich (1998): DOI 10.1162/089976698300017197
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize comparator."""
        self.config = config or load_config()
        self.comparison_config = self.config.get('comparison', {})
        self.eval_config = self.config.get('evaluation', {})
        
        self.base_path = Path(__file__).parent.parent
        self.evaluator = ModelEvaluator(self.config)
        
        # Initialize enhanced comparator with research-based metrics
        self.enhanced_comparator = EnhancedModelComparator(self.config)
        
        # Use enhanced weights if available
        self.weights = self.eval_config.get('weights', self.comparison_config.get('weights', {
            'magnitude_accuracy': 0.35,
            'azimuth_accuracy': 0.15,
            'macro_f1': 0.20,
            'mcc': 0.15,
            'loeo_stability': 0.10,
            'false_positive_rate': 0.05
        }))
        
        logger.info("ModelComparator initialized with enhanced evaluation metrics")
    
    def compare(self, champion_path: str = None, challenger_path: str = None) -> Dict[str, Any]:
        """
        Compare champion and challenger models using enhanced evaluation.
        
        Args:
            champion_path: Path to champion model directory
            challenger_path: Path to challenger model directory
            
        Returns:
            Comparison results with decision
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL COMPARISON (Enhanced Evaluation)")
        logger.info("=" * 60)
        
        # Default paths
        if champion_path is None:
            champion_path = self.base_path / self.config['paths']['champion_model']
        else:
            champion_path = Path(champion_path)
        
        if challenger_path is None:
            challenger_path = self.base_path / self.config['paths']['challenger_model']
        else:
            challenger_path = Path(challenger_path)
        
        # Evaluate both models
        logger.info("Evaluating Champion model...")
        champion_results = self.evaluator.evaluate_model(
            str(champion_path / 'best_model.pth'),
            str(champion_path / 'class_mappings.json')
        )
        
        logger.info("Evaluating Challenger model...")
        challenger_results = self.evaluator.evaluate_model(
            str(challenger_path / 'best_model.pth'),
            str(challenger_path / 'class_mappings.json')
        )
        
        # Prepare metrics for enhanced comparison
        champion_metrics = self._prepare_metrics(champion_results)
        challenger_metrics = self._prepare_metrics(challenger_results)
        
        # Use enhanced comparator for decision
        enhanced_result = self.enhanced_comparator.compare(
            champion_metrics, challenger_metrics
        )
        
        # Convert enhanced result to legacy format
        decision = {
            "winner": "challenger" if enhanced_result.decision['promote'] else "champion",
            "promote_challenger": enhanced_result.decision['promote'],
            "reason": enhanced_result.decision['reason'],
            "confidence": enhanced_result.decision['confidence'],
            "details": enhanced_result.decision.get('details', {})
        }
        
        # Compare metrics (legacy format)
        metrics_comparison = self.evaluator.compare_models(champion_results, challenger_results)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "champion": {
                "path": str(champion_path),
                "results": champion_results,
                "composite_score": enhanced_result.champion_score
            },
            "challenger": {
                "path": str(challenger_path),
                "results": challenger_results,
                "composite_score": enhanced_result.challenger_score
            },
            "comparison": metrics_comparison,
            "enhanced_comparison": {
                "score_difference": enhanced_result.score_difference,
                "statistical_test": enhanced_result.statistical_test,
                "regression_check": enhanced_result.regression_check
            },
            "decision": decision
        }
        
        # Log event
        log_pipeline_event("comparison_complete", {
            "champion_score": enhanced_result.champion_score,
            "challenger_score": enhanced_result.challenger_score,
            "score_difference": enhanced_result.score_difference,
            "decision": decision["winner"],
            "reason": decision["reason"],
            "confidence": decision["confidence"]
        })
        
        # Print summary
        self._print_summary(results)
        
        results["success"] = True
        return results
    
    def _prepare_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, float]:
        """Prepare metrics dictionary for enhanced comparator."""
        return {
            'magnitude_accuracy': eval_results.get('magnitude_accuracy', 0),
            'azimuth_accuracy': eval_results.get('azimuth_accuracy', 0),
            'macro_f1': eval_results.get('magnitude_f1', 0),  # Use magnitude F1 as proxy
            'mcc': eval_results.get('magnitude_mcc', 0) / 100.0,  # Convert to -1 to 1 scale
            'loeo_std': 2.5,  # Default value, would need LOEO validation
            'false_positive_rate': 3.0,  # Default value, would need calculation
            'large_recall': eval_results.get('magnitude', {}).get('per_class', {}).get('Large', {}).get('recall', 95) * 100
        }
    
    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite score from evaluation results."""
        metrics = {
            'magnitude_accuracy': results.get('magnitude_accuracy', 0),
            'azimuth_accuracy': results.get('azimuth_accuracy', 0),
            'loeo_validation': results.get('magnitude_accuracy', 0),  # Use mag acc as proxy
            'false_positive_rate': 0  # Would need to calculate from confusion matrix
        }
        
        return calculate_composite_score(metrics, self.weights)
    
    def _make_decision(self, champion_results: Dict, challenger_results: Dict,
                      champion_score: float, challenger_score: float,
                      comparison: Dict) -> Dict[str, Any]:
        """
        Make decision on whether to promote challenger.
        
        Decision rules:
        1. Challenger must have higher composite score
        2. No significant regression in any critical metric
        3. Meet minimum improvement threshold
        """
        decision = {
            "winner": None,
            "promote_challenger": False,
            "reason": "",
            "details": {}
        }
        
        min_improvement = self.comparison_config.get('min_improvement', 0.0)
        strict_mode = self.comparison_config.get('strict_mode', False)
        
        score_diff = challenger_score - champion_score
        
        # Check composite score
        if score_diff < min_improvement:
            decision["winner"] = "champion"
            decision["reason"] = f"Challenger score ({challenger_score:.4f}) not better than champion ({champion_score:.4f})"
            return decision
        
        # Check for regressions in critical metrics
        critical_metrics = ['magnitude_accuracy']
        regressions = []
        
        for metric in critical_metrics:
            champ_val = champion_results.get(metric, 0)
            chall_val = challenger_results.get(metric, 0)
            
            if chall_val < champ_val - 1.0:  # Allow 1% tolerance
                regressions.append(f"{metric}: {chall_val:.2f}% < {champ_val:.2f}%")
        
        if regressions:
            decision["winner"] = "champion"
            decision["reason"] = f"Challenger has regressions: {', '.join(regressions)}"
            decision["details"]["regressions"] = regressions
            return decision
        
        # Strict mode: require improvement in ALL metrics
        if strict_mode:
            improvements = comparison.get('improvements', {})
            if not all(improvements.values()):
                decision["winner"] = "champion"
                decision["reason"] = "Strict mode: Not all metrics improved"
                return decision
        
        # Challenger wins!
        decision["winner"] = "challenger"
        decision["promote_challenger"] = True
        decision["reason"] = f"Challenger score ({challenger_score:.4f}) > Champion ({champion_score:.4f})"
        decision["details"] = {
            "score_improvement": score_diff,
            "magnitude_improvement": challenger_results['magnitude_accuracy'] - champion_results['magnitude_accuracy'],
            "azimuth_improvement": challenger_results['azimuth_accuracy'] - champion_results['azimuth_accuracy']
        }
        
        return decision
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comparison summary."""
        decision = results['decision']
        enhanced = results.get('enhanced_comparison', {})
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("COMPARISON SUMMARY (Enhanced Evaluation)")
        logger.info("=" * 60)
        logger.info("")
        logger.info(f"Champion Score:   {results['champion']['composite_score']:.4f}")
        logger.info(f"Challenger Score: {results['challenger']['composite_score']:.4f}")
        logger.info(f"Score Difference: {enhanced.get('score_difference', 0):+.4f}")
        logger.info("")
        logger.info(f"Champion Magnitude:   {results['champion']['results']['magnitude_accuracy']:.2f}%")
        logger.info(f"Challenger Magnitude: {results['challenger']['results']['magnitude_accuracy']:.2f}%")
        logger.info("")
        logger.info(f"Champion Azimuth:   {results['champion']['results']['azimuth_accuracy']:.2f}%")
        logger.info(f"Challenger Azimuth: {results['challenger']['results']['azimuth_accuracy']:.2f}%")
        logger.info("")
        
        # Statistical test info
        stat_test = enhanced.get('statistical_test', {})
        if stat_test:
            logger.info(f"Statistical Test: {stat_test.get('method', 'N/A')}")
            logger.info(f"P-value: {stat_test.get('p_value', 'N/A')}")
            logger.info(f"Significant: {'Yes' if stat_test.get('significant') else 'No'}")
            logger.info("")
        
        # Regression check
        reg_check = enhanced.get('regression_check', {})
        if reg_check:
            logger.info(f"Regression Check: {'PASSED' if reg_check.get('passed') else 'FAILED'}")
            if not reg_check.get('passed'):
                for reg in reg_check.get('regressions', []):
                    logger.warning(f"  - {reg}")
            logger.info("")
        
        logger.info("-" * 60)
        
        if decision['promote_challenger']:
            logger.info("DECISION: CHALLENGER WINS! 🎉")
            logger.info(f"   Reason: {decision['reason']}")
            logger.info(f"   Confidence: {decision.get('confidence', 0):.1%}")
            logger.info("   Action: Promote challenger to production")
        else:
            logger.info("DECISION: CHAMPION RETAINS 🛡️")
            logger.info(f"   Reason: {decision['reason']}")
            logger.info("   Action: Keep current production model")
        
        logger.info("=" * 60)
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed comparison report."""
        lines = [
            "=" * 70,
            "MODEL COMPARISON REPORT",
            f"Generated: {results['timestamp']}",
            "=" * 70,
            "",
            "COMPOSITE SCORES:",
            "-" * 50,
            f"  Champion:   {results['champion']['composite_score']:.4f}",
            f"  Challenger: {results['challenger']['composite_score']:.4f}",
            f"  Difference: {results['challenger']['composite_score'] - results['champion']['composite_score']:+.4f}",
            "",
            "METRIC COMPARISON:",
            "-" * 50,
        ]
        
        comparison = results['comparison']['metrics_comparison']
        for metric, data in comparison.items():
            status = "✅" if data['improved'] else "❌"
            lines.append(f"  {metric}:")
            lines.append(f"    Champion:   {data['champion']:.2f}%")
            lines.append(f"    Challenger: {data['challenger']:.2f}%")
            lines.append(f"    Change:     {data['difference']:+.2f}% {status}")
            lines.append("")
        
        lines.extend([
            "DECISION:",
            "-" * 50,
            f"  Winner: {results['decision']['winner'].upper()}",
            f"  Promote Challenger: {'YES' if results['decision']['promote_challenger'] else 'NO'}",
            f"  Reason: {results['decision']['reason']}",
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)
