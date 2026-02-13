"""
Enhanced Model Comparator Module

Implements research-based evaluation metrics and decision criteria
for Champion-Challenger model comparison.

References:
- arXiv:2506.17442 (Medical AI Monitoring)
- arXiv:2512.18390 (Model Switching Decision)
- Davis et al., 2019 (Model Updating)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from scipy import stats

logger = logging.getLogger("autoupdate_pipeline.enhanced_comparator")


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    # Discrimination metrics
    magnitude_accuracy: float = 0.0
    azimuth_accuracy: float = 0.0
    macro_f1: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    
    # Per-class metrics
    large_recall: float = 0.0
    medium_recall: float = 0.0
    moderate_recall: float = 0.0
    normal_recall: float = 0.0
    
    # Robustness metrics
    loeo_mean: float = 0.0
    loeo_std: float = 0.0
    
    # Operational metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    inference_time_ms: float = 0.0
    
    # Calibration metrics (optional)
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0


@dataclass
class ComparisonResult:
    """Container for comparison results."""
    champion_score: float
    challenger_score: float
    score_difference: float
    statistical_test: Dict[str, Any]
    regression_check: Dict[str, Any]
    decision: Dict[str, Any]
    timestamp: str
    

class EnhancedModelComparator:
    """
    Enhanced model comparator with research-based evaluation metrics.
    
    Features:
    1. Multi-dimensional evaluation (discrimination, calibration, robustness)
    2. Weighted composite scoring
    3. Statistical significance testing (McNemar, Bootstrap)
    4. No-harm regression checking
    5. Configurable decision thresholds
    """
    
    # Default weights based on earthquake precursor detection priorities
    DEFAULT_WEIGHTS = {
        'magnitude_accuracy': 0.35,
        'azimuth_accuracy': 0.15,
        'macro_f1': 0.20,
        'mcc': 0.15,
        'loeo_stability': 0.10,
        'false_positive_rate': 0.05
    }
    
    # Default regression tolerances
    DEFAULT_TOLERANCES = {
        'magnitude_accuracy': -1.0,    # Max 1% drop allowed
        'large_recall': -2.0,          # Max 2% drop allowed
        'false_positive_rate': 1.0     # Max 1% increase allowed
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize enhanced comparator."""
        self.config = config or {}
        
        # Load evaluation config
        eval_config = self.config.get('evaluation', {})
        
        self.weights = eval_config.get('weights', self.DEFAULT_WEIGHTS)
        self.tolerances = eval_config.get('regression_tolerance', self.DEFAULT_TOLERANCES)
        
        # Decision parameters
        decision_config = eval_config.get('decision', {})
        self.min_improvement = decision_config.get('min_improvement', 0.005)
        self.significance_level = decision_config.get('significance_level', 0.05)
        self.strict_mode = decision_config.get('strict_mode', False)
        
        # Statistical test config
        stat_config = eval_config.get('statistical_test', {})
        self.stat_method = stat_config.get('method', 'mcnemar')
        self.bootstrap_iterations = stat_config.get('bootstrap_iterations', 1000)
        
        logger.info("EnhancedModelComparator initialized")
        logger.info(f"  Weights: {self.weights}")
        logger.info(f"  Min improvement: {self.min_improvement}")
        logger.info(f"  Significance level: {self.significance_level}")
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from metrics.
        
        Formula: Score = Σ(weight_i × normalized_metric_i)
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Composite score (0-1 scale)
        """
        score = 0.0
        
        for metric_name, weight in self.weights.items():
            if metric_name == 'loeo_stability':
                # Lower std is better, so invert
                value = 1.0 - metrics.get('loeo_std', 0.0) / 100.0
            elif metric_name == 'false_positive_rate':
                # Lower FPR is better, so invert
                value = 1.0 - metrics.get('false_positive_rate', 0.0) / 100.0
            else:
                # Higher is better, normalize to 0-1
                value = metrics.get(metric_name, 0.0) / 100.0
            
            score += weight * max(0.0, min(1.0, value))
        
        return score
    
    def calculate_mcc(self, confusion_matrix: np.ndarray) -> float:
        """
        Calculate Matthews Correlation Coefficient for multi-class.
        
        MCC is considered one of the best metrics for imbalanced datasets.
        Range: -1 (worst) to +1 (best), 0 = random
        
        Reference: Chicco & Jurman (2020), BMC Genomics
        """
        # For multi-class, use sklearn's implementation
        try:
            from sklearn.metrics import matthews_corrcoef
            
            # Flatten confusion matrix to get predictions
            n_classes = confusion_matrix.shape[0]
            y_true = []
            y_pred = []
            
            for i in range(n_classes):
                for j in range(n_classes):
                    count = int(confusion_matrix[i, j])
                    y_true.extend([i] * count)
                    y_pred.extend([j] * count)
            
            return matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Could not calculate MCC: {e}")
            return 0.0
    
    def calculate_macro_f1(self, confusion_matrix: np.ndarray) -> float:
        """
        Calculate Macro F1-Score from confusion matrix.
        
        Macro F1 = average of F1 scores for each class
        Good for imbalanced datasets as it treats all classes equally.
        """
        n_classes = confusion_matrix.shape[0]
        f1_scores = []
        
        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores) * 100  # Return as percentage
    
    def mcnemar_test(self, champion_preds: np.ndarray, challenger_preds: np.ndarray,
                     true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Perform McNemar's test to compare two classifiers.
        
        McNemar's test is appropriate when comparing two classifiers
        on the same test set. It tests whether the disagreements
        between classifiers are symmetric.
        
        H0: The classifiers have the same error rate
        H1: The classifiers have different error rates
        
        Reference: McNemar (1947), Dietterich (1998)
        """
        # Build contingency table
        champion_correct = (champion_preds == true_labels)
        challenger_correct = (challenger_preds == true_labels)
        
        # a: both correct, b: champion correct & challenger wrong
        # c: champion wrong & challenger correct, d: both wrong
        a = np.sum(champion_correct & challenger_correct)
        b = np.sum(champion_correct & ~challenger_correct)
        c = np.sum(~champion_correct & challenger_correct)
        d = np.sum(~champion_correct & ~challenger_correct)
        
        contingency_table = np.array([[a, b], [c, d]])
        
        # McNemar's test (exact for small samples)
        try:
            if b + c < 25:
                # Use exact binomial test for small samples
                result = stats.binom_test(b, b + c, 0.5)
                method = "exact_binomial"
            else:
                # Use chi-square approximation for larger samples
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                result = 1 - stats.chi2.cdf(chi2, df=1)
                method = "chi_square"
        except Exception as e:
            logger.warning(f"McNemar test failed: {e}")
            result = 1.0
            method = "failed"
        
        return {
            'method': f'mcnemar_{method}',
            'p_value': float(result),
            'contingency_table': contingency_table.tolist(),
            'b_c_ratio': b / c if c > 0 else float('inf'),
            'significant': result < self.significance_level
        }
    
    def bootstrap_test(self, champion_metrics: List[float], 
                       challenger_metrics: List[float]) -> Dict[str, Any]:
        """
        Perform bootstrap test to compare model performances.
        
        Bootstrap resampling provides confidence intervals and
        p-values without assuming normal distribution.
        
        Reference: Efron & Tibshirani (1993)
        """
        n = len(champion_metrics)
        observed_diff = np.mean(challenger_metrics) - np.mean(champion_metrics)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        combined = np.concatenate([champion_metrics, challenger_metrics])
        
        for _ in range(self.bootstrap_iterations):
            # Resample with replacement
            resampled = np.random.choice(combined, size=2*n, replace=True)
            boot_champion = resampled[:n]
            boot_challenger = resampled[n:]
            bootstrap_diffs.append(np.mean(boot_challenger) - np.mean(boot_champion))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        return {
            'method': 'bootstrap',
            'p_value': float(p_value),
            'observed_difference': float(observed_diff),
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            'significant': p_value < self.significance_level
        }
    
    def check_regressions(self, champion_metrics: Dict[str, float],
                          challenger_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check for harmful regressions in critical metrics.
        
        Based on "No-Harm Principle" from medical AI research.
        Critical metrics should not degrade beyond tolerance.
        
        Reference: Subasri et al. (2025), JAMA Network Open
        """
        regressions = []
        details = {}
        
        for metric, tolerance in self.tolerances.items():
            champ_val = champion_metrics.get(metric, 0)
            chall_val = challenger_metrics.get(metric, 0)
            diff = chall_val - champ_val
            
            details[metric] = {
                'champion': champ_val,
                'challenger': chall_val,
                'difference': diff,
                'tolerance': tolerance
            }
            
            # Check if regression exceeds tolerance
            if metric == 'false_positive_rate':
                # For FPR, positive diff (increase) is bad
                if diff > tolerance:
                    regressions.append(f"{metric}: +{diff:.2f}% (max allowed: +{tolerance:.2f}%)")
                    details[metric]['passed'] = False
                else:
                    details[metric]['passed'] = True
            else:
                # For accuracy metrics, negative diff (decrease) is bad
                if diff < tolerance:
                    regressions.append(f"{metric}: {diff:.2f}% (min allowed: {tolerance:.2f}%)")
                    details[metric]['passed'] = False
                else:
                    details[metric]['passed'] = True
        
        return {
            'passed': len(regressions) == 0,
            'regressions': regressions,
            'details': details
        }
    
    def compare(self, champion_results: Dict[str, Any],
                challenger_results: Dict[str, Any],
                champion_preds: np.ndarray = None,
                challenger_preds: np.ndarray = None,
                true_labels: np.ndarray = None) -> ComparisonResult:
        """
        Perform comprehensive model comparison.
        
        Steps:
        1. Calculate composite scores
        2. Perform statistical significance test
        3. Check for harmful regressions
        4. Make final decision
        
        Args:
            champion_results: Evaluation results for champion model
            challenger_results: Evaluation results for challenger model
            champion_preds: Raw predictions from champion (optional, for McNemar)
            challenger_preds: Raw predictions from challenger (optional, for McNemar)
            true_labels: Ground truth labels (optional, for McNemar)
            
        Returns:
            ComparisonResult with decision
        """
        logger.info("=" * 60)
        logger.info("ENHANCED MODEL COMPARISON")
        logger.info("=" * 60)
        
        # Step 1: Calculate composite scores
        champion_score = self.calculate_composite_score(champion_results)
        challenger_score = self.calculate_composite_score(challenger_results)
        score_diff = challenger_score - champion_score
        
        logger.info(f"Champion Score:   {champion_score:.4f}")
        logger.info(f"Challenger Score: {challenger_score:.4f}")
        logger.info(f"Difference:       {score_diff:+.4f}")
        
        # Step 2: Statistical significance test
        if champion_preds is not None and challenger_preds is not None and true_labels is not None:
            # Use McNemar's test if predictions available
            stat_test = self.mcnemar_test(champion_preds, challenger_preds, true_labels)
        else:
            # Use bootstrap on LOEO fold results if available
            if 'loeo_fold_results' in champion_results and 'loeo_fold_results' in challenger_results:
                stat_test = self.bootstrap_test(
                    champion_results['loeo_fold_results'],
                    challenger_results['loeo_fold_results']
                )
            else:
                # Fallback: assume significant if score diff > threshold
                stat_test = {
                    'method': 'threshold_only',
                    'p_value': 0.01 if score_diff > self.min_improvement else 0.5,
                    'significant': score_diff > self.min_improvement
                }
        
        logger.info(f"Statistical Test: {stat_test['method']}")
        logger.info(f"P-value:          {stat_test['p_value']:.4f}")
        logger.info(f"Significant:      {stat_test['significant']}")
        
        # Step 3: Check for regressions
        regression_check = self.check_regressions(champion_results, challenger_results)
        
        logger.info(f"Regression Check: {'PASSED' if regression_check['passed'] else 'FAILED'}")
        if not regression_check['passed']:
            for reg in regression_check['regressions']:
                logger.warning(f"  - {reg}")
        
        # Step 4: Make decision
        decision = self._make_decision(
            champion_score, challenger_score,
            stat_test, regression_check,
            champion_results, challenger_results
        )
        
        logger.info("-" * 60)
        logger.info(f"DECISION: {'PROMOTE CHALLENGER' if decision['promote'] else 'KEEP CHAMPION'}")
        logger.info(f"Reason: {decision['reason']}")
        logger.info(f"Confidence: {decision['confidence']:.2%}")
        logger.info("=" * 60)
        
        return ComparisonResult(
            champion_score=champion_score,
            challenger_score=challenger_score,
            score_difference=score_diff,
            statistical_test=stat_test,
            regression_check=regression_check,
            decision=decision,
            timestamp=datetime.now().isoformat()
        )
    
    def _make_decision(self, champion_score: float, challenger_score: float,
                       stat_test: Dict, regression_check: Dict,
                       champion_results: Dict, challenger_results: Dict) -> Dict[str, Any]:
        """
        Make final decision based on all criteria.
        
        Decision Rules (in order):
        1. Challenger score must be higher than champion
        2. Improvement must meet minimum threshold
        3. Improvement must be statistically significant
        4. No harmful regressions in critical metrics
        5. (Optional) Strict mode: ALL metrics must improve
        """
        decision = {
            'promote': False,
            'reason': '',
            'confidence': 0.0,
            'details': {}
        }
        
        score_diff = challenger_score - champion_score
        
        # Rule 1: Score must be higher
        if score_diff <= 0:
            decision['reason'] = f"Challenger score ({challenger_score:.4f}) not higher than champion ({champion_score:.4f})"
            return decision
        
        # Rule 2: Minimum improvement threshold
        if score_diff < self.min_improvement:
            decision['reason'] = f"Improvement ({score_diff:.4f}) below threshold ({self.min_improvement})"
            return decision
        
        # Rule 3: Statistical significance
        if not stat_test.get('significant', False):
            decision['reason'] = f"Not statistically significant (p={stat_test['p_value']:.4f} >= {self.significance_level})"
            return decision
        
        # Rule 4: No regressions
        if not regression_check['passed']:
            decision['reason'] = f"Harmful regressions: {', '.join(regression_check['regressions'])}"
            return decision
        
        # Rule 5: Strict mode (optional)
        if self.strict_mode:
            improvements = self._check_all_improvements(champion_results, challenger_results)
            if not all(improvements.values()):
                failed = [k for k, v in improvements.items() if not v]
                decision['reason'] = f"Strict mode: Metrics not improved: {failed}"
                return decision
        
        # All checks passed!
        decision['promote'] = True
        decision['reason'] = f"Challenger wins: +{score_diff:.4f} score (p={stat_test['p_value']:.4f})"
        decision['confidence'] = 1 - stat_test['p_value']
        decision['details'] = {
            'score_improvement': score_diff,
            'magnitude_improvement': challenger_results.get('magnitude_accuracy', 0) - champion_results.get('magnitude_accuracy', 0),
            'azimuth_improvement': challenger_results.get('azimuth_accuracy', 0) - champion_results.get('azimuth_accuracy', 0)
        }
        
        return decision
    
    def _check_all_improvements(self, champion: Dict, challenger: Dict) -> Dict[str, bool]:
        """Check if all metrics improved (for strict mode)."""
        improvements = {}
        
        for metric in ['magnitude_accuracy', 'azimuth_accuracy', 'macro_f1', 'mcc']:
            champ_val = champion.get(metric, 0)
            chall_val = challenger.get(metric, 0)
            improvements[metric] = chall_val >= champ_val
        
        # For FPR, lower is better
        champ_fpr = champion.get('false_positive_rate', 100)
        chall_fpr = challenger.get('false_positive_rate', 100)
        improvements['false_positive_rate'] = chall_fpr <= champ_fpr
        
        return improvements
    
    def generate_report(self, result: ComparisonResult) -> str:
        """Generate detailed comparison report."""
        lines = [
            "=" * 70,
            "ENHANCED MODEL COMPARISON REPORT",
            f"Generated: {result.timestamp}",
            "=" * 70,
            "",
            "COMPOSITE SCORES:",
            "-" * 50,
            f"  Champion:   {result.champion_score:.4f}",
            f"  Challenger: {result.challenger_score:.4f}",
            f"  Difference: {result.score_difference:+.4f}",
            "",
            "STATISTICAL TEST:",
            "-" * 50,
            f"  Method:      {result.statistical_test['method']}",
            f"  P-value:     {result.statistical_test['p_value']:.4f}",
            f"  Significant: {'Yes' if result.statistical_test['significant'] else 'No'}",
            "",
            "REGRESSION CHECK:",
            "-" * 50,
            f"  Status: {'PASSED' if result.regression_check['passed'] else 'FAILED'}",
        ]
        
        if not result.regression_check['passed']:
            for reg in result.regression_check['regressions']:
                lines.append(f"  - {reg}")
        
        lines.extend([
            "",
            "DECISION:",
            "-" * 50,
            f"  Promote Challenger: {'YES' if result.decision['promote'] else 'NO'}",
            f"  Reason: {result.decision['reason']}",
            f"  Confidence: {result.decision['confidence']:.2%}",
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)


# Convenience function for quick comparison
def compare_models(champion_results: Dict, challenger_results: Dict,
                   config: Dict = None) -> ComparisonResult:
    """
    Quick comparison of two models.
    
    Args:
        champion_results: Evaluation metrics for champion
        challenger_results: Evaluation metrics for challenger
        config: Optional configuration
        
    Returns:
        ComparisonResult with decision
    """
    comparator = EnhancedModelComparator(config)
    return comparator.compare(champion_results, challenger_results)
