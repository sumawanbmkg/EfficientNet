#!/usr/bin/env python3
"""
Demo Evaluation Formula

Script ini mendemonstrasikan perhitungan formula evaluasi
untuk keputusan update model dengan contoh konkret.

Usage:
    python scripts/demo_evaluation.py
    python scripts/demo_evaluation.py --scenario improved
    python scripts/demo_evaluation.py --scenario regression
    python scripts/demo_evaluation.py --scenario marginal
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enhanced_comparator import EnhancedModelComparator


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """Print section header."""
    print(f"\n{title}")
    print("-" * 50)


def demo_composite_score_calculation(champion, challenger, weights):
    """Demonstrate composite score calculation step by step."""
    
    print_header("STEP 1: COMPOSITE SCORE CALCULATION")
    
    print_section("Bobot (Weights)")
    for metric, weight in weights.items():
        print(f"  {metric}: {weight:.0%}")
    
    print_section("Champion Metrics")
    for metric, value in champion.items():
        if metric != 'fold_results':
            print(f"  {metric}: {value}")
    
    print_section("Challenger Metrics")
    for metric, value in challenger.items():
        if metric != 'fold_results':
            print(f"  {metric}: {value}")
    
    # Calculate scores manually
    print_section("Perhitungan Champion Score")
    
    champion_components = []
    
    # Magnitude Accuracy
    norm_mag = champion['magnitude_accuracy'] / 100.0
    contrib_mag = weights['magnitude_accuracy'] * norm_mag
    champion_components.append(contrib_mag)
    print(f"  Magnitude: {weights['magnitude_accuracy']:.2f} × ({champion['magnitude_accuracy']}/100)")
    print(f"           = {weights['magnitude_accuracy']:.2f} × {norm_mag:.4f} = {contrib_mag:.4f}")
    
    # Azimuth Accuracy
    norm_azi = champion['azimuth_accuracy'] / 100.0
    contrib_azi = weights['azimuth_accuracy'] * norm_azi
    champion_components.append(contrib_azi)
    print(f"  Azimuth:   {weights['azimuth_accuracy']:.2f} × ({champion['azimuth_accuracy']}/100)")
    print(f"           = {weights['azimuth_accuracy']:.2f} × {norm_azi:.4f} = {contrib_azi:.4f}")
    
    # Macro F1
    norm_f1 = champion['macro_f1'] / 100.0
    contrib_f1 = weights['macro_f1'] * norm_f1
    champion_components.append(contrib_f1)
    print(f"  Macro F1:  {weights['macro_f1']:.2f} × ({champion['macro_f1']}/100)")
    print(f"           = {weights['macro_f1']:.2f} × {norm_f1:.4f} = {contrib_f1:.4f}")
    
    # MCC
    norm_mcc = (champion['mcc'] + 1) / 2.0
    contrib_mcc = weights['mcc'] * norm_mcc
    champion_components.append(contrib_mcc)
    print(f"  MCC:       {weights['mcc']:.2f} × (({champion['mcc']}+1)/2)")
    print(f"           = {weights['mcc']:.2f} × {norm_mcc:.4f} = {contrib_mcc:.4f}")
    
    # LOEO Stability (inverted)
    norm_loeo = 1.0 - champion['loeo_std'] / 100.0
    contrib_loeo = weights['loeo_stability'] * norm_loeo
    champion_components.append(contrib_loeo)
    print(f"  LOEO Stab: {weights['loeo_stability']:.2f} × (1 - {champion['loeo_std']}/100)")
    print(f"           = {weights['loeo_stability']:.2f} × {norm_loeo:.4f} = {contrib_loeo:.4f}")
    
    # FPR (inverted)
    norm_fpr = 1.0 - champion['false_positive_rate'] / 100.0
    contrib_fpr = weights['false_positive_rate'] * norm_fpr
    champion_components.append(contrib_fpr)
    print(f"  FPR:       {weights['false_positive_rate']:.2f} × (1 - {champion['false_positive_rate']}/100)")
    print(f"           = {weights['false_positive_rate']:.2f} × {norm_fpr:.4f} = {contrib_fpr:.4f}")
    
    champion_score = sum(champion_components)
    print(f"\n  TOTAL Champion Score = {' + '.join([f'{c:.4f}' for c in champion_components])}")
    print(f"                       = {champion_score:.4f}")
    
    # Calculate challenger score
    print_section("Perhitungan Challenger Score")
    
    challenger_components = []
    
    norm_mag = challenger['magnitude_accuracy'] / 100.0
    contrib_mag = weights['magnitude_accuracy'] * norm_mag
    challenger_components.append(contrib_mag)
    
    norm_azi = challenger['azimuth_accuracy'] / 100.0
    contrib_azi = weights['azimuth_accuracy'] * norm_azi
    challenger_components.append(contrib_azi)
    
    norm_f1 = challenger['macro_f1'] / 100.0
    contrib_f1 = weights['macro_f1'] * norm_f1
    challenger_components.append(contrib_f1)
    
    norm_mcc = (challenger['mcc'] + 1) / 2.0
    contrib_mcc = weights['mcc'] * norm_mcc
    challenger_components.append(contrib_mcc)
    
    norm_loeo = 1.0 - challenger['loeo_std'] / 100.0
    contrib_loeo = weights['loeo_stability'] * norm_loeo
    challenger_components.append(contrib_loeo)
    
    norm_fpr = 1.0 - challenger['false_positive_rate'] / 100.0
    contrib_fpr = weights['false_positive_rate'] * norm_fpr
    challenger_components.append(contrib_fpr)
    
    challenger_score = sum(challenger_components)
    print(f"  TOTAL Challenger Score = {challenger_score:.4f}")
    
    print_section("Score Comparison")
    diff = challenger_score - champion_score
    print(f"  Champion Score:   {champion_score:.4f}")
    print(f"  Challenger Score: {challenger_score:.4f}")
    print(f"  Difference (ΔS): {diff:+.4f} ({diff/champion_score*100:+.2f}%)")
    
    return champion_score, challenger_score


def demo_statistical_test(champion, challenger):
    """Demonstrate statistical significance testing."""
    
    print_header("STEP 2: STATISTICAL SIGNIFICANCE TEST")
    
    if 'fold_results' in champion and 'fold_results' in challenger:
        print_section("Bootstrap Test (menggunakan hasil per-fold)")
        
        champ_folds = champion['fold_results']
        chall_folds = challenger['fold_results']
        
        print(f"  Champion fold results:   {champ_folds}")
        print(f"  Challenger fold results: {chall_folds}")
        
        mean_champ = np.mean(champ_folds)
        mean_chall = np.mean(chall_folds)
        observed_diff = mean_chall - mean_champ
        
        print(f"\n  Mean Champion:   {mean_champ:.2f}%")
        print(f"  Mean Challenger: {mean_chall:.2f}%")
        print(f"  Observed Diff:   {observed_diff:+.2f}%")
        
        # Simple bootstrap
        n_iterations = 1000
        combined = champ_folds + chall_folds
        n = len(champ_folds)
        
        bootstrap_diffs = []
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_iterations):
            resampled = np.random.choice(combined, size=2*n, replace=True)
            boot_champ = resampled[:n]
            boot_chall = resampled[n:]
            bootstrap_diffs.append(np.mean(boot_chall) - np.mean(boot_champ))
        
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        print(f"\n  Bootstrap iterations: {n_iterations}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        significance_level = 0.05
        is_significant = p_value < significance_level
        
        print(f"\n  Significance level: {significance_level}")
        print(f"  Is significant? {p_value:.4f} < {significance_level} → {'YES ✓' if is_significant else 'NO ✗'}")
        
        return p_value, is_significant
    
    return 0.5, False


def demo_regression_check(champion, challenger):
    """Demonstrate regression checking."""
    
    print_header("STEP 3: REGRESSION CHECK (No-Harm Principle)")
    
    tolerances = {
        'magnitude_accuracy': -1.0,
        'large_recall': -2.0,
        'false_positive_rate': 1.0
    }
    
    print_section("Toleransi yang Diizinkan")
    print(f"  Magnitude Accuracy: max {tolerances['magnitude_accuracy']}% drop")
    print(f"  Large Recall:       max {tolerances['large_recall']}% drop")
    print(f"  False Positive Rate: max +{tolerances['false_positive_rate']}% increase")
    
    print_section("Pengecekan Regresi")
    
    regressions = []
    
    # Magnitude Accuracy
    diff_mag = challenger['magnitude_accuracy'] - champion['magnitude_accuracy']
    passed_mag = diff_mag >= tolerances['magnitude_accuracy']
    status_mag = "✓ PASS" if passed_mag else "✗ FAIL"
    print(f"\n  Magnitude Accuracy:")
    print(f"    Champion:   {champion['magnitude_accuracy']:.2f}%")
    print(f"    Challenger: {challenger['magnitude_accuracy']:.2f}%")
    print(f"    Difference: {diff_mag:+.2f}%")
    print(f"    Tolerance:  ≥ {tolerances['magnitude_accuracy']}%")
    print(f"    Result:     {diff_mag:+.2f}% ≥ {tolerances['magnitude_accuracy']}% → {status_mag}")
    if not passed_mag:
        regressions.append(f"Magnitude Accuracy: {diff_mag:+.2f}%")
    
    # Large Recall
    diff_recall = challenger['large_recall'] - champion['large_recall']
    passed_recall = diff_recall >= tolerances['large_recall']
    status_recall = "✓ PASS" if passed_recall else "✗ FAIL"
    print(f"\n  Large Earthquake Recall:")
    print(f"    Champion:   {champion['large_recall']:.2f}%")
    print(f"    Challenger: {challenger['large_recall']:.2f}%")
    print(f"    Difference: {diff_recall:+.2f}%")
    print(f"    Tolerance:  ≥ {tolerances['large_recall']}%")
    print(f"    Result:     {diff_recall:+.2f}% ≥ {tolerances['large_recall']}% → {status_recall}")
    if not passed_recall:
        regressions.append(f"Large Recall: {diff_recall:+.2f}%")
    
    # False Positive Rate
    diff_fpr = challenger['false_positive_rate'] - champion['false_positive_rate']
    passed_fpr = diff_fpr <= tolerances['false_positive_rate']
    status_fpr = "✓ PASS" if passed_fpr else "✗ FAIL"
    print(f"\n  False Positive Rate:")
    print(f"    Champion:   {champion['false_positive_rate']:.2f}%")
    print(f"    Challenger: {challenger['false_positive_rate']:.2f}%")
    print(f"    Difference: {diff_fpr:+.2f}%")
    print(f"    Tolerance:  ≤ +{tolerances['false_positive_rate']}%")
    print(f"    Result:     {diff_fpr:+.2f}% ≤ +{tolerances['false_positive_rate']}% → {status_fpr}")
    if not passed_fpr:
        regressions.append(f"FPR: {diff_fpr:+.2f}%")
    
    passed = len(regressions) == 0
    
    print_section("Hasil Regression Check")
    if passed:
        print("  Status: ✓ PASSED - Tidak ada regresi berbahaya")
    else:
        print("  Status: ✗ FAILED - Ditemukan regresi:")
        for reg in regressions:
            print(f"    - {reg}")
    
    return passed, regressions


def demo_final_decision(champion_score, challenger_score, p_value, is_significant, 
                        regression_passed, regressions):
    """Demonstrate final decision making."""
    
    print_header("STEP 4: FINAL DECISION")
    
    min_improvement = 0.005
    significance_level = 0.05
    
    print_section("Kriteria Keputusan")
    print(f"  1. Score improvement ≥ {min_improvement} ({min_improvement*100:.1f}%)")
    print(f"  2. Statistical significance: p < {significance_level}")
    print(f"  3. No harmful regressions")
    
    print_section("Evaluasi Kriteria")
    
    score_diff = challenger_score - champion_score
    
    # Criterion 1
    crit1_pass = score_diff >= min_improvement
    print(f"\n  Kriteria 1: Score Improvement")
    print(f"    ΔS = {score_diff:.4f}")
    print(f"    Threshold = {min_improvement}")
    print(f"    {score_diff:.4f} ≥ {min_improvement} → {'✓ PASS' if crit1_pass else '✗ FAIL'}")
    
    # Criterion 2
    crit2_pass = is_significant
    print(f"\n  Kriteria 2: Statistical Significance")
    print(f"    p-value = {p_value:.4f}")
    print(f"    Threshold = {significance_level}")
    print(f"    {p_value:.4f} < {significance_level} → {'✓ PASS' if crit2_pass else '✗ FAIL'}")
    
    # Criterion 3
    crit3_pass = regression_passed
    print(f"\n  Kriteria 3: No Harmful Regressions")
    print(f"    Regressions found: {len(regressions)}")
    print(f"    → {'✓ PASS' if crit3_pass else '✗ FAIL'}")
    
    # Final decision
    all_pass = crit1_pass and crit2_pass and crit3_pass
    
    print_section("KEPUTUSAN AKHIR")
    print()
    
    if all_pass:
        confidence = 1 - p_value
        print("  ╔════════════════════════════════════════════════════════════╗")
        print("  ║                                                            ║")
        print("  ║   ✅ PROMOTE CHALLENGER - Model baru layak di-deploy!     ║")
        print("  ║                                                            ║")
        print("  ╚════════════════════════════════════════════════════════════╝")
        print()
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Reason: Challenger wins with +{score_diff:.4f} score (p={p_value:.4f})")
    else:
        print("  ╔════════════════════════════════════════════════════════════╗")
        print("  ║                                                            ║")
        print("  ║   ❌ REJECT - Keep Champion model                          ║")
        print("  ║                                                            ║")
        print("  ╚════════════════════════════════════════════════════════════╝")
        print()
        
        if not crit1_pass:
            print(f"  Reason: Insufficient improvement ({score_diff:.4f} < {min_improvement})")
        elif not crit2_pass:
            print(f"  Reason: Not statistically significant (p={p_value:.4f})")
        else:
            print(f"  Reason: Harmful regressions detected: {regressions}")
    
    return all_pass


def get_scenario_data(scenario):
    """Get test data for different scenarios."""
    
    # Base champion (current production model)
    champion = {
        'magnitude_accuracy': 97.53,
        'azimuth_accuracy': 69.30,
        'macro_f1': 72.15,
        'mcc': 0.68,
        'loeo_std': 2.5,
        'false_positive_rate': 3.2,
        'large_recall': 95.0,
        'fold_results': [96.8, 97.2, 98.1, 97.5, 96.9, 97.8, 98.0, 97.3, 98.5, 96.2]
    }
    
    if scenario == 'improved':
        # Scenario: Clear improvement
        challenger = {
            'magnitude_accuracy': 98.10,
            'azimuth_accuracy': 71.50,
            'macro_f1': 74.80,
            'mcc': 0.72,
            'loeo_std': 2.0,
            'false_positive_rate': 2.8,
            'large_recall': 96.5,
            'fold_results': [97.5, 98.0, 98.5, 98.2, 97.8, 98.3, 98.6, 98.0, 99.0, 97.1]
        }
        description = "Skenario: Challenger LEBIH BAIK secara signifikan"
        
    elif scenario == 'regression':
        # Scenario: Has regression in critical metric
        challenger = {
            'magnitude_accuracy': 96.20,  # REGRESSION!
            'azimuth_accuracy': 72.00,
            'macro_f1': 75.00,
            'mcc': 0.73,
            'loeo_std': 1.8,
            'false_positive_rate': 2.5,
            'large_recall': 92.0,  # REGRESSION!
            'fold_results': [95.8, 96.2, 96.5, 96.0, 95.9, 96.3, 96.6, 96.0, 97.0, 95.7]
        }
        description = "Skenario: Challenger memiliki REGRESI pada metrik kritis"
        
    elif scenario == 'marginal':
        # Scenario: Marginal improvement (not significant)
        challenger = {
            'magnitude_accuracy': 97.60,
            'azimuth_accuracy': 69.50,
            'macro_f1': 72.30,
            'mcc': 0.69,
            'loeo_std': 2.4,
            'false_positive_rate': 3.1,
            'large_recall': 95.2,
            'fold_results': [96.9, 97.3, 98.2, 97.6, 97.0, 97.9, 98.1, 97.4, 98.6, 96.3]
        }
        description = "Skenario: Challenger hanya SEDIKIT lebih baik (tidak signifikan)"
        
    else:  # default
        challenger = {
            'magnitude_accuracy': 98.10,
            'azimuth_accuracy': 71.50,
            'macro_f1': 74.80,
            'mcc': 0.72,
            'loeo_std': 2.0,
            'false_positive_rate': 2.8,
            'large_recall': 96.5,
            'fold_results': [97.5, 98.0, 98.5, 98.2, 97.8, 98.3, 98.6, 98.0, 99.0, 97.1]
        }
        description = "Skenario: Default (Challenger lebih baik)"
    
    return champion, challenger, description


def main():
    parser = argparse.ArgumentParser(description='Demo Evaluation Formula')
    parser.add_argument('--scenario', type=str, default='improved',
                        choices=['improved', 'regression', 'marginal'],
                        help='Scenario to demonstrate')
    args = parser.parse_args()
    
    # Get scenario data
    champion, challenger, description = get_scenario_data(args.scenario)
    
    # Default weights
    weights = {
        'magnitude_accuracy': 0.35,
        'azimuth_accuracy': 0.15,
        'macro_f1': 0.20,
        'mcc': 0.15,
        'loeo_stability': 0.10,
        'false_positive_rate': 0.05
    }
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  DEMO: FORMULA EVALUASI UNTUK KEPUTUSAN UPDATE MODEL".center(66) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print(f"\n{description}")
    
    # Step 1: Composite Score
    champion_score, challenger_score = demo_composite_score_calculation(
        champion, challenger, weights
    )
    
    # Step 2: Statistical Test
    p_value, is_significant = demo_statistical_test(champion, challenger)
    
    # Step 3: Regression Check
    regression_passed, regressions = demo_regression_check(champion, challenger)
    
    # Step 4: Final Decision
    decision = demo_final_decision(
        champion_score, challenger_score,
        p_value, is_significant,
        regression_passed, regressions
    )
    
    print("\n" + "=" * 70)
    print("Demo selesai!")
    print("=" * 70 + "\n")
    
    return 0 if decision else 1


if __name__ == '__main__':
    sys.exit(main())
