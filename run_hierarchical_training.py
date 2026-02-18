"""
Run Hierarchical Training for Both Models
==========================================
Trains both EfficientNet and ConvNeXt with hierarchical classification approach.

Usage:
    python run_hierarchical_training.py
    python run_hierarchical_training.py --model efficientnet
    python run_hierarchical_training.py --model convnext
    python run_hierarchical_training.py --model both
"""

import os
import sys
import json
from datetime import datetime

def run_training(model_type):
    """Run training for specified model"""
    from train_hierarchical_model import main
    
    print(f"\n{'='*70}")
    print(f"Starting {model_type.upper()} Hierarchical Training")
    print(f"{'='*70}\n")
    
    results, save_dir = main(model_type)
    return results, save_dir

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Hierarchical Classification Training')
    parser.add_argument('--model', type=str, default='both',
                        choices=['efficientnet', 'convnext', 'both'],
                        help='Model to train (default: both)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("HIERARCHICAL CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model(s): {args.model}")
    print("=" * 70)
    
    all_results = {}
    
    if args.model in ['efficientnet', 'both']:
        try:
            results_eff, dir_eff = run_training('efficientnet')
            all_results['efficientnet'] = {
                'results': {
                    'stage1': results_eff['stage1']['best_acc'],
                    'stage2': results_eff['stage2']['best_acc'],
                    'stage3': results_eff['stage3']['best_acc']
                },
                'save_dir': dir_eff
            }
        except Exception as e:
            print(f"Error training EfficientNet: {e}")
            all_results['efficientnet'] = {'error': str(e)}
    
    if args.model in ['convnext', 'both']:
        try:
            results_conv, dir_conv = run_training('convnext')
            all_results['convnext'] = {
                'results': {
                    'stage1': results_conv['stage1']['best_acc'],
                    'stage2': results_conv['stage2']['best_acc'],
                    'stage3': results_conv['stage3']['best_acc']
                },
                'save_dir': dir_conv
            }
        except Exception as e:
            print(f"Error training ConvNeXt: {e}")
            all_results['convnext'] = {'error': str(e)}
    
    # Print comparison
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'Stage 1 (Binary)':<20} {'Stage 2 (Mag)':<20} {'Stage 3 (Azi)':<20}")
    print("-" * 75)
    
    for model_name, data in all_results.items():
        if 'error' in data:
            print(f"{model_name:<15} ERROR: {data['error']}")
        else:
            r = data['results']
            print(f"{model_name:<15} {r['stage1']:.2f}%{'':<14} {r['stage2']:.2f}%{'':<14} {r['stage3']:.2f}%")
    
    print("\n" + "=" * 70)
    
    # Save comparison
    os.makedirs('experiments_hierarchical', exist_ok=True)
    comparison_file = f"experiments_hierarchical/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Comparison saved to: {comparison_file}")
    
    return all_results

if __name__ == '__main__':
    main()
