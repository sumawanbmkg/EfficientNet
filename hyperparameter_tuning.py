"""
Hyperparameter Tuning for Multi-Task Earthquake CNN
Uses grid search or random search to find optimal hyperparameters
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
from train_multi_task import EarthquakeTrainer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning with grid/random search"""
    
    def __init__(self, base_config, param_grid, output_dir='tuning_results'):
        """
        Initialize tuner
        
        Args:
            base_config: Base configuration dictionary
            param_grid: Dictionary of parameters to tune
            output_dir: Directory to save results
        """
        self.base_config = base_config
        self.param_grid = param_grid
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = []
    
    def grid_search(self, max_trials=None):
        """
        Grid search over parameter combinations
        
        Args:
            max_trials: Maximum number of trials (None = all combinations)
        """
        # Generate all combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        logger.info(f"Grid search: {len(combinations)} combinations")
        
        for i, combo in enumerate(combinations):
            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {i+1}/{len(combinations)}")
            logger.info(f"{'='*80}")
            
            # Create config for this trial
            config = self.base_config.copy()
            for key, value in zip(keys, combo):
                config[key] = value
            
            logger.info(f"Config: {json.dumps({k: config[k] for k in keys}, indent=2)}")
            
            # Train model
            try:
                trainer = EarthquakeTrainer(config, output_dir=self.output_dir)
                history = trainer.train()
                
                # Get best validation metrics
                best_epoch = trainer.best_epoch
                best_val_loss = trainer.best_val_loss
                best_val_mag_acc = history['val_mag_acc'][best_epoch]
                best_val_az_acc = history['val_az_acc'][best_epoch]
                
                # Save results
                result = {
                    'trial': i + 1,
                    'config': {k: config[k] for k in keys},
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val_loss,
                    'best_val_mag_acc': best_val_mag_acc,
                    'best_val_az_acc': best_val_az_acc,
                    'exp_dir': trainer.exp_dir
                }
                
                self.results.append(result)
                
                logger.info(f"Trial {i+1} completed:")
                logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
                logger.info(f"  Best Val Mag Acc: {best_val_mag_acc:.4f}")
                logger.info(f"  Best Val Az Acc: {best_val_az_acc:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save all results
        self._save_results()
        
        return self.results
    
    def random_search(self, n_trials=10):
        """
        Random search over parameter space
        
        Args:
            n_trials: Number of random trials
        """
        logger.info(f"Random search: {n_trials} trials")
        
        keys = list(self.param_grid.keys())
        
        for i in range(n_trials):
            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {i+1}/{n_trials}")
            logger.info(f"{'='*80}")
            
            # Random sample
            config = self.base_config.copy()
            for key in keys:
                config[key] = np.random.choice(self.param_grid[key])
            
            logger.info(f"Config: {json.dumps({k: config[k] for k in keys}, indent=2)}")
            
            # Train model
            try:
                trainer = EarthquakeTrainer(config, output_dir=self.output_dir)
                history = trainer.train()
                
                # Get best validation metrics
                best_epoch = trainer.best_epoch
                best_val_loss = trainer.best_val_loss
                best_val_mag_acc = history['val_mag_acc'][best_epoch]
                best_val_az_acc = history['val_az_acc'][best_epoch]
                
                # Save results
                result = {
                    'trial': i + 1,
                    'config': {k: config[k] for k in keys},
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val_loss,
                    'best_val_mag_acc': best_val_mag_acc,
                    'best_val_az_acc': best_val_az_acc,
                    'exp_dir': trainer.exp_dir
                }
                
                self.results.append(result)
                
                logger.info(f"Trial {i+1} completed:")
                logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
                logger.info(f"  Best Val Mag Acc: {best_val_mag_acc:.4f}")
                logger.info(f"  Best Val Az Acc: {best_val_az_acc:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save all results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save tuning results"""
        # Save as JSON
        results_path = os.path.join(self.output_dir, 'tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Save as CSV
        df_results = []
        for result in self.results:
            row = {
                'trial': result['trial'],
                'best_epoch': result['best_epoch'],
                'best_val_loss': result['best_val_loss'],
                'best_val_mag_acc': result['best_val_mag_acc'],
                'best_val_az_acc': result['best_val_az_acc'],
                'exp_dir': result['exp_dir']
            }
            row.update(result['config'])
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        df.to_csv(os.path.join(self.output_dir, 'tuning_results.csv'), index=False)
        
        logger.info(f"Tuning results saved to: {self.output_dir}")
        
        # Print best configuration
        best_result = min(self.results, key=lambda x: x['best_val_loss'])
        logger.info(f"\n{'='*80}")
        logger.info("BEST CONFIGURATION:")
        logger.info(f"{'='*80}")
        logger.info(f"Trial: {best_result['trial']}")
        logger.info(f"Config: {json.dumps(best_result['config'], indent=2)}")
        logger.info(f"Best Val Loss: {best_result['best_val_loss']:.4f}")
        logger.info(f"Best Val Mag Acc: {best_result['best_val_mag_acc']:.4f}")
        logger.info(f"Best Val Az Acc: {best_result['best_val_az_acc']:.4f}")
        logger.info(f"Experiment Dir: {best_result['exp_dir']}")


def main():
    """Main tuning function"""
    # Base configuration
    base_config = {
        'dataset_dir': 'dataset_spectrogram_ssh',  # Use existing dataset
        'batch_size': 16,
        'val_split': 0.2,
        'test_split': 0.1,
        'num_workers': 0,
        'seed': 42,
        'pretrained': True,
        'num_magnitude_classes': 5,
        'num_azimuth_classes': 8,
        'learn_weights': True,
        'epochs': 30,  # Reduced for tuning
    }
    
    # Parameters to tune
    param_grid = {
        'backbone': ['resnet18', 'resnet50'],
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'dropout_rate': [0.3, 0.5, 0.7],
        'optimizer': ['adam', 'sgd'],
        'weight_decay': [1e-4, 1e-5]
    }
    
    # Create tuner
    tuner = HyperparameterTuner(base_config, param_grid, output_dir='tuning_results')
    
    # Run random search (faster than grid search)
    results = tuner.random_search(n_trials=5)  # Adjust n_trials as needed
    
    print(f"\n[OK] Hyperparameter tuning completed!")
    print(f"Results saved to: tuning_results/")


if __name__ == '__main__':
    main()
