"""
Leave-One-Event-Out (LOEO) Cross-Validation Splitter
Ensures no data leakage by splitting based on earthquake Event ID.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Generator
import json


class LOEOSplitter:
    """
    Leave-One-Event-Out Cross-Validation Splitter.
    
    Unlike random splitting, LOEO ensures that all samples from the same
    earthquake event are either in training or testing, never both.
    This prevents temporal data leakage from windowed samples.
    
    Args:
        n_folds (int): Number of folds. Default: 10
        random_state (int): Random seed for reproducibility. Default: 42
    """
    
    def __init__(self, n_folds=10, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        
    def get_event_id(self, sample_path: str) -> str:
        """
        Extract event ID from sample path.
        
        Example:
            'Medium_SCN_2018-01-17_H06_3comp_spec.png' -> 'SCN_20180117'
        """
        parts = sample_path.split('_')
        if len(parts) >= 3:
            station = parts[1]
            date = parts[2].replace('-', '')
            return f"{station}_{date}"
        return sample_path
        
    def group_by_event(self, samples: List[str]) -> dict:
        """
        Group samples by their earthquake event ID.
        
        Args:
            samples: List of sample paths/names
            
        Returns:
            Dictionary mapping event_id -> list of sample indices
        """
        event_groups = defaultdict(list)
        for idx, sample in enumerate(samples):
            event_id = self.get_event_id(sample)
            event_groups[event_id].append(idx)
        return dict(event_groups)
        
    def split(self, samples: List[str]) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate train/test splits for LOEO cross-validation.
        
        Args:
            samples: List of sample paths/names
            
        Yields:
            (train_indices, test_indices) for each fold
        """
        np.random.seed(self.random_state)
        
        # Group samples by event
        event_groups = self.group_by_event(samples)
        event_ids = list(event_groups.keys())
        
        # Shuffle events
        np.random.shuffle(event_ids)
        
        # Split events into folds
        n_events = len(event_ids)
        fold_size = n_events // self.n_folds
        
        for fold in range(self.n_folds):
            # Determine test events for this fold
            start_idx = fold * fold_size
            if fold == self.n_folds - 1:
                # Last fold gets remaining events
                test_events = event_ids[start_idx:]
            else:
                test_events = event_ids[start_idx:start_idx + fold_size]
            
            # Train events are all others
            train_events = [e for e in event_ids if e not in test_events]
            
            # Get sample indices
            train_indices = []
            for event in train_events:
                train_indices.extend(event_groups[event])
                
            test_indices = []
            for event in test_events:
                test_indices.extend(event_groups[event])
            
            yield train_indices, test_indices
            
    def get_fold_info(self, samples: List[str]) -> List[dict]:
        """
        Get detailed information about each fold.
        
        Returns:
            List of fold info dictionaries
        """
        event_groups = self.group_by_event(samples)
        event_ids = list(event_groups.keys())
        
        np.random.seed(self.random_state)
        np.random.shuffle(event_ids)
        
        fold_info = []
        n_events = len(event_ids)
        fold_size = n_events // self.n_folds
        
        for fold in range(self.n_folds):
            start_idx = fold * fold_size
            if fold == self.n_folds - 1:
                test_events = event_ids[start_idx:]
            else:
                test_events = event_ids[start_idx:start_idx + fold_size]
            
            train_events = [e for e in event_ids if e not in test_events]
            
            train_samples = sum(len(event_groups[e]) for e in train_events)
            test_samples = sum(len(event_groups[e]) for e in test_events)
            
            fold_info.append({
                'fold': fold + 1,
                'train_events': len(train_events),
                'test_events': len(test_events),
                'train_samples': train_samples,
                'test_samples': test_samples,
                'test_event_ids': test_events
            })
            
        return fold_info


# Example usage
if __name__ == "__main__":
    # Simulated sample list
    samples = [
        'Large_MLB_2021-04-16_H00_3comp_spec.png',
        'Large_MLB_2021-04-16_H01_3comp_spec.png',
        'Large_MLB_2021-04-16_H02_3comp_spec.png',
        'Medium_SCN_2018-01-17_H00_3comp_spec.png',
        'Medium_SCN_2018-01-17_H01_3comp_spec.png',
        'Normal_KPY_2020-01-01_H00_3comp_spec.png',
        'Normal_KPY_2020-01-01_H01_3comp_spec.png',
        # ... more samples
    ]
    
    splitter = LOEOSplitter(n_folds=10, random_state=42)
    
    # Get fold information
    fold_info = splitter.get_fold_info(samples)
    print("LOEO Fold Information:")
    print(json.dumps(fold_info, indent=2))
    
    # Generate splits
    print("\nGenerating splits...")
    for fold, (train_idx, test_idx) in enumerate(splitter.split(samples)):
        print(f"Fold {fold+1}: Train={len(train_idx)}, Test={len(test_idx)}")
