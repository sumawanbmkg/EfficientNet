"""
Earthquake Model Auto-Update Pipeline

A comprehensive pipeline for automatically updating earthquake prediction models
with new validated data using the Champion-Challenger pattern.

Modules:
    - data_ingestion: Handle new earthquake event data
    - data_validator: Validate incoming data
    - trainer: Train new candidate models
    - evaluator: Evaluate model performance
    - model_comparator: Compare champion vs challenger
    - deployer: Deploy winning models
    - utils: Utility functions

Author: Earthquake Prediction Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Earthquake Prediction Research Team"

from .data_validator import DataValidator
from .data_ingestion import DataIngestion
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .model_comparator import ModelComparator
from .deployer import ModelDeployer

__all__ = [
    "DataValidator",
    "DataIngestion", 
    "ModelTrainer",
    "ModelEvaluator",
    "ModelComparator",
    "ModelDeployer"
]
