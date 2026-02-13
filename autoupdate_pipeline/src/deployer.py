"""
Model Deployer Module

Handles deployment of winning models to production.
Implements Champion-Challenger pattern with full model management.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .utils import (
    load_config, load_registry, save_registry,
    archive_model, cleanup_old_archives, increment_version,
    log_pipeline_event
)

logger = logging.getLogger("autoupdate_pipeline.deployer")


class ModelDeployer:
    """
    Deploys winning models to production.
    
    Features:
    - Backup current champion before replacement
    - Archive old models
    - Update model registry
    - Optional manual approval
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize deployer."""
        self.config = config or load_config()
        self.deploy_config = self.config.get('deployment', {})
        
        self.base_path = Path(__file__).parent.parent
        self.champion_path = self.base_path / self.config['paths']['champion_model']
        self.challenger_path = self.base_path / self.config['paths']['challenger_model']
        self.archive_path = self.base_path / self.config['paths']['archive_models']
        
        logger.info("ModelDeployer initialized")
    
    def deploy_challenger(self, comparison_results: Dict[str, Any], 
                         force: bool = False) -> Dict[str, Any]:
        """
        Deploy challenger model as new champion.
        
        Args:
            comparison_results: Results from ModelComparator
            force: Force deployment even if approval required
            
        Returns:
            Deployment results
        """
        result = {
            "success": False,
            "message": "",
            "old_champion": None,
            "new_champion": None,
            "archived_to": None
        }
        
        # Check if challenger won
        if not comparison_results['decision']['promote_challenger']:
            result["message"] = "Challenger did not win comparison. Deployment aborted."
            logger.warning(result["message"])
            return result
        
        # Check approval requirement
        if self.deploy_config.get('require_approval', True) and not force:
            result["message"] = "Manual approval required. Use force=True to override."
            logger.info(result["message"])
            return result
        
        logger.info("=" * 60)
        logger.info("STARTING DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            # Load registry
            registry = load_registry()
            old_champion = registry.get('champion', {})
            
            # Backup current champion
            if self.deploy_config.get('backup_champion', True) and self.champion_path.exists():
                old_model_id = old_champion.get('model_id', 'unknown')
                archive_dest = archive_model(self.champion_path, self.archive_path, old_model_id)
                result["archived_to"] = str(archive_dest)
                logger.info(f"Old champion archived to: {archive_dest}")
                
                # Add to archive list
                if 'archive' not in registry:
                    registry['archive'] = []
                registry['archive'].append({
                    'model_id': old_model_id,
                    'archived_at': datetime.now().isoformat(),
                    'path': str(archive_dest),
                    'metrics': old_champion.get('metrics', {})
                })
            
            # Clear champion directory
            if self.champion_path.exists():
                shutil.rmtree(self.champion_path)
            self.champion_path.mkdir(parents=True, exist_ok=True)
            
            # Copy challenger to champion
            for item in self.challenger_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.champion_path / item.name)
            
            logger.info(f"Challenger copied to champion directory")
            
            # Update registry
            new_version = increment_version(old_champion.get('version', '1.0.0'))
            challenger_results = comparison_results['challenger']['results']
            
            registry['champion'] = {
                'model_id': f"convnext_v{new_version}",
                'version': new_version,
                'architecture': 'convnext_tiny',
                'path': str(self.champion_path / 'best_model.pth'),
                'config_path': str(self.champion_path / 'class_mappings.json'),
                'deployed_at': datetime.now().isoformat(),
                'metrics': {
                    'magnitude_accuracy': challenger_results['magnitude_accuracy'],
                    'azimuth_accuracy': challenger_results['azimuth_accuracy'],
                    'magnitude_f1': challenger_results['magnitude_f1'],
                    'azimuth_f1': challenger_results['azimuth_f1'],
                    'composite_score': comparison_results['challenger']['composite_score']
                },
                'previous_version': old_champion.get('version'),
                'status': 'active'
            }
            
            # Clear challenger
            registry['challenger'] = None
            
            # Clear validated events (they're now part of the model)
            registry['validated_events'] = []
            registry['pending_events'] = {'count': 0, 'events': [], 'last_added': None}
            
            # Update pipeline history
            history = registry.get('pipeline_history', {})
            history['total_runs'] = history.get('total_runs', 0) + 1
            history['successful_updates'] = history.get('successful_updates', 0) + 1
            history['last_run'] = datetime.now().isoformat()
            registry['pipeline_history'] = history
            
            save_registry(registry)
            
            # Cleanup old archives
            max_archive = self.deploy_config.get('max_archive_models', 10)
            cleanup_old_archives(self.archive_path, max_archive)
            
            # Log event
            log_pipeline_event("deployment_complete", {
                'new_version': new_version,
                'old_version': old_champion.get('version'),
                'magnitude_accuracy': challenger_results['magnitude_accuracy'],
                'azimuth_accuracy': challenger_results['azimuth_accuracy']
            })
            
            result["success"] = True
            result["message"] = f"Deployment successful! New version: {new_version}"
            result["old_champion"] = old_champion
            result["new_champion"] = registry['champion']
            
            logger.info("=" * 60)
            logger.info("DEPLOYMENT COMPLETE")
            logger.info(f"New Champion Version: {new_version}")
            logger.info(f"Magnitude Accuracy: {challenger_results['magnitude_accuracy']:.2f}%")
            logger.info(f"Azimuth Accuracy: {challenger_results['azimuth_accuracy']:.2f}%")
            logger.info("=" * 60)
            
        except Exception as e:
            result["message"] = f"Deployment failed: {str(e)}"
            logger.error(result["message"], exc_info=True)
            
            # Update failure count
            registry = load_registry()
            history = registry.get('pipeline_history', {})
            history['failed_updates'] = history.get('failed_updates', 0) + 1
            registry['pipeline_history'] = history
            save_registry(registry)
            
            log_pipeline_event("deployment_failed", {'error': str(e)})
        
        return result
    
    def rollback(self, version: str = None) -> Dict[str, Any]:
        """
        Rollback to a previous model version.
        
        Args:
            version: Version to rollback to (latest archived if None)
            
        Returns:
            Rollback results
        """
        result = {
            "success": False,
            "message": "",
            "rolled_back_to": None
        }
        
        logger.info("=" * 60)
        logger.info("STARTING ROLLBACK")
        logger.info("=" * 60)
        
        try:
            registry = load_registry()
            archives = registry.get('archive', [])
            
            if not archives:
                result["message"] = "No archived models available for rollback"
                logger.warning(result["message"])
                return result
            
            # Find target version
            if version:
                target = next((a for a in archives if version in a['model_id']), None)
                if not target:
                    result["message"] = f"Version {version} not found in archives"
                    logger.warning(result["message"])
                    return result
            else:
                # Use most recent archive
                target = archives[-1]
            
            target_path = Path(target['path'])
            
            if not target_path.exists():
                result["message"] = f"Archive path not found: {target_path}"
                logger.error(result["message"])
                return result
            
            # Backup current champion
            current_champion = registry.get('champion', {})
            if self.champion_path.exists():
                current_id = current_champion.get('model_id', 'rollback_backup')
                archive_model(self.champion_path, self.archive_path, f"{current_id}_pre_rollback")
            
            # Clear and restore
            if self.champion_path.exists():
                shutil.rmtree(self.champion_path)
            shutil.copytree(target_path, self.champion_path)
            
            # Update registry
            registry['champion'] = {
                'model_id': target['model_id'],
                'version': target['model_id'].split('_v')[-1] if '_v' in target['model_id'] else '1.0.0',
                'architecture': 'convnext_tiny',
                'path': str(self.champion_path / 'best_model.pth'),
                'config_path': str(self.champion_path / 'class_mappings.json'),
                'deployed_at': datetime.now().isoformat(),
                'metrics': target.get('metrics', {}),
                'status': 'active',
                'rollback_from': current_champion.get('version')
            }
            
            save_registry(registry)
            
            log_pipeline_event("rollback_complete", {
                'rolled_back_to': target['model_id'],
                'from_version': current_champion.get('version')
            })
            
            result["success"] = True
            result["message"] = f"Rollback successful to {target['model_id']}"
            result["rolled_back_to"] = target['model_id']
            
            logger.info(f"Rollback complete: {target['model_id']}")
            
        except Exception as e:
            result["message"] = f"Rollback failed: {str(e)}"
            logger.error(result["message"], exc_info=True)
            log_pipeline_event("rollback_failed", {'error': str(e)})
        
        return result
    
    def list_archived_models(self) -> list:
        """List all archived models."""
        registry = load_registry()
        return registry.get('archive', [])
    
    def get_current_champion(self) -> Dict[str, Any]:
        """Get current champion model info."""
        registry = load_registry()
        return registry.get('champion', {})
    
    def get_model_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get model info by version number.
        
        Args:
            version: Version string (e.g., "1.0.1")
            
        Returns:
            Model info dict or None if not found
        """
        registry = load_registry()
        
        # Check champion
        champion = registry.get('champion', {})
        if champion.get('version') == version:
            return {**champion, 'status': 'champion'}
        
        # Check archives
        for archived in registry.get('archive', []):
            if version in archived.get('model_id', ''):
                return {**archived, 'status': 'archived'}
        
        return None
    
    def get_all_versions(self) -> List[Dict[str, Any]]:
        """
        Get list of all model versions with their status.
        
        Returns:
            List of model info dicts sorted by version
        """
        registry = load_registry()
        versions = []
        
        # Add champion
        champion = registry.get('champion', {})
        if champion:
            versions.append({
                'version': champion.get('version', '?'),
                'model_id': champion.get('model_id', ''),
                'status': 'champion',
                'deployed_at': champion.get('deployed_at', ''),
                'metrics': champion.get('metrics', {})
            })
        
        # Add archives
        for archived in registry.get('archive', []):
            version = archived.get('model_id', '').split('_v')[-1] if '_v' in archived.get('model_id', '') else '?'
            versions.append({
                'version': version,
                'model_id': archived.get('model_id', ''),
                'status': 'archived',
                'archived_at': archived.get('archived_at', ''),
                'metrics': archived.get('metrics', {})
            })
        
        return versions
    
    def create_model_metadata(self, model_dir: Path, model_info: Dict[str, Any]) -> Path:
        """
        Create metadata.json file for a model.
        
        Args:
            model_dir: Directory containing the model
            model_info: Model information dict
            
        Returns:
            Path to created metadata file
        """
        metadata = {
            'model_id': model_info.get('model_id', ''),
            'version': model_info.get('version', ''),
            'architecture': model_info.get('architecture', 'convnext_tiny'),
            'created_at': datetime.now().isoformat(),
            'metrics': model_info.get('metrics', {}),
            'training_config': model_info.get('training_config', {}),
            'training_data': model_info.get('training_data', {}),
            'status': model_info.get('status', 'active')
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Created metadata at: {metadata_path}")
        return metadata_path
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results dict
        """
        model1 = self.get_model_by_version(version1)
        model2 = self.get_model_by_version(version2)
        
        if not model1 or not model2:
            return {
                'success': False,
                'message': f"Model not found: {version1 if not model1 else version2}"
            }
        
        metrics1 = model1.get('metrics', {})
        metrics2 = model2.get('metrics', {})
        
        comparison = {
            'success': True,
            'model1': {
                'version': version1,
                'status': model1.get('status', ''),
                'metrics': metrics1
            },
            'model2': {
                'version': version2,
                'status': model2.get('status', ''),
                'metrics': metrics2
            },
            'differences': {}
        }
        
        # Calculate differences
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            diff = val2 - val1
            comparison['differences'][metric] = {
                'model1': val1,
                'model2': val2,
                'difference': diff,
                'better': 'model2' if diff > 0 else ('model1' if diff < 0 else 'equal')
            }
        
        return comparison
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary of model registry.
        
        Returns:
            Summary dict with counts and status
        """
        registry = load_registry()
        
        champion = registry.get('champion', {})
        archives = registry.get('archive', [])
        validated = registry.get('validated_events', [])
        history = registry.get('pipeline_history', {})
        
        return {
            'champion': {
                'model_id': champion.get('model_id', 'N/A'),
                'version': champion.get('version', 'N/A'),
                'magnitude_accuracy': champion.get('metrics', {}).get('magnitude_accuracy', 0),
                'azimuth_accuracy': champion.get('metrics', {}).get('azimuth_accuracy', 0),
                'composite_score': champion.get('metrics', {}).get('composite_score', 0)
            },
            'archive_count': len(archives),
            'validated_events_count': len(validated),
            'total_runs': history.get('total_runs', 0),
            'successful_updates': history.get('successful_updates', 0),
            'failed_updates': history.get('failed_updates', 0),
            'last_run': history.get('last_run', 'Never')
        }
