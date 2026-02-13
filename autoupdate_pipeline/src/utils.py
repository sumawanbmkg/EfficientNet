"""
Utility functions for the auto-update pipeline.
"""

import os
import json
import yaml
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Setup logging
def setup_logging(log_file: str = None, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger("autoupdate_pipeline")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = "config/pipeline_config.yaml"
    config_path = Path(__file__).parent.parent / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_registry(registry_path: str = "config/model_registry.json") -> Dict[str, Any]:
    """Load model registry from JSON file."""
    registry_path = Path(__file__).parent.parent / registry_path
    
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    return registry


def save_registry(registry: Dict[str, Any], registry_path: str = "config/model_registry.json"):
    """Save model registry to JSON file."""
    registry_path = Path(__file__).parent.parent / registry_path
    
    registry['last_updated'] = datetime.now().isoformat()
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)


def generate_model_id(architecture: str, version: str = None) -> str:
    """Generate unique model ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if version:
        return f"{architecture}_{version}_{timestamp}"
    return f"{architecture}_{timestamp}"


def increment_version(current_version: str) -> str:
    """Increment semantic version (e.g., 1.0.0 -> 1.0.1)."""
    parts = current_version.split('.')
    parts[-1] = str(int(parts[-1]) + 1)
    return '.'.join(parts)


def copy_model_files(src_dir: Path, dst_dir: Path, files: List[str] = None):
    """Copy model files from source to destination."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if files is None:
        # Copy all files
        for item in src_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dst_dir / item.name)
    else:
        # Copy specific files
        for file in files:
            src_file = src_dir / file
            if src_file.exists():
                shutil.copy2(src_file, dst_dir / file)


def archive_model(model_dir: Path, archive_dir: Path, model_id: str):
    """Archive a model to the archive directory."""
    archive_path = archive_dir / model_id
    archive_path.mkdir(parents=True, exist_ok=True)
    
    for item in model_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, archive_path / item.name)
    
    return archive_path


def cleanup_old_archives(archive_dir: Path, max_keep: int = 10):
    """Remove old archived models, keeping only the most recent ones."""
    if not archive_dir.exists():
        return
    
    archives = sorted(archive_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for archive in archives[max_keep:]:
        if archive.is_dir():
            shutil.rmtree(archive)


def log_pipeline_event(event_type: str, details: Dict[str, Any], log_file: str = "logs/pipeline_history.json"):
    """Log a pipeline event to the history file."""
    log_path = Path(__file__).parent.parent / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing history
    if log_path.exists():
        with open(log_path, 'r') as f:
            history = json.load(f)
    else:
        history = {"events": []}
    
    # Add new event
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    }
    history["events"].append(event)
    
    # Save updated history
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted composite score from metrics."""
    score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in metrics:
            # Normalize metric to 0-1 range (assuming percentages)
            normalized = metrics[metric] / 100.0 if metrics[metric] > 1 else metrics[metric]
            
            # Invert for metrics where lower is better (e.g., false_positive_rate)
            if 'false' in metric.lower() or 'error' in metric.lower():
                normalized = 1.0 - normalized
            
            score += weight * normalized
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable table string."""
    lines = ["=" * 50]
    lines.append("METRICS SUMMARY")
    lines.append("=" * 50)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {metric:30s}: {value:8.2f}%")
        else:
            lines.append(f"  {metric:30s}: {value}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def check_trigger_conditions(registry: Dict, config: Dict) -> Dict[str, Any]:
    """Check if pipeline trigger conditions are met."""
    triggers = config.get('triggers', {})
    pending = registry.get('pending_events', {})
    
    result = {
        "should_trigger": False,
        "reasons": [],
        "pending_count": pending.get('count', 0),
        "min_required": triggers.get('min_new_events', 20)
    }
    
    # Check minimum events
    if pending.get('count', 0) >= triggers.get('min_new_events', 20):
        result["should_trigger"] = True
        result["reasons"].append(f"Minimum events reached ({pending['count']} >= {triggers['min_new_events']})")
    
    # Check time since last training
    last_run = registry.get('pipeline_history', {}).get('last_run')
    if last_run:
        last_run_date = datetime.fromisoformat(last_run)
        days_since = (datetime.now() - last_run_date).days
        max_days = triggers.get('max_days_between_training', 90)
        
        if days_since >= max_days:
            result["should_trigger"] = True
            result["reasons"].append(f"Max days exceeded ({days_since} >= {max_days})")
        
        result["days_since_last_run"] = days_since
    
    return result
