#!/usr/bin/env python
"""
Setup Champion Model

Copy current production model to champion directory.
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_registry, save_registry


def setup_champion():
    """Copy production model to champion directory."""
    config = load_config()
    base_path = Path(__file__).parent.parent
    
    # Source paths (production model)
    prod_model = base_path / config['paths']['production_model']
    prod_config = base_path / config['paths']['production_config']
    
    # Destination
    champion_dir = base_path / config['paths']['champion_model']
    
    print("\n" + "=" * 60)
    print("SETUP CHAMPION MODEL")
    print("=" * 60)
    
    print(f"\nSource model: {prod_model}")
    print(f"Destination:  {champion_dir}")
    
    # Check source exists
    if not prod_model.exists():
        print(f"\n❌ Production model not found: {prod_model}")
        print("\nAvailable ConvNeXt models:")
        
        convnext_dir = base_path.parent / "experiments_convnext"
        if convnext_dir.exists():
            for d in convnext_dir.iterdir():
                if d.is_dir():
                    model_file = d / "best_model.pth"
                    if model_file.exists():
                        print(f"   - {model_file}")
        return False
    
    # Create champion directory
    champion_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    print("\n[1/3] Copying model file...")
    shutil.copy2(prod_model, champion_dir / "best_model.pth")
    print("✅ Model copied")
    
    # Copy config if exists
    print("\n[2/3] Copying config...")
    if prod_config.exists():
        shutil.copy2(prod_config, champion_dir / "config.json")
        print("✅ Config copied")
    else:
        # Create default config
        default_config = {
            "architecture": "convnext_tiny",
            "num_mag_classes": 4,
            "num_azi_classes": 9,
            "input_size": 224
        }
        with open(champion_dir / "config.json", 'w') as f:
            json.dump(default_config, f, indent=2)
        print("✅ Default config created")
    
    # Create class mappings
    print("\n[3/3] Creating class mappings...")
    mappings = {
        "magnitude": {
            "0": "Large",
            "1": "Medium",
            "2": "Moderate",
            "3": "Normal"
        },
        "azimuth": {
            "0": "E",
            "1": "N",
            "2": "NE",
            "3": "NW",
            "4": "Normal",
            "5": "S",
            "6": "SE",
            "7": "SW",
            "8": "W"
        }
    }
    with open(champion_dir / "class_mappings.json", 'w') as f:
        json.dump(mappings, f, indent=2)
    print("✅ Class mappings created")
    
    # Update registry
    print("\n[4/4] Updating registry...")
    registry = load_registry()
    registry['champion'] = {
        'model_id': 'convnext_v1.0.0',
        'version': '1.0.0',
        'architecture': 'convnext_tiny',
        'path': str(champion_dir / 'best_model.pth'),
        'config_path': str(champion_dir / 'class_mappings.json'),
        'deployed_at': datetime.now().isoformat(),
        'metrics': {
            'magnitude_accuracy': 98.36,
            'azimuth_accuracy': 50.66,
            'composite_score': 0.8234
        },
        'status': 'active'
    }
    save_registry(registry)
    print("✅ Registry updated")
    
    print("\n" + "=" * 60)
    print("✅ CHAMPION MODEL SETUP COMPLETE")
    print("=" * 60)
    print(f"\nChampion model ready at: {champion_dir}")
    print("\nNext steps:")
    print("1. Run: python scripts/check_status.py")
    print("2. Add events: python scripts/add_new_event.py add ...")
    
    return True


if __name__ == "__main__":
    success = setup_champion()
    sys.exit(0 if success else 1)
