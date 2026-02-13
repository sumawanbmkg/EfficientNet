#!/usr/bin/env python
"""
Rollback Model

Script to rollback to a previous model version.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_registry
from src.deployer import ModelDeployer


def list_available_versions():
    """List all available versions for rollback."""
    registry = load_registry()
    archives = registry.get('archive', [])
    champion = registry.get('champion', {})
    
    print("\n" + "=" * 60)
    print("📦 AVAILABLE MODEL VERSIONS")
    print("=" * 60)
    
    # Current champion
    print("\n🏆 CURRENT CHAMPION:")
    if champion:
        print(f"   Version:  {champion.get('version', 'Unknown')}")
        print(f"   Model ID: {champion.get('model_id', 'Unknown')}")
        print(f"   Deployed: {champion.get('deployed_at', 'Unknown')}")
        metrics = champion.get('metrics', {})
        if metrics:
            print(f"   Mag Acc:  {metrics.get('magnitude_accuracy', 0):.2f}%")
            print(f"   Azi Acc:  {metrics.get('azimuth_accuracy', 0):.2f}%")
    else:
        print("   No champion model")
    
    # Archived versions
    print("\n📁 ARCHIVED VERSIONS:")
    if not archives:
        print("   No archived versions available")
    else:
        for i, arch in enumerate(reversed(archives), 1):
            print(f"\n   [{i}] {arch.get('model_id')}")
            print(f"       Archived: {arch.get('archived_at', 'Unknown')}")
            metrics = arch.get('metrics', {})
            if metrics:
                print(f"       Mag Acc:  {metrics.get('magnitude_accuracy', 0):.2f}%")
                print(f"       Azi Acc:  {metrics.get('azimuth_accuracy', 0):.2f}%")
    
    print("\n" + "=" * 60)


def rollback(version: str = None, confirm: bool = False):
    """
    Rollback to a previous version.
    
    Args:
        version: Version to rollback to (latest if None)
        confirm: Skip confirmation prompt
    """
    deployer = ModelDeployer()
    registry = load_registry()
    archives = registry.get('archive', [])
    
    if not archives:
        print("❌ No archived versions available for rollback")
        return False
    
    # Determine target version
    if version:
        target = next((a for a in archives if version in a['model_id']), None)
        if not target:
            print(f"❌ Version '{version}' not found in archives")
            print("\nAvailable versions:")
            for arch in archives:
                print(f"   - {arch['model_id']}")
            return False
    else:
        target = archives[-1]
    
    print("\n" + "=" * 60)
    print("⚠️  MODEL ROLLBACK")
    print("=" * 60)
    
    current = registry.get('champion', {})
    print(f"\nCurrent Champion: {current.get('model_id', 'Unknown')}")
    print(f"Rollback Target:  {target['model_id']}")
    
    # Show metrics comparison
    current_metrics = current.get('metrics', {})
    target_metrics = target.get('metrics', {})
    
    print("\n📊 Metrics Comparison:")
    print(f"                    Current    →    Target")
    print(f"   Magnitude Acc:   {current_metrics.get('magnitude_accuracy', 0):6.2f}%       {target_metrics.get('magnitude_accuracy', 0):6.2f}%")
    print(f"   Azimuth Acc:     {current_metrics.get('azimuth_accuracy', 0):6.2f}%       {target_metrics.get('azimuth_accuracy', 0):6.2f}%")
    
    # Confirmation
    if not confirm:
        print("\n⚠️  This will replace the current champion model!")
        response = input("\nProceed with rollback? [y/N]: ").strip().lower()
        if response != 'y':
            print("Rollback cancelled.")
            return False
    
    # Perform rollback
    print("\n🔄 Performing rollback...")
    result = deployer.rollback(version)
    
    if result["success"]:
        print(f"\n✅ {result['message']}")
        print(f"   Rolled back to: {result['rolled_back_to']}")
        return True
    else:
        print(f"\n❌ Rollback failed: {result['message']}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Rollback to a previous model version")
    
    parser.add_argument("--version", "-v", help="Version to rollback to (latest if not specified)")
    parser.add_argument("--list", "-l", action="store_true", help="List available versions")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_versions()
        return
    
    success = rollback(version=args.version, confirm=args.yes)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
