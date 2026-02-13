#!/usr/bin/env python
"""
Check Pipeline Status

Display current status of the auto-update pipeline.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_registry


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime string for display."""
    if not iso_string:
        return "Never"
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_string


def check_status():
    """Display comprehensive pipeline status."""
    config = load_config()
    registry = load_registry()
    
    print("\n" + "=" * 70)
    print("🔄 EARTHQUAKE MODEL AUTO-UPDATE PIPELINE STATUS")
    print("=" * 70)
    
    # Champion Model Status
    print("\n📊 CURRENT CHAMPION MODEL")
    print("-" * 40)
    champion = registry.get('champion', {})
    
    if champion:
        print(f"   Model ID:     {champion.get('model_id', 'Unknown')}")
        print(f"   Version:      {champion.get('version', 'Unknown')}")
        print(f"   Architecture: {champion.get('architecture', 'Unknown')}")
        print(f"   Deployed:     {format_datetime(champion.get('deployed_at'))}")
        print(f"   Status:       {champion.get('status', 'Unknown')}")
        
        metrics = champion.get('metrics', {})
        if metrics:
            print(f"\n   Performance Metrics:")
            print(f"   - Magnitude Accuracy: {metrics.get('magnitude_accuracy', 0):.2f}%")
            print(f"   - Azimuth Accuracy:   {metrics.get('azimuth_accuracy', 0):.2f}%")
            print(f"   - Composite Score:    {metrics.get('composite_score', 0):.4f}")
    else:
        print("   No champion model registered")
    
    # Pending Events
    print("\n📥 PENDING EVENTS")
    print("-" * 40)
    pending = registry.get('pending_events', {})
    pending_count = pending.get('count', 0)
    print(f"   Count:      {pending_count}")
    print(f"   Last Added: {format_datetime(pending.get('last_added'))}")
    
    if pending.get('events'):
        print(f"\n   Recent Events:")
        for event in pending['events'][-3:]:
            print(f"   - {event.get('date')} | {event.get('station')} | {event.get('magnitude')}")
    
    # Validated Events
    print("\n✅ VALIDATED EVENTS")
    print("-" * 40)
    validated = registry.get('validated_events', [])
    print(f"   Count: {len(validated)}")
    
    if validated:
        print(f"\n   Recent Validated:")
        for event in validated[-3:]:
            print(f"   - {event.get('date')} | {event.get('station')} | {event.get('magnitude')}")
    
    # Trigger Status
    print("\n⚡ TRIGGER STATUS")
    print("-" * 40)
    triggers = config.get('triggers', {})
    min_events = triggers.get('min_new_events', 20)
    max_days = triggers.get('max_days_between_training', 90)
    
    print(f"   Min Events Required: {min_events}")
    print(f"   Current Validated:   {len(validated)}")
    print(f"   Progress:            {len(validated)}/{min_events} ({len(validated)/min_events*100:.1f}%)")
    
    # Days since last training
    if champion.get('deployed_at'):
        try:
            last_date = datetime.fromisoformat(champion['deployed_at'].replace('Z', '+00:00'))
            # Make both naive for comparison
            if last_date.tzinfo is not None:
                last_date = last_date.replace(tzinfo=None)
            days_since = (datetime.now() - last_date).days
            print(f"\n   Days Since Last Update: {days_since}")
            print(f"   Max Days Allowed:       {max_days}")
        except:
            print(f"\n   Days Since Last Update: Unknown")
    
    # Determine if ready
    ready = len(validated) >= min_events
    print(f"\n   🚦 Pipeline Ready: {'YES ✅' if ready else 'NO ⏳'}")
    
    # Pipeline History
    print("\n📈 PIPELINE HISTORY")
    print("-" * 40)
    history = registry.get('pipeline_history', {})
    print(f"   Total Runs:        {history.get('total_runs', 0)}")
    print(f"   Successful Updates: {history.get('successful_updates', 0)}")
    print(f"   Failed Updates:     {history.get('failed_updates', 0)}")
    print(f"   Last Run:          {format_datetime(history.get('last_run'))}")
    
    # Archived Models
    print("\n📦 ARCHIVED MODELS")
    print("-" * 40)
    archives = registry.get('archive', [])
    print(f"   Total Archived: {len(archives)}")
    
    if archives:
        print(f"\n   Recent Archives:")
        for arch in archives[-3:]:
            print(f"   - {arch.get('model_id')} | Archived: {format_datetime(arch.get('archived_at'))}")
    
    # Quick Commands
    print("\n" + "=" * 70)
    print("📋 QUICK COMMANDS")
    print("=" * 70)
    print("""
   Add new event:
   $ python scripts/add_new_event.py add -d 2026-02-10 -s GTO -m Large -a NE

   Validate pending events:
   $ python scripts/add_new_event.py validate

   Run pipeline (if ready):
   $ python scripts/run_pipeline.py

   Force run pipeline:
   $ python scripts/run_pipeline.py --force

   Rollback to previous model:
   $ python scripts/rollback_model.py
""")
    
    print("=" * 70)


if __name__ == "__main__":
    check_status()
