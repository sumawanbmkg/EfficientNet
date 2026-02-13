#!/usr/bin/env python
"""
Add New Earthquake Event

Script to add new validated earthquake events to the pipeline.
Supports numeric input for magnitude and azimuth.
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_registry, save_registry, log_pipeline_event
from src.data_validator import DataValidator
from src.data_ingestion import DataIngestion


def add_event(date: str, station: str, magnitude: float, azimuth: float,
              spectrogram_path: str = None, notes: str = None) -> dict:
    """
    Add a new earthquake event with numeric values.
    
    Args:
        date: Event date (YYYY-MM-DD)
        station: Station code (e.g., GTO, SCN)
        magnitude: Earthquake magnitude (e.g., 5.7)
        azimuth: Azimuth in degrees (0-360)
        spectrogram_path: Path to spectrogram image (optional)
        notes: Optional notes
        
    Returns:
        Result dictionary
    """
    config = load_config()
    validator = DataValidator(config)
    ingestion = DataIngestion(config)
    
    # Create event data with numeric values
    event_data = {
        "date": date,
        "station": station,
        "magnitude": magnitude,  # Numeric value
        "azimuth": azimuth,      # Degrees
        "spectrogram_path": spectrogram_path,
        "notes": notes,
        "added_at": datetime.now().isoformat(),
        "added_by": "manual"
    }
    
    print(f"\n{'='*50}")
    print("ADDING NEW EARTHQUAKE EVENT")
    print(f"{'='*50}")
    print(f"Date: {date}")
    print(f"Station: {station}")
    print(f"Magnitude: {magnitude}")
    print(f"Azimuth: {azimuth}°")
    
    # Validate event (will convert to classes)
    print("\n[1/3] Validating event data...")
    validation = validator.validate_event(event_data)
    
    if not validation["is_valid"]:
        print(f"\nEvent validation failed:")
        for error in validation["errors"]:
            print(f"   - {error}")
        return {"success": False, "message": "Validation failed", "errors": validation["errors"]}
    
    # Show conversion results
    converted = validation["converted_data"]
    print(f"✅ Validation passed")
    print(f"   Magnitude {magnitude} → {converted['magnitude_class']}")
    print(f"   Azimuth {azimuth}° → {converted['azimuth_class']}")
    
    if validation["warnings"]:
        print(f"\n⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")
    
    # Add to pending (use converted data)
    print("\n[2/3] Adding to pending events...")
    result = ingestion.add_pending_event(converted)
    
    if not result["success"]:
        print(f"❌ Failed to add: {result['message']}")
        return result
    
    print(f"✅ Added to pending (ID: {result['event_id']})")
    
    # Check status
    print("\n[3/3] Checking validation status...")
    registry = load_registry()
    pending = registry.get('pending_events', {})
    
    print(f"   Pending events: {pending.get('count', 0)}")
    print(f"   Validated events: {len(registry.get('validated_events', []))}")
    
    # Log event
    log_pipeline_event("event_added", {
        "date": date,
        "station": station,
        "magnitude": magnitude,
        "magnitude_class": converted['magnitude_class'],
        "azimuth": azimuth,
        "azimuth_class": converted['azimuth_class']
    })
    
    print(f"\n{'='*50}")
    print("✅ EVENT ADDED SUCCESSFULLY")
    print(f"{'='*50}")
    
    return {
        "success": True,
        "message": "Event added successfully",
        "event_id": result["event_id"],
        "pending_count": pending.get('count', 0),
        "converted": {
            "magnitude_class": converted['magnitude_class'],
            "azimuth_class": converted['azimuth_class']
        }
    }


def validate_pending_events() -> dict:
    """Validate all pending events and move to validated."""
    config = load_config()
    ingestion = DataIngestion(config)
    
    print("\n[VALIDATING PENDING EVENTS]")
    result = ingestion.validate_pending_events()
    
    print(f"Validated: {result['validated_count']} events")
    print(f"Failed: {result['failed_count']} events")
    
    return result


def list_pending_events():
    """List all pending events."""
    registry = load_registry()
    pending = registry.get('pending_events', {})
    events = pending.get('events', [])
    
    print(f"\n{'='*60}")
    print(f"PENDING EVENTS ({len(events)} total)")
    print(f"{'='*60}")
    
    if not events:
        print("No pending events.")
        return
    
    for i, event in enumerate(events, 1):
        mag_val = event.get('magnitude_value', event.get('magnitude', '-'))
        azi_val = event.get('azimuth_value', event.get('azimuth', '-'))
        mag_class = event.get('magnitude_class', '-')
        azi_class = event.get('azimuth_class', '-')
        
        print(f"\n[{i}] ID: {event.get('event_id', 'N/A')}")
        print(f"    Date: {event.get('date')} | Station: {event.get('station')}")
        print(f"    Magnitude: {mag_val} ({mag_class})")
        print(f"    Azimuth: {azi_val}° ({azi_class})")
        print(f"    Added: {event.get('added_at', 'Unknown')}")


def list_validated_events():
    """List all validated events."""
    registry = load_registry()
    events = registry.get('validated_events', [])
    
    print(f"\n{'='*60}")
    print(f"VALIDATED EVENTS ({len(events)} total)")
    print(f"{'='*60}")
    
    if not events:
        print("No validated events.")
        return
    
    for i, event in enumerate(events, 1):
        mag_val = event.get('magnitude_value', event.get('magnitude', '-'))
        azi_val = event.get('azimuth_value', event.get('azimuth', '-'))
        mag_class = event.get('magnitude_class', '-')
        azi_class = event.get('azimuth_class', '-')
        
        print(f"\n[{i}] ID: {event.get('event_id', 'N/A')}")
        print(f"    Date: {event.get('date')} | Station: {event.get('station')}")
        print(f"    Magnitude: {mag_val} ({mag_class})")
        print(f"    Azimuth: {azi_val}° ({azi_class})")
        print(f"    Validated: {event.get('validated_at', 'Unknown')}")


def delete_pending_event(event_id: str) -> bool:
    """Delete a pending event by ID."""
    registry = load_registry()
    pending = registry.get('pending_events', {'count': 0, 'events': []})
    events = pending.get('events', [])
    
    # Find and remove event
    found = False
    new_events = []
    for event in events:
        if event.get('event_id') == event_id:
            found = True
            print(f"✅ Deleted event: {event_id}")
            print(f"   Date: {event.get('date')}")
            print(f"   Station: {event.get('station')}")
        else:
            new_events.append(event)
    
    if not found:
        print(f"❌ Event not found: {event_id}")
        print("\nAvailable pending events:")
        for event in events:
            print(f"   - {event.get('event_id')}")
        return False
    
    # Update registry
    pending['events'] = new_events
    pending['count'] = len(new_events)
    registry['pending_events'] = pending
    save_registry(registry)
    
    return True


def clear_all_pending():
    """Clear all pending events."""
    registry = load_registry()
    pending = registry.get('pending_events', {'count': 0, 'events': []})
    count = pending.get('count', 0)
    
    pending['events'] = []
    pending['count'] = 0
    registry['pending_events'] = pending
    save_registry(registry)
    
    print(f"✅ Cleared {count} pending events")


def delete_validated_event(event_id: str) -> bool:
    """Delete a validated event by ID."""
    registry = load_registry()
    events = registry.get('validated_events', [])
    
    # Find and remove event
    found = False
    new_events = []
    for event in events:
        if event.get('event_id') == event_id:
            found = True
            print(f"✅ Deleted validated event: {event_id}")
            print(f"   Date: {event.get('date')}")
            print(f"   Station: {event.get('station')}")
        else:
            new_events.append(event)
    
    if not found:
        print(f"❌ Event not found: {event_id}")
        print("\nAvailable validated events:")
        for event in events:
            print(f"   - {event.get('event_id')}")
        return False
    
    # Update registry
    registry['validated_events'] = new_events
    save_registry(registry)
    
    return True


def clear_all_validated():
    """Clear all validated events."""
    registry = load_registry()
    events = registry.get('validated_events', [])
    count = len(events)
    
    registry['validated_events'] = []
    save_registry(registry)
    
    print(f"✅ Cleared {count} validated events")


def show_classification_guide():
    """Show magnitude and azimuth classification guide."""
    print(f"\n{'='*60}")
    print("CLASSIFICATION GUIDE")
    print(f"{'='*60}")
    
    print("\n📊 MAGNITUDE CLASSIFICATION:")
    print("   Large    : M >= 6.0")
    print("   Medium   : 5.0 <= M < 6.0")
    print("   Moderate : 4.0 <= M < 5.0")
    print("   Normal   : M < 4.0 (or no earthquake)")
    
    print("\n🧭 AZIMUTH CLASSIFICATION (degrees):")
    print("   N  : 337.5° - 22.5°  (North)")
    print("   NE : 22.5° - 67.5°   (Northeast)")
    print("   E  : 67.5° - 112.5°  (East)")
    print("   SE : 112.5° - 157.5° (Southeast)")
    print("   S  : 157.5° - 202.5° (South)")
    print("   SW : 202.5° - 247.5° (Southwest)")
    print("   W  : 247.5° - 292.5° (West)")
    print("   NW : 292.5° - 337.5° (Northwest)")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Add new earthquake event to pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add event with numeric values
  python add_new_event.py add -d 2026-02-10 -s GTO -m 5.7 -a 45
  
  # Add event with Normal class (no earthquake)
  python add_new_event.py add -d 2026-02-10 -s SCN -m 0 -a 0
  
  # List all events
  python add_new_event.py list
  
  # Show classification guide
  python add_new_event.py guide
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add event command
    add_parser = subparsers.add_parser("add", help="Add new event")
    add_parser.add_argument("--date", "-d", required=True, help="Event date (YYYY-MM-DD)")
    add_parser.add_argument("--station", "-s", required=True, help="Station code (e.g., GTO, SCN)")
    add_parser.add_argument("--magnitude", "-m", required=True, type=float,
                          help="Earthquake magnitude (e.g., 5.7). Use 0 for Normal/no earthquake")
    add_parser.add_argument("--azimuth", "-a", required=True, type=float,
                          help="Azimuth in degrees (0-360). Use 0 for Normal/no earthquake")
    add_parser.add_argument("--spectrogram", help="Path to spectrogram image (optional)")
    add_parser.add_argument("--notes", help="Optional notes")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate pending events")
    
    # List commands
    list_parser = subparsers.add_parser("list", help="List events")
    list_parser.add_argument("--type", "-t", choices=["pending", "validated", "all"],
                           default="all", help="Type of events to list")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete event")
    delete_parser.add_argument("--id", "-i", required=True, help="Event ID to delete (e.g., GTO_20260210)")
    delete_parser.add_argument("--validated", "-v", action="store_true", help="Delete from validated list")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all events")
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm clearing")
    clear_parser.add_argument("--validated", "-v", action="store_true", help="Clear validated events")
    
    # Guide command
    guide_parser = subparsers.add_parser("guide", help="Show classification guide")
    
    args = parser.parse_args()
    
    if args.command == "add":
        result = add_event(
            date=args.date,
            station=args.station,
            magnitude=args.magnitude,
            azimuth=args.azimuth,
            spectrogram_path=args.spectrogram,
            notes=args.notes
        )
        sys.exit(0 if result["success"] else 1)
        
    elif args.command == "validate":
        result = validate_pending_events()
        sys.exit(0)
        
    elif args.command == "list":
        if args.type in ["pending", "all"]:
            list_pending_events()
        if args.type in ["validated", "all"]:
            list_validated_events()
    
    elif args.command == "delete":
        if args.validated:
            result = delete_validated_event(args.id)
        else:
            result = delete_pending_event(args.id)
        sys.exit(0 if result else 1)
    
    elif args.command == "clear":
        if args.validated:
            if args.confirm:
                clear_all_validated()
            else:
                print("⚠️  Use --confirm to clear all validated events")
                sys.exit(1)
        else:
            if args.confirm:
                clear_all_pending()
            else:
                print("⚠️  Use --confirm to clear all pending events")
                sys.exit(1)
    
    elif args.command == "guide":
        show_classification_guide()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
