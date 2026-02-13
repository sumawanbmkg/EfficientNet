"""
Data Ingestion Module

Handles the ingestion of new earthquake event data into the pipeline.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from .utils import load_config, load_registry, save_registry, log_pipeline_event
from .data_validator import DataValidator

logger = logging.getLogger("autoupdate_pipeline.ingestion")


class DataIngestion:
    """
    Handles ingestion of new earthquake events.
    
    Workflow:
    1. Receive new event data
    2. Validate the data
    3. Copy spectrogram to pending folder
    4. Update registry with pending event
    5. Check if trigger conditions are met
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data ingestion module."""
        self.config = config or load_config()
        self.validator = DataValidator(self.config)
        
        # Setup paths
        self.base_path = Path(__file__).parent.parent
        self.pending_path = self.base_path / self.config['paths']['pending_data']
        self.validated_path = self.base_path / self.config['paths']['validated_data']
        
        # Create directories
        self.pending_path.mkdir(parents=True, exist_ok=True)
        self.validated_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataIngestion initialized")
    
    def add_pending_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for add_event - adds event to pending queue."""
        return self.add_event(event_data, event_data.get('spectrogram_path'))
    
    def validate_pending_events(self) -> Dict[str, Any]:
        """Validate and move all pending events to validated."""
        events = self.move_to_validated()
        return {
            "validated_count": len(events),
            "failed_count": 0,
            "events": events
        }
    
    def add_event(self, event_data: Dict[str, Any], spectrogram_source: str = None) -> Dict[str, Any]:
        """
        Add a new earthquake event to the pipeline.
        
        Args:
            event_data: Dictionary containing event information
                - date: Event date (YYYY-MM-DD)
                - station: Station code
                - magnitude: Magnitude class (Large, Medium, Moderate, Normal)
                - azimuth: Azimuth class (N, NE, E, SE, S, SW, W, NW, Normal)
            spectrogram_source: Path to source spectrogram image
            
        Returns:
            Dictionary with ingestion result
        """
        result = {
            "success": False,
            "event_id": None,
            "message": "",
            "trigger_check": None
        }
        
        # Generate event ID
        event_id = f"{event_data['station']}_{event_data['date'].replace('-', '')}"
        event_data['event_id'] = event_id
        
        # Set spectrogram path if source provided
        if spectrogram_source:
            dest_filename = f"{event_id}_spec.png"
            dest_path = self.pending_path / dest_filename
            event_data['spectrogram_path'] = str(dest_path)
        
        # Validate event
        validation = self.validator.validate_event(event_data)
        
        if not validation["is_valid"]:
            result["message"] = f"Validation failed: {validation['errors']}"
            logger.error(result["message"])
            return result
        
        # Load registry to check for duplicates
        registry = load_registry()
        existing_events = registry.get('pending_events', {}).get('events', [])
        
        if self.validator.check_duplicate(event_data, existing_events):
            result["message"] = "Duplicate event detected"
            logger.warning(result["message"])
            return result
        
        # Copy spectrogram to pending folder
        if spectrogram_source and Path(spectrogram_source).exists():
            shutil.copy2(spectrogram_source, dest_path)
            logger.info(f"Spectrogram copied to: {dest_path}")
        
        # Update registry - store both numeric values and classes
        pending = registry.get('pending_events', {'count': 0, 'events': []})
        pending['count'] = pending.get('count', 0) + 1
        pending['last_added'] = datetime.now().isoformat()
        
        # Build event record with all available data
        event_record = {
            'event_id': event_id,
            'date': event_data['date'],
            'station': event_data['station'],
            'spectrogram_path': event_data.get('spectrogram_path'),
            'added_at': datetime.now().isoformat()
        }
        
        # Store numeric values if available
        if 'magnitude_value' in event_data:
            event_record['magnitude_value'] = event_data['magnitude_value']
        if 'azimuth_value' in event_data:
            event_record['azimuth_value'] = event_data['azimuth_value']
        
        # Store class labels
        event_record['magnitude_class'] = event_data.get('magnitude_class', event_data.get('magnitude'))
        event_record['azimuth_class'] = event_data.get('azimuth_class', event_data.get('azimuth'))
        
        # Keep original values for backward compatibility
        event_record['magnitude'] = event_data.get('magnitude_class', event_data.get('magnitude'))
        event_record['azimuth'] = event_data.get('azimuth_class', event_data.get('azimuth'))
        
        pending['events'].append(event_record)
        registry['pending_events'] = pending
        
        save_registry(registry)
        
        # Log event
        log_pipeline_event("event_added", {
            "event_id": event_id,
            "date": event_data['date'],
            "station": event_data['station'],
            "magnitude": event_data['magnitude'],
            "azimuth": event_data['azimuth']
        })
        
        # Check trigger conditions
        from .utils import check_trigger_conditions
        trigger_check = check_trigger_conditions(registry, self.config)
        
        result["success"] = True
        result["event_id"] = event_id
        result["message"] = f"Event added successfully. Pending count: {pending['count']}"
        result["trigger_check"] = trigger_check
        
        logger.info(result["message"])
        
        if trigger_check["should_trigger"]:
            logger.info(f"🚀 Trigger conditions met! Reasons: {trigger_check['reasons']}")
        
        return result
    
    def add_events_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple events in batch.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Batch ingestion results
        """
        results = {
            "total": len(events),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for event in events:
            result = self.add_event(event)
            results["results"].append(result)
            
            if result["success"]:
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        logger.info(f"Batch ingestion complete: {results['successful']}/{results['total']} successful")
        
        return results
    
    def get_pending_events(self) -> List[Dict[str, Any]]:
        """Get list of pending events."""
        registry = load_registry()
        return registry.get('pending_events', {}).get('events', [])
    
    def get_pending_count(self) -> int:
        """Get count of pending events."""
        registry = load_registry()
        return registry.get('pending_events', {}).get('count', 0)
    
    def move_to_validated(self, event_ids: List[str] = None):
        """
        Move pending events to validated folder.
        
        Args:
            event_ids: List of event IDs to move. If None, move all.
        """
        registry = load_registry()
        pending = registry.get('pending_events', {'count': 0, 'events': []})
        
        events_to_move = []
        events_to_keep = []
        
        for event in pending.get('events', []):
            if event_ids is None or event['event_id'] in event_ids:
                events_to_move.append(event)
            else:
                events_to_keep.append(event)
        
        # Move spectrogram files (if they exist)
        for event in events_to_move:
            spec_path = event.get('spectrogram_path')
            if spec_path:
                src_path = Path(spec_path)
                if src_path.exists():
                    dest_path = self.validated_path / src_path.name
                    shutil.move(str(src_path), str(dest_path))
                    event['spectrogram_path'] = str(dest_path)
            event['validated_at'] = datetime.now().isoformat()
        
        # Update registry
        pending['events'] = events_to_keep
        pending['count'] = len(events_to_keep)
        registry['pending_events'] = pending
        
        # Add to validated list (could be stored separately)
        if 'validated_events' not in registry:
            registry['validated_events'] = []
        registry['validated_events'].extend(events_to_move)
        
        save_registry(registry)
        
        logger.info(f"Moved {len(events_to_move)} events to validated")
        
        return events_to_move
    
    def clear_pending(self):
        """Clear all pending events (use with caution)."""
        registry = load_registry()
        
        # Remove spectrogram files
        pending = registry.get('pending_events', {}).get('events', [])
        for event in pending:
            spec_path = Path(event.get('spectrogram_path', ''))
            if spec_path.exists():
                spec_path.unlink()
        
        # Clear registry
        registry['pending_events'] = {'count': 0, 'events': [], 'last_added': None}
        save_registry(registry)
        
        logger.warning("All pending events cleared")
    
    def generate_status_report(self) -> str:
        """Generate a status report of pending events."""
        registry = load_registry()
        pending = registry.get('pending_events', {})
        
        lines = [
            "=" * 60,
            "DATA INGESTION STATUS REPORT",
            "=" * 60,
            f"Pending Events: {pending.get('count', 0)}",
            f"Last Added: {pending.get('last_added', 'Never')}",
            ""
        ]
        
        events = pending.get('events', [])
        if events:
            lines.append("PENDING EVENTS:")
            lines.append("-" * 40)
            
            # Group by magnitude
            by_magnitude = {}
            for event in events:
                mag = event.get('magnitude', 'Unknown')
                if mag not in by_magnitude:
                    by_magnitude[mag] = []
                by_magnitude[mag].append(event)
            
            for mag, mag_events in sorted(by_magnitude.items()):
                lines.append(f"\n  {mag}: {len(mag_events)} events")
                for event in mag_events[:5]:  # Show first 5
                    lines.append(f"    - {event['date']} | {event['station']} | {event['azimuth']}")
                if len(mag_events) > 5:
                    lines.append(f"    ... and {len(mag_events) - 5} more")
        
        # Trigger status
        from .utils import check_trigger_conditions, load_config
        config = load_config()
        trigger = check_trigger_conditions(registry, config)
        
        lines.append("")
        lines.append("TRIGGER STATUS:")
        lines.append("-" * 40)
        lines.append(f"  Should Trigger: {'YES ✅' if trigger['should_trigger'] else 'NO ❌'}")
        lines.append(f"  Pending: {trigger['pending_count']} / {trigger['min_required']} required")
        
        if trigger.get('days_since_last_run'):
            lines.append(f"  Days Since Last Run: {trigger['days_since_last_run']}")
        
        if trigger['reasons']:
            lines.append(f"  Reasons: {', '.join(trigger['reasons'])}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
