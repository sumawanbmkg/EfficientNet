"""
Data Validator Module

Validates new earthquake event data before adding to the dataset.
Supports both numeric values (magnitude, azimuth in degrees) and class labels.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np

from .utils import load_config

logger = logging.getLogger("autoupdate_pipeline.validator")


class DataValidator:
    """
    Validates earthquake event data for quality and consistency.
    
    Supports two input modes:
    1. Numeric mode: magnitude (float), azimuth (degrees 0-360)
    2. Class mode: magnitude_class (Large/Medium/Moderate/Normal), azimuth_class (N/NE/E/etc)
    
    Validation checks:
    1. Required fields present
    2. Valid magnitude value/class
    3. Valid azimuth value/class
    4. Valid station code
    5. Valid date format
    6. Spectrogram file exists and is valid (optional)
    7. No duplicate events
    """
    
    # Magnitude classification thresholds
    MAGNITUDE_THRESHOLDS = {
        'Large': 6.0,      # M >= 6.0
        'Medium': 5.0,     # 5.0 <= M < 6.0
        'Moderate': 4.0,   # 4.0 <= M < 5.0
        'Normal': 0.0      # M < 4.0 or no earthquake
    }
    
    # Azimuth to direction mapping (degrees)
    AZIMUTH_DIRECTIONS = {
        'N': (337.5, 22.5),
        'NE': (22.5, 67.5),
        'E': (67.5, 112.5),
        'SE': (112.5, 157.5),
        'S': (157.5, 202.5),
        'SW': (202.5, 247.5),
        'W': (247.5, 292.5),
        'NW': (292.5, 337.5),
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize validator with configuration."""
        self.config = config or load_config()
        self.validation_config = self.config.get('data_validation', {})
        
        # Load valid values
        self.required_fields = self.validation_config.get('required_fields', [])
        self.magnitude_classes = self.validation_config.get('magnitude_classes', [])
        self.azimuth_classes = self.validation_config.get('azimuth_classes', [])
        self.valid_stations = self.validation_config.get('valid_stations', [])
        
        logger.info("DataValidator initialized")
    
    @classmethod
    def magnitude_to_class(cls, magnitude: float) -> str:
        """
        Convert numeric magnitude to class label.
        
        Args:
            magnitude: Earthquake magnitude (e.g., 5.7)
            
        Returns:
            Class label: Large, Medium, Moderate, or Normal
        """
        if magnitude >= cls.MAGNITUDE_THRESHOLDS['Large']:
            return 'Large'
        elif magnitude >= cls.MAGNITUDE_THRESHOLDS['Medium']:
            return 'Medium'
        elif magnitude >= cls.MAGNITUDE_THRESHOLDS['Moderate']:
            return 'Moderate'
        else:
            return 'Normal'
    
    @classmethod
    def azimuth_to_class(cls, azimuth_degrees: float) -> str:
        """
        Convert azimuth in degrees to direction class.
        
        Args:
            azimuth_degrees: Azimuth in degrees (0-360)
            
        Returns:
            Direction class: N, NE, E, SE, S, SW, W, NW
        """
        # Normalize to 0-360
        azimuth = azimuth_degrees % 360
        
        # Special case for North (wraps around 360)
        if azimuth >= 337.5 or azimuth < 22.5:
            return 'N'
        
        for direction, (start, end) in cls.AZIMUTH_DIRECTIONS.items():
            if direction == 'N':
                continue  # Already handled
            if start <= azimuth < end:
                return direction
        
        return 'N'  # Default fallback
    
    def validate_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single earthquake event.
        
        Accepts either:
        - Numeric: magnitude (float), azimuth (float degrees)
        - Class: magnitude_class (str), azimuth_class (str)
        
        Args:
            event_data: Dictionary containing event information
            
        Returns:
            Dictionary with is_valid, errors, warnings, and converted_data
        """
        errors = []
        warnings = []
        converted_data = event_data.copy()
        
        # Check required fields
        required = ['date', 'station']
        for field in required:
            if field not in event_data or event_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check magnitude (numeric or class)
        has_mag_numeric = 'magnitude' in event_data and event_data['magnitude'] is not None
        has_mag_class = 'magnitude_class' in event_data and event_data['magnitude_class'] is not None
        
        if not has_mag_numeric and not has_mag_class:
            errors.append("Missing magnitude: provide either 'magnitude' (numeric) or 'magnitude_class'")
        
        # Check azimuth (numeric or class)
        has_azi_numeric = 'azimuth' in event_data and event_data['azimuth'] is not None
        has_azi_class = 'azimuth_class' in event_data and event_data['azimuth_class'] is not None
        
        if not has_azi_numeric and not has_azi_class:
            errors.append("Missing azimuth: provide either 'azimuth' (degrees) or 'azimuth_class'")
        
        if errors:
            return {"is_valid": False, "errors": errors, "warnings": warnings, "converted_data": None}
        
        # Validate date format
        try:
            date_str = event_data.get('date')
            if isinstance(date_str, str):
                datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            errors.append(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
        
        # Validate and convert magnitude
        if has_mag_numeric:
            try:
                mag_value = float(event_data['magnitude'])
                if mag_value < 0 or mag_value > 10:
                    errors.append(f"Invalid magnitude value: {mag_value}. Expected 0-10")
                else:
                    converted_data['magnitude_value'] = mag_value
                    converted_data['magnitude_class'] = self.magnitude_to_class(mag_value)
                    logger.info(f"Converted magnitude {mag_value} -> {converted_data['magnitude_class']}")
            except (ValueError, TypeError):
                errors.append(f"Invalid magnitude format: {event_data['magnitude']}. Expected numeric value")
        elif has_mag_class:
            mag_class = event_data['magnitude_class']
            if mag_class not in self.magnitude_classes:
                errors.append(f"Invalid magnitude class: {mag_class}. Valid: {self.magnitude_classes}")
            else:
                converted_data['magnitude_class'] = mag_class
        
        # Validate and convert azimuth
        if has_azi_numeric:
            try:
                azi_value = float(event_data['azimuth'])
                if azi_value < 0 or azi_value > 360:
                    warnings.append(f"Azimuth {azi_value} normalized to 0-360 range")
                    azi_value = azi_value % 360
                converted_data['azimuth_value'] = azi_value
                converted_data['azimuth_class'] = self.azimuth_to_class(azi_value)
                logger.info(f"Converted azimuth {azi_value}° -> {converted_data['azimuth_class']}")
            except (ValueError, TypeError):
                errors.append(f"Invalid azimuth format: {event_data['azimuth']}. Expected numeric value (degrees)")
        elif has_azi_class:
            azi_class = event_data['azimuth_class']
            if azi_class not in self.azimuth_classes:
                errors.append(f"Invalid azimuth class: {azi_class}. Valid: {self.azimuth_classes}")
            else:
                converted_data['azimuth_class'] = azi_class
        
        # Validate station
        station = event_data.get('station')
        if station not in self.valid_stations:
            errors.append(f"Invalid station: {station}. Valid: {self.valid_stations}")
        
        # Validate spectrogram file (optional)
        spec_path = event_data.get('spectrogram_path')
        if spec_path:
            spec_valid, spec_error = self._validate_spectrogram(spec_path)
            if not spec_valid:
                warnings.append(spec_error)  # Warning, not error
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Event validated successfully: {event_data.get('date')} - {event_data.get('station')}")
        else:
            logger.warning(f"Event validation failed: {errors}")
        
        return {
            "is_valid": is_valid, 
            "errors": errors, 
            "warnings": warnings,
            "converted_data": converted_data if is_valid else None
        }
    
    def _validate_spectrogram(self, spec_path: str) -> Tuple[bool, Optional[str]]:
        """Validate spectrogram image file."""
        path = Path(spec_path)
        
        if not path.exists():
            return False, f"Spectrogram file not found: {spec_path}"
        
        try:
            img = Image.open(path)
            
            # Check image size (should be 224x224 for our models)
            if img.size != (224, 224):
                return False, f"Invalid spectrogram size: {img.size}. Expected (224, 224)"
            
            # Check image mode (should be RGB)
            if img.mode != 'RGB':
                return False, f"Invalid image mode: {img.mode}. Expected RGB"
            
            # Check if image is not corrupted (can be loaded as array)
            arr = np.array(img)
            if arr.shape != (224, 224, 3):
                return False, f"Invalid array shape: {arr.shape}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error reading spectrogram: {str(e)}"
    
    def check_duplicate(self, event_data: Dict[str, Any], existing_events: List[Dict]) -> bool:
        """
        Check if event already exists in the dataset.
        
        Args:
            event_data: New event to check
            existing_events: List of existing events
            
        Returns:
            True if duplicate found, False otherwise
        """
        new_key = (
            event_data.get('date'),
            event_data.get('station'),
        )
        
        for existing in existing_events:
            existing_key = (
                existing.get('date'),
                existing.get('station'),
            )
            
            if new_key == existing_key:
                logger.warning(f"Duplicate event found: {new_key}")
                return True
        
        return False
    
    def validate_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total": len(events),
            "valid": 0,
            "invalid": 0,
            "valid_events": [],
            "invalid_events": []
        }
        
        for event in events:
            validation = self.validate_event(event)
            
            if validation["is_valid"]:
                results["valid"] += 1
                results["valid_events"].append(validation["converted_data"])
            else:
                results["invalid"] += 1
                results["invalid_events"].append({
                    "event": event,
                    "errors": validation["errors"]
                })
        
        logger.info(f"Batch validation complete: {results['valid']}/{results['total']} valid")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Total Events: {results['total']}",
            f"Valid Events: {results['valid']}",
            f"Invalid Events: {results['invalid']}",
            f"Success Rate: {results['valid']/results['total']*100:.1f}%",
            ""
        ]
        
        if results['invalid_events']:
            lines.append("INVALID EVENTS:")
            lines.append("-" * 40)
            for item in results['invalid_events']:
                event = item['event']
                errors = item['errors']
                lines.append(f"  Event: {event.get('date')} - {event.get('station')}")
                for error in errors:
                    lines.append(f"    - {error}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
