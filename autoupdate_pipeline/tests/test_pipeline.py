#!/usr/bin/env python
"""
Unit Tests for Auto-Update Pipeline

Run with: python -m pytest tests/test_pipeline.py -v
"""

import sys
import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, increment_version, generate_model_id
from src.data_validator import DataValidator
from src.data_ingestion import DataIngestion


class TestUtils:
    """Test utility functions."""
    
    def test_increment_version_patch(self):
        """Test patch version increment."""
        assert increment_version("1.0.0") == "1.0.1"
        assert increment_version("1.0.9") == "1.0.10"
    
    def test_increment_version_minor(self):
        """Test minor version increment."""
        assert increment_version("1.0.0", "minor") == "1.1.0"
    
    def test_increment_version_major(self):
        """Test major version increment."""
        assert increment_version("1.0.0", "major") == "2.0.0"
    
    def test_generate_model_id(self):
        """Test model ID generation."""
        model_id = generate_model_id("convnext")
        assert model_id.startswith("convnext_")
        assert len(model_id) > 10


class TestDataValidator:
    """Test data validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        config = load_config()
        return DataValidator(config)
    
    def test_valid_event(self, validator):
        """Test validation of valid event."""
        event = {
            "date": "2026-02-10",
            "station": "GTO",
            "magnitude": "Large",
            "azimuth": "NE"
        }
        result = validator.validate_event(event)
        assert result["is_valid"] == True
        assert len(result["errors"]) == 0

    
    def test_invalid_station(self, validator):
        """Test validation with invalid station."""
        event = {
            "date": "2026-02-10",
            "station": "INVALID",
            "magnitude": "Large",
            "azimuth": "NE"
        }
        result = validator.validate_event(event)
        assert result["is_valid"] == False
        assert any("station" in e.lower() for e in result["errors"])
    
    def test_invalid_magnitude(self, validator):
        """Test validation with invalid magnitude."""
        event = {
            "date": "2026-02-10",
            "station": "GTO",
            "magnitude": "Invalid",
            "azimuth": "NE"
        }
        result = validator.validate_event(event)
        assert result["is_valid"] == False
    
    def test_invalid_date_format(self, validator):
        """Test validation with invalid date format."""
        event = {
            "date": "10-02-2026",  # Wrong format
            "station": "GTO",
            "magnitude": "Large",
            "azimuth": "NE"
        }
        result = validator.validate_event(event)
        assert result["is_valid"] == False
    
    def test_missing_required_field(self, validator):
        """Test validation with missing field."""
        event = {
            "date": "2026-02-10",
            "station": "GTO",
            # Missing magnitude and azimuth
        }
        result = validator.validate_event(event)
        assert result["is_valid"] == False


class TestDataIngestion:
    """Test data ingestion."""
    
    @pytest.fixture
    def ingestion(self):
        """Create ingestion instance."""
        config = load_config()
        return DataIngestion(config)
    
    @patch('src.data_ingestion.save_registry')
    @patch('src.data_ingestion.load_registry')
    def test_add_pending_event(self, mock_load, mock_save, ingestion):
        """Test adding pending event."""
        mock_load.return_value = {
            "pending_events": {"count": 0, "events": []}
        }
        
        event = {
            "date": "2026-02-10",
            "station": "GTO",
            "magnitude": "Large",
            "azimuth": "NE"
        }
        
        result = ingestion.add_pending_event(event)
        assert result["success"] == True
        assert "event_id" in result


class TestModelComparison:
    """Test model comparison logic."""
    
    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        weights = {
            "magnitude_accuracy": 0.40,
            "azimuth_accuracy": 0.20,
            "loeo_validation": 0.30,
            "false_positive_rate": 0.10
        }
        
        metrics = {
            "magnitude_accuracy": 98.0,
            "azimuth_accuracy": 50.0,
            "loeo_validation": 95.0,
            "false_positive_rate": 5.0  # Lower is better
        }
        
        # Normalize and calculate
        score = (
            weights["magnitude_accuracy"] * (metrics["magnitude_accuracy"] / 100) +
            weights["azimuth_accuracy"] * (metrics["azimuth_accuracy"] / 100) +
            weights["loeo_validation"] * (metrics["loeo_validation"] / 100) +
            weights["false_positive_rate"] * (1 - metrics["false_positive_rate"] / 100)
        )
        
        assert 0 <= score <= 1
        assert score > 0.8  # Should be high with these metrics


class TestVersioning:
    """Test version management."""
    
    def test_version_comparison(self):
        """Test version string comparison."""
        from packaging import version
        
        v1 = version.parse("1.0.0")
        v2 = version.parse("1.0.1")
        v3 = version.parse("1.1.0")
        
        assert v1 < v2 < v3
    
    def test_version_increment_sequence(self):
        """Test sequential version increments."""
        v = "1.0.0"
        v = increment_version(v)  # 1.0.1
        v = increment_version(v)  # 1.0.2
        v = increment_version(v, "minor")  # 1.1.0
        v = increment_version(v, "major")  # 2.0.0
        
        assert v == "2.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
