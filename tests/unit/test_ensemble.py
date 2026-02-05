"""
Unit tests for EnsembleDetector.

Tests:
- Basic detection
- Multi-detector combination
- Confidence handling
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.pii.ensemble_detector import EnsembleDetector


class TestEnsembleDetector:
    """Test cases for EnsembleDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return EnsembleDetector()

    def test_initialization(self, detector):
        """Test ensemble detector initialization."""
        assert detector is not None

    def test_basic_detection(self, detector):
        """Test basic PII detection."""
        text = "My PAN is ABCDE1234F and email is test@example.com"
        result = detector.detect(text)

        # Ensemble detector returns list directly
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_text(self, detector):
        """Test empty text input."""
        result = detector.detect("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_no_pii_text(self, detector):
        """Test text without PII."""
        text = "The weather is nice today."
        result = detector.detect(text)
        assert isinstance(result, list)

    def test_confidence_scores(self, detector):
        """Test confidence scores."""
        text = "PAN: ABCDE1234F"
        result = detector.detect(text)

        for entity in result:
            assert 'confidence' in entity
            assert 0.0 <= entity['confidence'] <= 1.0

    def test_entity_positions(self, detector):
        """Test entity positions."""
        text = "PAN: ABCDE1234F"
        result = detector.detect(text)

        for entity in result:
            assert 'start' in entity
            assert 'end' in entity
            assert entity['start'] >= 0
            assert entity['end'] <= len(text)

    def test_detection_metadata(self, detector):
        """Test detection metadata."""
        text = "PAN: ABCDE1234F"
        result = detector.detect(text)

        # Check result is list and has entities
        assert isinstance(result, list)
        if len(result) > 0:
            # Check first entity has required fields
            entity = result[0]
            assert 'entity_type' in entity
            assert 'text' in entity
            assert 'confidence' in entity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
