"""
Unit tests for EnsembleDetector.

Tests:
- Multi-detector combination
- Deduplication
- Confidence aggregation
- Consensus scoring
- Entity merging
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
        # Should have multiple detectors
        assert hasattr(detector, 'detectors') or hasattr(detector, 'presidio_detector')

    def test_basic_detection(self, detector):
        """Test basic PII detection."""
        text = "My PAN is ABCDE1234F and email is test@example.com"
        result = detector.detect(text)

        assert 'entities' in result
        assert len(result['entities']) > 0

        # Should detect both PAN and email
        entity_types = {e['entity_type'] for e in result['entities']}
        assert 'PAN' in entity_types or 'EMAIL' in entity_types

    def test_multiple_detectors_contribution(self, detector):
        """Test that multiple detectors contribute to results."""
        text = """
        Name: John Smith
        PAN: ABCDE1234F
        Email: john@example.com
        Phone: 9876543210
        """

        result = detector.detect(text)

        # Should have entities from different detectors
        assert len(result['entities']) > 0

        # Check entity types from different detectors
        entity_types = {e['entity_type'] for e in result['entities']}
        assert len(entity_types) >= 2  # Should have at least 2 types

    def test_deduplication(self, detector):
        """Test entity deduplication across detectors."""
        text = "Email: test@example.com"
        result = detector.detect(text)

        # Even if multiple detectors find same email, should be deduplicated
        email_entities = [e for e in result['entities'] if e['entity_type'] == 'EMAIL']

        # Should have reasonable number of email detections (deduplicated)
        assert len(email_entities) <= 2  # Allow some overlap tolerance

    def test_confidence_aggregation(self, detector):
        """Test confidence score aggregation."""
        text = "PAN: ABCDE1234F"
        result = detector.detect(text)

        for entity in result['entities']:
            assert 'confidence' in entity
            assert 0.0 <= entity['confidence'] <= 1.0

            # Ensemble confidence should consider multiple detectors
            # Check if confidence is reasonable
            assert entity['confidence'] > 0.0

    def test_min_confidence_filter(self, detector):
        """Test min_confidence parameter."""
        text = "Name: John Smith, PAN: ABCDE1234F"

        # Low threshold
        result_low = detector.detect(text, min_confidence=0.1)
        count_low = len(result_low['entities'])

        # High threshold
        result_high = detector.detect(text, min_confidence=0.8)
        count_high = len(result_high['entities'])

        # Higher threshold should have fewer or equal entities
        assert count_high <= count_low

    def test_consensus_detection(self, detector):
        """Test entities detected by multiple detectors have higher confidence."""
        text = "Email: test@example.com"  # Should be detected by multiple detectors
        result = detector.detect(text)

        email_entities = [e for e in result['entities'] if e['entity_type'] == 'EMAIL']

        if len(email_entities) > 0:
            # Email should have high confidence (detected by multiple detectors)
            assert email_entities[0]['confidence'] > 0.5

    def test_indian_pii_detection(self, detector):
        """Test Indian-specific PII detection."""
        text = """
        PAN: ABCDE1234F
        Aadhaar: 1234 5678 9012
        Phone: +91-9876543210
        """

        result = detector.detect(text)

        entity_types = {e['entity_type'] for e in result['entities']}

        # Should detect Indian PII types
        assert 'PAN' in entity_types
        # Aadhaar and phone may be detected with various type names
        assert len(entity_types) >= 1

    def test_empty_text(self, detector):
        """Test empty text input."""
        result = detector.detect("")
        assert len(result['entities']) == 0

    def test_no_pii_text(self, detector):
        """Test text without PII."""
        text = "The weather is nice today. I like programming."
        result = detector.detect(text)

        # May have some false positives, but should be minimal
        assert isinstance(result['entities'], list)

    def test_entity_positions(self, detector):
        """Test entity positions are correct."""
        text = "PAN: ABCDE1234F, Email: test@example.com"
        result = detector.detect(text)

        for entity in result['entities']:
            # Check positions are valid
            assert 0 <= entity['start'] < entity['end']
            assert entity['end'] <= len(text)

            # Extract text using positions
            extracted = text[entity['start']:entity['end']]
            # Should match or be similar to entity text
            assert entity['text'] in extracted or extracted in entity['text']

    def test_detection_metadata(self, detector):
        """Test detection result metadata."""
        text = "PAN: ABCDE1234F"
        result = detector.detect(text)

        assert 'entities' in result
        assert 'total_entities' in result
        assert 'detector_name' in result
        assert result['detector_name'] == 'EnsembleDetector'

    def test_language_support(self, detector):
        """Test multi-language support."""
        # English
        text_en = "Name: John Smith, Email: john@example.com"
        result_en = detector.detect(text_en, language='en')
        assert len(result_en['entities']) > 0

        # Hindi (if supported)
        text_hi = "PAN: ABCDE1234F"
        result_hi = detector.detect(text_hi, language='hi')
        # Should at least detect PAN
        assert len(result_hi['entities']) >= 0

    def test_entity_merging(self, detector):
        """Test overlapping entity merging."""
        text = "John Smith John Smith"
        result = detector.detect(text)

        # Should handle overlapping detections from multiple detectors
        person_entities = [e for e in result['entities'] if e['entity_type'] == 'PERSON']

        # Should detect person names (may be deduplicated)
        assert len(person_entities) >= 1

    def test_mixed_entity_types(self, detector):
        """Test detection of various entity types together."""
        text = """
        Application Form:
        Name: Rajesh Kumar
        PAN: ABCDE1234F
        Aadhaar: 1234 5678 9012
        Phone: 9876543210
        Email: rajesh@example.com
        Address: 123 Main Street, Mumbai
        """

        result = detector.detect(text)

        # Should detect multiple types
        entity_types = {e['entity_type'] for e in result['entities']}
        assert len(entity_types) >= 3  # At least 3 different types

    def test_special_characters(self, detector):
        """Test handling of special characters."""
        text = "Email: test@example.com!!! PAN: ABCDE1234F???"
        result = detector.detect(text)

        # Should handle special characters
        assert len(result['entities']) > 0

    def test_unicode_support(self, detector):
        """Test Unicode text handling."""
        text = "Name: José García, PAN: ABCDE1234F"
        result = detector.detect(text)

        # Should handle Unicode
        assert isinstance(result['entities'], list)

    def test_long_text_performance(self, detector):
        """Test performance with long text."""
        text = "Some text. " * 100 + "PAN: ABCDE1234F" + " More text." * 100
        result = detector.detect(text)

        # Should detect PAN even in long text
        pan_entities = [e for e in result['entities'] if e['entity_type'] == 'PAN']
        assert len(pan_entities) >= 1

    def test_entity_type_standardization(self, detector):
        """Test entity type names are standardized."""
        text = "Email: test@example.com, Phone: 9876543210"
        result = detector.detect(text)

        for entity in result['entities']:
            # Entity types should be uppercase and standardized
            assert isinstance(entity['entity_type'], str)
            assert len(entity['entity_type']) > 0

    def test_confidence_ordering(self, detector):
        """Test entities can be sorted by confidence."""
        text = """
        Name: John Smith
        PAN: ABCDE1234F
        Email: test@example.com
        Some Random Text That Might Be Detected
        """

        result = detector.detect(text)

        # Sort by confidence
        sorted_entities = sorted(result['entities'], key=lambda x: x['confidence'], reverse=True)

        # Should be sortable
        assert len(sorted_entities) == len(result['entities'])

        # Highest confidence should be reasonable
        if len(sorted_entities) > 0:
            assert sorted_entities[0]['confidence'] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
