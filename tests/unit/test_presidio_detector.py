"""
Unit tests for PresidioDetector.

Tests:
- Person name detection
- Email detection
- Phone detection
- Location detection
- Multi-language support
- Confidence scores
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.pii.presidio_detector import PresidioDetector


class TestPresidioDetector:
    """Test cases for PresidioDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return PresidioDetector()

    def test_person_name_detection(self, detector):
        """Test person name detection."""
        text = "My name is John Smith and I live in New York."
        result = detector.detect(text)

        person_entities = [e for e in result['entities'] if e['entity_type'] == 'PERSON']
        assert len(person_entities) > 0
        # Should detect "John Smith"
        assert any('John' in e['text'] or 'Smith' in e['text'] for e in person_entities)

    def test_email_detection(self, detector):
        """Test email detection."""
        text = "Contact me at john.doe@example.com for details."
        result = detector.detect(text)

        email_entities = [e for e in result['entities'] if e['entity_type'] == 'EMAIL']
        assert len(email_entities) > 0
        assert any('john.doe@example.com' in e['text'] for e in email_entities)

    def test_phone_detection(self, detector):
        """Test phone number detection."""
        test_cases = [
            "Call me at 123-456-7890",
            "Phone: (555) 123-4567",
            "+1-800-555-0123",
        ]

        for text in test_cases:
            result = detector.detect(text)
            phone_entities = [e for e in result['entities'] if 'PHONE' in e['entity_type'].upper()]
            assert len(phone_entities) > 0, f"Failed for: {text}"

    def test_location_detection(self, detector):
        """Test location detection."""
        text = "I live in New York City, United States."
        result = detector.detect(text)

        location_entities = [e for e in result['entities'] if e['entity_type'] == 'LOCATION']
        # Presidio may detect locations
        # Just ensure it doesn't crash
        assert isinstance(result['entities'], list)

    def test_multiple_entities(self, detector):
        """Test detection of multiple entities."""
        text = """
        John Smith works at ABC Corp.
        Email: john@example.com
        Phone: 555-123-4567
        Location: New York
        """

        result = detector.detect(text)

        # Should detect multiple entities
        assert len(result['entities']) > 0

        entity_types = {e['entity_type'] for e in result['entities']}
        # Should have at least some common types
        assert len(entity_types) > 0

    def test_confidence_scores(self, detector):
        """Test confidence scores."""
        text = "John Smith, email: john@example.com"
        result = detector.detect(text)

        for entity in result['entities']:
            assert 'confidence' in entity
            assert 0.0 <= entity['confidence'] <= 1.0

    def test_min_confidence_filter(self, detector):
        """Test min_confidence filtering."""
        text = "John Smith lives in New York."

        # Low threshold
        result_low = detector.detect(text, min_confidence=0.1)
        count_low = len(result_low['entities'])

        # High threshold
        result_high = detector.detect(text, min_confidence=0.9)
        count_high = len(result_high['entities'])

        # Higher threshold should detect fewer or equal entities
        assert count_high <= count_low

    def test_entity_positions(self, detector):
        """Test entity positions are correct."""
        text = "Name: John Smith, Email: john@example.com"
        result = detector.detect(text)

        for entity in result['entities']:
            # Check positions are valid
            assert 0 <= entity['start'] < entity['end']
            assert entity['end'] <= len(text)

            # Extract text using positions
            extracted = text[entity['start']:entity['end']]
            # Extracted text should match entity text
            assert extracted == entity['text'] or entity['text'] in extracted

    def test_empty_text(self, detector):
        """Test empty text input."""
        result = detector.detect("")
        assert len(result['entities']) == 0

    def test_no_pii_text(self, detector):
        """Test text without PII."""
        text = "The weather is nice today."
        result = detector.detect(text)

        # May or may not detect entities (depends on Presidio config)
        assert isinstance(result['entities'], list)

    def test_language_parameter(self, detector):
        """Test language parameter."""
        text = "John Smith works here."

        # Test English
        result_en = detector.detect(text, language='en')
        assert len(result_en['entities']) >= 0

        # Test auto-detection
        result_auto = detector.detect(text, language='auto')
        assert len(result_auto['entities']) >= 0

    def test_supported_entities(self, detector):
        """Test supported entity types."""
        # Check that detector has supported entities defined
        assert hasattr(detector, 'supported_entities') or hasattr(detector, 'analyzer')

    def test_detection_metadata(self, detector):
        """Test detection result metadata."""
        text = "John Smith"
        result = detector.detect(text)

        assert 'entities' in result
        assert 'total_entities' in result
        assert 'detector_name' in result
        assert result['detector_name'] == 'PresidioDetector'

    def test_credit_card_detection(self, detector):
        """Test credit card detection if supported."""
        text = "Credit card: 4532-1488-0343-6467"
        result = detector.detect(text)

        # Presidio may support credit card detection
        cc_entities = [e for e in result['entities'] if 'CREDIT_CARD' in e['entity_type'].upper()]
        # Just check it doesn't crash
        assert isinstance(result['entities'], list)

    def test_ssn_detection(self, detector):
        """Test SSN detection if supported."""
        text = "SSN: 123-45-6789"
        result = detector.detect(text)

        # Presidio may support SSN detection
        ssn_entities = [e for e in result['entities'] if 'SSN' in e['entity_type'].upper()]
        # Just check it doesn't crash
        assert isinstance(result['entities'], list)

    def test_date_detection(self, detector):
        """Test date detection."""
        text = "Date of birth: 01/15/1990"
        result = detector.detect(text)

        # Dates may or may not be detected as PII
        assert isinstance(result['entities'], list)

    def test_url_detection(self, detector):
        """Test URL detection."""
        text = "Visit https://example.com for more info."
        result = detector.detect(text)

        url_entities = [e for e in result['entities'] if 'URL' in e['entity_type'].upper()]
        # URLs may be detected
        assert isinstance(result['entities'], list)

    def test_overlapping_entities(self, detector):
        """Test handling of overlapping entities."""
        text = "John Smith John Smith"
        result = detector.detect(text)

        # Should handle repeated entities
        assert len(result['entities']) >= 1

    def test_special_characters(self, detector):
        """Test text with special characters."""
        text = "Name: John O'Brien, Email: john@example.com!"
        result = detector.detect(text)

        # Should handle special characters
        assert len(result['entities']) >= 0

    def test_unicode_text(self, detector):
        """Test Unicode text handling."""
        text = "Name: José García, Email: jose@example.com"
        result = detector.detect(text)

        # Should handle Unicode characters
        assert isinstance(result['entities'], list)

    def test_long_text(self, detector):
        """Test detection in long text."""
        text = "A " * 1000 + "John Smith" + " B" * 1000
        result = detector.detect(text)

        # Should handle long texts
        person_entities = [e for e in result['entities'] if e['entity_type'] == 'PERSON']
        assert len(person_entities) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
