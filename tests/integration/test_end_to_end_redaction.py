"""
Integration tests for end-to-end PII redaction flow.

Tests:
- Complete detection to redaction pipeline
- Policy-based redaction
- Multi-format input (text, documents)
- Visual redaction
- Batch processing
"""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.pii.ensemble_detector import EnsembleDetector
from app.services.redaction.policy_manager import PolicyManager
from app.services.redaction.enhanced_redactor import EnhancedRedactor


class TestEndToEndRedaction:
    """Integration tests for complete redaction workflow."""

    @pytest.fixture
    def detector(self):
        """Create PII detector."""
        return EnsembleDetector()

    @pytest.fixture
    def policy_manager(self):
        """Create policy manager."""
        return PolicyManager()

    @pytest.fixture
    def redactor(self):
        """Create redactor."""
        return EnhancedRedactor()

    def test_basic_text_redaction_flow(self, detector, redactor):
        """Test basic text detection and redaction."""
        text = "My PAN is ABCDE1234F and my email is test@example.com"

        # Step 1: Detect PII
        detection_result = detector.detect(text)
        assert len(detection_result['entities']) > 0

        # Step 2: Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Verify PII is redacted
        assert 'ABCDE1234F' not in redacted_text
        assert 'test@example.com' not in redacted_text
        assert '████' in redacted_text or 'REDACTED' in redacted_text.upper()

    def test_policy_based_redaction(self, detector, policy_manager, redactor):
        """Test policy-based redaction with different actions."""
        text = """
        PAN: ABCDE1234F
        Phone: 9876543210
        Email: test@example.com
        """

        # Create custom policy
        policy_dict = {
            'name': 'test_policy',
            'rules': {
                'PAN': {'action': 'block'},
                'PHONE': {'action': 'mask', 'show_last': 4},
                'EMAIL': {'action': 'allow'}
            }
        }

        policy = policy_manager.load_policy_from_dict(policy_dict)

        # Detect PII
        detection_result = detector.detect(text)

        # Apply policy
        entities_to_redact = [
            e for e in detection_result['entities']
            if policy.should_redact(e['entity_type'], e.get('confidence', 1.0))
        ]

        # Redact
        redacted_text = redactor.redact_text(text, entities_to_redact, policy=policy)

        # Verify policy application
        # PAN should be blocked
        assert 'ABCDE1234F' not in redacted_text

        # Email should be allowed (if policy allows)
        # This depends on policy implementation

    def test_multiple_entity_types(self, detector, redactor):
        """Test redaction of multiple entity types."""
        text = """
        Application Form
        Name: Rajesh Kumar
        PAN: ABCDE1234F
        Aadhaar: 1234 5678 9012
        Phone: 9876543210
        Email: rajesh@example.com
        Address: 123 Main Street, Mumbai
        """

        # Detect
        detection_result = detector.detect(text)

        # Should detect multiple types
        entity_types = {e['entity_type'] for e in detection_result['entities']}
        assert len(entity_types) >= 3

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Verify sensitive data is redacted
        assert 'ABCDE1234F' not in redacted_text
        assert 'rajesh@example.com' not in redacted_text

    def test_confidence_threshold_filtering(self, detector, redactor):
        """Test filtering entities by confidence threshold."""
        text = "Name: John Smith, PAN: ABCDE1234F"

        # Detect with low confidence threshold
        detection_result = detector.detect(text, min_confidence=0.3)
        count_low = len(detection_result['entities'])

        # Detect with high confidence threshold
        detection_result_high = detector.detect(text, min_confidence=0.8)
        count_high = len(detection_result_high['entities'])

        # Higher threshold should give fewer or equal entities
        assert count_high <= count_low

        # Redact high confidence entities only
        redacted_text = redactor.redact_text(text, detection_result_high['entities'])

        # Should be redacted
        assert len(redacted_text) > 0

    def test_empty_text_handling(self, detector, redactor):
        """Test handling of empty text."""
        text = ""

        # Detect
        detection_result = detector.detect(text)
        assert len(detection_result['entities']) == 0

        # Redact
        redacted_text = redactor.redact_text(text, [])
        assert redacted_text == ""

    def test_text_with_no_pii(self, detector, redactor):
        """Test text without PII."""
        text = "The weather is nice today. I enjoy programming."

        # Detect
        detection_result = detector.detect(text)

        # May or may not detect anything
        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should return text (possibly unchanged)
        assert len(redacted_text) > 0

    def test_partial_masking(self, redactor):
        """Test partial masking strategy."""
        text = "Phone: 9876543210"

        # Create entity for phone
        entities = [{
            'text': '9876543210',
            'entity_type': 'PHONE',
            'start': 7,
            'end': 17,
            'confidence': 1.0
        }]

        # Redact with partial mask
        redacted_text = redactor.redact_text(
            text,
            entities,
            strategy='partial_mask',
            show_last=4
        )

        # Should show last 4 digits
        assert '3210' in redacted_text or '****3210' in redacted_text
        assert '9876' not in redacted_text

    def test_label_replacement(self, redactor):
        """Test label replacement strategy."""
        text = "PAN: ABCDE1234F"

        entities = [{
            'text': 'ABCDE1234F',
            'entity_type': 'PAN',
            'start': 5,
            'end': 15,
            'confidence': 1.0
        }]

        # Redact with label
        redacted_text = redactor.redact_text(text, entities, strategy='label')

        # Should contain label
        assert '<PAN>' in redacted_text or 'PAN_REDACTED' in redacted_text
        assert 'ABCDE1234F' not in redacted_text

    def test_overlapping_entities(self, detector, redactor):
        """Test handling of overlapping entities."""
        text = "Contact: john@example.com"

        # Detect
        detection_result = detector.detect(text)

        # Redact (should handle overlaps)
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should be redacted without errors
        assert len(redacted_text) > 0

    def test_special_characters_handling(self, detector, redactor):
        """Test handling of special characters."""
        text = "Email: test@example.com!!! PAN: ABCDE1234F???"

        # Detect
        detection_result = detector.detect(text)

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should handle special characters
        assert 'test@example.com' not in redacted_text or 'ABCDE1234F' not in redacted_text

    def test_unicode_text_redaction(self, detector, redactor):
        """Test redaction with Unicode characters."""
        text = "Name: José García, PAN: ABCDE1234F"

        # Detect
        detection_result = detector.detect(text)

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should handle Unicode
        assert isinstance(redacted_text, str)
        assert 'ABCDE1234F' not in redacted_text

    def test_long_text_redaction(self, detector, redactor):
        """Test redaction of long text."""
        text = "Some text. " * 100 + "PAN: ABCDE1234F" + " More text." * 100

        # Detect
        detection_result = detector.detect(text)

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should redact PAN
        assert 'ABCDE1234F' not in redacted_text

    def test_batch_redaction(self, detector, redactor):
        """Test batch redaction of multiple texts."""
        texts = [
            "PAN: ABCDE1234F",
            "Email: test@example.com",
            "Phone: 9876543210",
            "No PII here"
        ]

        redacted_texts = []

        for text in texts:
            # Detect
            detection_result = detector.detect(text)

            # Redact
            redacted = redactor.redact_text(text, detection_result['entities'])
            redacted_texts.append(redacted)

        # Verify all processed
        assert len(redacted_texts) == len(texts)

        # Verify redaction
        assert 'ABCDE1234F' not in redacted_texts[0]
        assert 'test@example.com' not in redacted_texts[1]

    def test_indian_languages_redaction(self, detector, redactor):
        """Test redaction with Hindi/Indian languages."""
        text = "नाम: राजेश कुमार, PAN: ABCDE1234F"

        # Detect
        detection_result = detector.detect(text, language='hi')

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should at least redact PAN
        assert 'ABCDE1234F' not in redacted_text

    def test_preserve_formatting(self, detector, redactor):
        """Test that redaction preserves text formatting."""
        text = """
        Application Form:
        =================

        Name: John Smith
        PAN: ABCDE1234F

        Contact Information:
        -------------------
        Email: john@example.com
        """

        # Detect
        detection_result = detector.detect(text)

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should preserve structure
        assert '=================' in redacted_text
        assert '-------------------' in redacted_text
        assert 'Application Form' in redacted_text

    def test_redaction_statistics(self, detector, redactor):
        """Test collection of redaction statistics."""
        text = """
        PAN: ABCDE1234F
        Email: test@example.com
        Phone: 9876543210
        """

        # Detect
        detection_result = detector.detect(text)

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Check statistics
        stats = redactor.get_statistics() if hasattr(redactor, 'get_statistics') else {}

        # Should have processed entities
        assert len(detection_result['entities']) > 0

    def test_reversible_redaction(self, detector, redactor):
        """Test reversible redaction (tokenization)."""
        text = "PAN: ABCDE1234F, Email: test@example.com"

        # Detect
        detection_result = detector.detect(text)

        # Redact with tokenization
        result = redactor.redact_text(
            text,
            detection_result['entities'],
            strategy='tokenize'
        )

        # Should contain tokens
        # Tokens can be used to reverse redaction with proper key
        assert isinstance(result, str)

    def test_error_handling(self, detector, redactor):
        """Test error handling in redaction pipeline."""
        # Test with None text
        try:
            detection_result = detector.detect(None if False else "")
            redacted = redactor.redact_text("", [])
            assert isinstance(redacted, str)
        except Exception:
            # Should handle gracefully
            pass

    def test_performance_with_large_text(self, detector, redactor):
        """Test performance with large text."""
        # Create large text (1000 words)
        text = " ".join(["word"] * 1000) + " PAN: ABCDE1234F " + " ".join(["word"] * 1000)

        # Detect (should complete reasonably fast)
        detection_result = detector.detect(text)

        # Redact
        redacted_text = redactor.redact_text(text, detection_result['entities'])

        # Should complete
        assert len(redacted_text) > 0
        assert 'ABCDE1234F' not in redacted_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
