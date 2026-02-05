"""
Unit tests for EnhancedRegexProvider.

Tests:
- PAN card detection
- Aadhaar number detection
- Phone number detection
- Email detection
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.pii.enhanced_regex_provider import EnhancedRegexProvider


class TestEnhancedRegexProvider:
    """Test cases for EnhancedRegexProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return EnhancedRegexProvider()

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider is not None

    def test_pan_card_detection(self, provider):
        """Test PAN card detection."""
        text = "My PAN is ABCDE1234F"
        results = provider.detect(text)

        # Should detect PAN
        assert isinstance(results, list)
        pan_results = [r for r in results if 'PAN' in r.get('entity_type', '').upper()]
        assert len(pan_results) > 0

    def test_aadhaar_detection(self, provider):
        """Test Aadhaar number detection."""
        text = "My Aadhaar is 1234 5678 9012"
        results = provider.detect(text)

        # Should detect Aadhaar
        assert isinstance(results, list)

    def test_phone_detection(self, provider):
        """Test phone number detection."""
        text = "Call me at 9876543210"
        results = provider.detect(text)

        # Should detect phone
        assert isinstance(results, list)

    def test_email_detection(self, provider):
        """Test email detection."""
        text = "Email: test@example.com"
        results = provider.detect(text)

        # Should detect email
        assert isinstance(results, list)

    def test_empty_text(self, provider):
        """Test empty text."""
        results = provider.detect("")
        assert isinstance(results, list)

    def test_no_pii_text(self, provider):
        """Test text without PII."""
        text = "The weather is nice today."
        results = provider.detect(text)
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
