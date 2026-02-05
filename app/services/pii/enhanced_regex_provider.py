"""
Enhanced Regex Provider with comprehensive patterns and locale-aware detection.
"""

import re
from typing import List, Dict, Any, Pattern
from datetime import datetime


class EnhancedRegexProvider:
    """
    Enhanced regex-based PII detection with:
    - Phone numbers (Indian and international)
    - Email addresses
    - Bank account numbers
    - Dates (multiple formats)
    - Indian IDs (PAN, Aadhaar, Voter ID, Driving License)
    - Credit cards
    - SSN
    - Locale-aware formats
    """

    def __init__(self):
        """Initialize regex patterns."""
        self.patterns = self._init_patterns()
        self.compiled_patterns = {
            label: re.compile(pattern, re.IGNORECASE if 'EMAIL' in label or 'URL' in label else 0)
            for label, pattern in self.patterns.items()
        }

    def _init_patterns(self) -> Dict[str, str]:
        """Initialize all regex patterns."""
        return {
            # Email
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

            # Phone Numbers
            "PHONE_IN": r'\b(\+?91[\s-]?)?[6-9]\d{9}\b',  # Indian
            "PHONE_US": r'\b(\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',  # US
            "PHONE_INTL": r'\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}',  # International

            # Indian Government IDs
            "PAN": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',  # PAN Card
            "AADHAAR": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Aadhaar
            "VOTER_ID": r'\b[A-Z]{3}[0-9]{7}\b',  # Voter ID
            "DRIVING_LICENSE_IN": r'\b[A-Z]{2}[-]?\d{2}[-]?\d{4}[-]?\d{7}\b',  # Indian DL

            # Bank Account Numbers
            "BANK_ACCOUNT_IN": r'\b\d{9,18}\b',  # Indian bank account (9-18 digits)
            "IFSC": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',  # IFSC code
            "SWIFT": r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',  # SWIFT/BIC code

            # Credit Card
            "CREDIT_CARD": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',

            # SSN (US)
            "SSN": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',

            # Passport
            "PASSPORT": r'\b[A-Z]\d{7}\b',  # Simple pattern

            # Dates (multiple formats)
            "DATE_DMY": r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # DD-MM-YYYY
            "DATE_MDY": r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # MM-DD-YYYY
            "DATE_YMD": r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b',  # YYYY-MM-DD
            "DATE_TEXT": r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{2,4}\b',

            # IP Address
            "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',

            # URL
            "URL": r'https?://[^\s<>"{}|\\^`\[\]]+',

            # Medical Record Number
            "MEDICAL_RECORD": r'\bMRN[-\s:]?\d{6,10}\b',

            # Vehicle Registration
            "VEHICLE_REG_IN": r'\b[A-Z]{2}[-]?\d{2}[-]?[A-Z]{1,2}[-]?\d{4}\b',
        }

    def detect(self, text: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Detect PII using regex patterns.

        Args:
            text: Input text
            entity_types: Filter by specific entity types (None = all)

        Returns:
            List of detected entities
        """
        entities = []

        for label, pattern in self.compiled_patterns.items():
            # Filter by entity types if specified
            if entity_types and label not in entity_types:
                continue

            for match in pattern.finditer(text):
                matched_text = match.group()

                # Validate match
                if not self._validate_match(label, matched_text):
                    continue

                # Calculate confidence
                confidence = self._calculate_confidence(label, matched_text)

                entities.append({
                    "entity_type": label,
                    "start": match.start(),
                    "end": match.end(),
                    "text": matched_text,
                    "confidence": confidence,
                    "source": "regex",
                    "pattern": label
                })

        return sorted(entities, key=lambda x: x['start'])

    def _validate_match(self, label: str, text: str) -> bool:
        """
        Validate matched text with additional checks.

        Args:
            label: Entity type label
            text: Matched text

        Returns:
            True if valid
        """
        # PAN validation (checksum could be added)
        if label == "PAN":
            return len(text) == 10 and text[3].isdigit()

        # Aadhaar validation (basic length check)
        if label == "AADHAAR":
            digits = ''.join(c for c in text if c.isdigit())
            return len(digits) == 12

        # Credit card validation (Luhn algorithm)
        if label == "CREDIT_CARD":
            return self._validate_credit_card(text)

        # IP address validation
        if label == "IP_ADDRESS":
            parts = text.split('.')
            return all(0 <= int(p) <= 255 for p in parts)

        # Email validation (basic)
        if label == "EMAIL":
            return '@' in text and '.' in text.split('@')[1]

        # Bank account validation
        if label == "BANK_ACCOUNT_IN":
            return 9 <= len(text) <= 18

        return True

    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        # Remove spaces and dashes
        number = ''.join(c for c in number if c.isdigit())

        if len(number) < 13 or len(number) > 19:
            return False

        # Luhn algorithm
        def luhn_checksum(card_number):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card_number)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10

        return luhn_checksum(number) == 0

    def _calculate_confidence(self, label: str, text: str) -> float:
        """
        Calculate confidence score for matched entity.

        Args:
            label: Entity type label
            text: Matched text

        Returns:
            Confidence score (0-1)
        """
        # High confidence patterns
        high_confidence = ['EMAIL', 'PAN', 'IFSC', 'SWIFT']
        if label in high_confidence:
            return 0.95

        # Medium-high confidence
        medium_high = ['AADHAAR', 'VOTER_ID', 'DRIVING_LICENSE_IN', 'PHONE_IN']
        if label in medium_high:
            return 0.90

        # Medium confidence
        medium = ['CREDIT_CARD', 'SSN', 'PASSPORT']
        if label in medium:
            return 0.85

        # Lower confidence (dates, numbers can be ambiguous)
        if label.startswith('DATE_'):
            return 0.75

        if label in ['BANK_ACCOUNT_IN', 'IP_ADDRESS']:
            return 0.70

        return 0.80  # Default

    def detect_by_category(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect and group entities by category.

        Args:
            text: Input text

        Returns:
            Dictionary of entity categories
        """
        all_entities = self.detect(text)

        categories = {
            'contact': [],
            'identification': [],
            'financial': [],
            'dates': [],
            'other': []
        }

        category_map = {
            'EMAIL': 'contact',
            'PHONE_IN': 'contact',
            'PHONE_US': 'contact',
            'PHONE_INTL': 'contact',
            'URL': 'contact',

            'PAN': 'identification',
            'AADHAAR': 'identification',
            'VOTER_ID': 'identification',
            'DRIVING_LICENSE_IN': 'identification',
            'SSN': 'identification',
            'PASSPORT': 'identification',

            'BANK_ACCOUNT_IN': 'financial',
            'IFSC': 'financial',
            'SWIFT': 'financial',
            'CREDIT_CARD': 'financial',

            'DATE_DMY': 'dates',
            'DATE_MDY': 'dates',
            'DATE_YMD': 'dates',
            'DATE_TEXT': 'dates',
        }

        for entity in all_entities:
            category = category_map.get(entity['entity_type'], 'other')
            categories[category].append(entity)

        return categories

    def get_statistics(self, text: str) -> Dict[str, int]:
        """
        Get statistics on detected entities.

        Args:
            text: Input text

        Returns:
            Dictionary with entity counts
        """
        entities = self.detect(text)
        stats = {}

        for entity in entities:
            entity_type = entity['entity_type']
            stats[entity_type] = stats.get(entity_type, 0) + 1

        return stats
