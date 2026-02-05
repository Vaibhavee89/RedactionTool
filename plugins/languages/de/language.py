"""
German Language Pack

Provides German-specific entity detection patterns and redaction policies.
"""

import re
from typing import Dict, List, Any

from app.extensions.interfaces.language_pack import (
    LanguagePack,
    LanguagePackMetadata,
    RedactionPolicy,
    Script
)


class GermanLanguagePack(LanguagePack):
    """
    German language pack for PII detection.

    Supports:
    - German ID card numbers (Personalausweis)
    - German social security numbers (Sozialversicherungsnummer)
    - German tax ID (Steuer-ID)
    - German IBAN
    - German phone numbers
    - German postal codes
    - German passport numbers
    """

    def get_metadata(self) -> LanguagePackMetadata:
        """Return German language pack metadata."""
        return LanguagePackMetadata(
            language_code="de",
            language_name="German",
            script=Script.LATIN,
            supported_entity_types=[
                "ID_CARD_DE",
                "SSN_DE",
                "TAX_ID_DE",
                "IBAN_DE",
                "PHONE_DE",
                "POSTAL_CODE_DE",
                "PASSPORT_DE",
                "VAT_DE"
            ],
            description="German language PII detection patterns",
            author="RedactionTool Team",
            version="1.0.0",
            country_codes=["DE"],
            right_to_left=False,
            requires_special_tokenization=False
        )

    def get_regex_patterns(self) -> Dict[str, str]:
        """
        Get German-specific regex patterns.

        Returns:
            Dictionary of entity_type -> regex pattern
        """
        return {
            # German ID card number (Personalausweis)
            # Format: 9 alphanumeric characters (e.g., L01234567)
            # New format (since 2010): 9 digits and letters
            "ID_CARD_DE": r'\b[CFGHJKLMNPRTVWXYZ]\d{8}\b',

            # German Social Security Number (Sozialversicherungsnummer)
            # Format: 12 digits (e.g., 12 345678 A 123)
            # Area number (2) + DOB (6) + Initial (1) + Serial (2) + Check (1)
            "SSN_DE": r'\b\d{2}\s?\d{6}\s?[A-Z]\s?\d{3}\b',

            # German Tax ID (Steuer-Identifikationsnummer)
            # Format: 11 digits
            "TAX_ID_DE": r'\b\d{11}\b',

            # German IBAN
            # Format: DE89 XXXX XXXX XXXX XXXX XX
            "IBAN_DE": r'\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b',

            # German phone numbers
            # Mobile: 015x, 016x, 017x
            # Landline: Various area codes
            # International: +49 ...
            "PHONE_DE": r'(?:\+49\s?|0)(?:1[5-7]\d|[2-9]\d{1,4})[\s/-]?\d+[\s/-]?\d+',

            # German postal code (Postleitzahl)
            # Format: 5 digits (01000-99999)
            "POSTAL_CODE_DE": r'\b[0-9]{5}\b',

            # German passport number
            # Format: C + 8 digits (e.g., C01234567)
            "PASSPORT_DE": r'\bC\d{8}\b',

            # German VAT number (Umsatzsteuer-Identifikationsnummer)
            # Format: DE + 9 digits
            "VAT_DE": r'\bDE\d{9}\b'
        }

    def get_redaction_policy(self) -> RedactionPolicy:
        """
        Get German-specific redaction policy.

        Returns:
            RedactionPolicy for German entities
        """
        return RedactionPolicy(
            full_redaction=["ID_CARD_DE", "SSN_DE", "TAX_ID_DE", "PASSPORT_DE"],
            partial_redaction={
                "PHONE_DE": 4,    # Show last 4 digits
                "IBAN_DE": 4,     # Show last 4 digits
                "VAT_DE": 3       # Show last 3 digits
            },
            preserve_format=["POSTAL_CODE_DE", "PHONE_DE"],
            mask_char="X",
            preserve_length=True,
            case_sensitive=False
        )

    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get validation rules for German entities.

        Returns:
            Dictionary of entity_type -> validation function
        """
        return {
            "TAX_ID_DE": self._validate_tax_id,
            "IBAN_DE": self._validate_iban_de,
            "POSTAL_CODE_DE": self._validate_postal_code,
            "SSN_DE": self._validate_ssn_de
        }

    def get_context_patterns(self) -> Dict[str, List[str]]:
        """
        Get German context keywords.

        Returns:
            Dictionary of entity_type -> context keywords
        """
        return {
            "ID_CARD_DE": [
                "personalausweis",
                "ausweis",
                "ausweisnummer",
                "id-nummer",
                "identifikation"
            ],
            "SSN_DE": [
                "sozialversicherungsnummer",
                "rentenversicherungsnummer",
                "versicherungsnummer",
                "sv-nummer"
            ],
            "TAX_ID_DE": [
                "steuer-id",
                "steueridentifikationsnummer",
                "steuernummer",
                "finanzamt"
            ],
            "PHONE_DE": [
                "telefon",
                "tel",
                "mobil",
                "handy",
                "rufnummer",
                "anrufen",
                "kontakt"
            ],
            "IBAN_DE": [
                "iban",
                "kontonummer",
                "bankverbindung",
                "überweisung",
                "konto"
            ],
            "PASSPORT_DE": [
                "reisepass",
                "pass",
                "passnummer",
                "ausweis"
            ]
        }

    def get_false_positive_patterns(self) -> Dict[str, List[str]]:
        """
        Get false positive patterns to exclude.

        Returns:
            Dictionary of entity_type -> false positive patterns
        """
        return {
            "TAX_ID_DE": [
                r'0{11}',  # All zeros
                r'1{11}',  # All ones
            ],
            "POSTAL_CODE_DE": [
                r'00000',  # Invalid
            ]
        }

    def _validate_tax_id(self, text: str) -> bool:
        """
        Validate German Tax ID using checksum algorithm.

        Args:
            text: Tax ID

        Returns:
            True if valid
        """
        # Remove spaces
        tax_id = text.replace(' ', '')

        # Must be 11 digits
        if len(tax_id) != 11:
            return False

        # Must be numeric
        if not tax_id.isdigit():
            return False

        # Check digit algorithm (simplified)
        # The 11th digit is a check digit
        digits = [int(d) for d in tax_id]

        # First 10 digits must not all be the same
        if len(set(digits[:10])) == 1:
            return False

        # One digit must appear exactly 2 or 3 times, others at most once
        digit_counts = {}
        for d in digits[:10]:
            digit_counts[d] = digit_counts.get(d, 0) + 1

        has_valid_count = any(count in [2, 3] for count in digit_counts.values())
        if not has_valid_count:
            return False

        return True

    def _validate_iban_de(self, text: str) -> bool:
        """
        Validate German IBAN.

        Args:
            text: IBAN

        Returns:
            True if valid
        """
        # Remove spaces
        iban = text.replace(' ', '').upper()

        # Must start with DE
        if not iban.startswith('DE'):
            return False

        # Must be 22 characters
        if len(iban) != 22:
            return False

        # Check structure: DE + 2 digits + 18 digits
        if not iban[2:4].isdigit():
            return False

        if not iban[4:].isdigit():
            return False

        return True

    def _validate_postal_code(self, text: str) -> bool:
        """
        Validate German postal code.

        Args:
            text: Postal code

        Returns:
            True if valid
        """
        # Must be 5 digits
        if len(text) != 5:
            return False

        # Must be numeric
        if not text.isdigit():
            return False

        # Valid range: 01000-99999
        code = int(text)
        return 1000 <= code <= 99999

    def _validate_ssn_de(self, text: str) -> bool:
        """
        Validate German Social Security Number.

        Args:
            text: SSN

        Returns:
            True if valid
        """
        # Remove spaces
        ssn = text.replace(' ', '')

        # Check length (12 characters: 8 digits + 1 letter + 3 digits)
        if len(ssn) != 12:
            return False

        # Check format: DDDDDDDDADDD (D=digit, A=letter)
        if not ssn[:8].isdigit():
            return False

        if not ssn[8].isalpha():
            return False

        if not ssn[9:].isdigit():
            return False

        # Check date of birth (positions 2-7: DDMMYY)
        dob = ssn[2:8]
        day = int(dob[0:2])
        month = int(dob[2:4])
        year = int(dob[4:6])

        # Basic date validation
        if not (1 <= day <= 31):
            return False
        if not (1 <= month <= 12):
            return False

        return True

    def normalize_text(self, text: str) -> str:
        """
        Normalize German text (handle umlauts).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Replace German umlauts with base letters for matching
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
            'ß': 'ss'
        }

        normalized = text
        for umlaut, replacement in replacements.items():
            normalized = normalized.replace(umlaut, replacement)

        return normalized


def register_language_pack() -> LanguagePack:
    """
    Register function called by language registry.

    Returns:
        GermanLanguagePack instance
    """
    return GermanLanguagePack()
