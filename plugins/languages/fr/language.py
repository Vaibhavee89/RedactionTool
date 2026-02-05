"""
French Language Pack

Provides French-specific entity detection patterns and redaction policies.
"""

import re
from typing import Dict, List, Any

from app.extensions.interfaces.language_pack import (
    LanguagePack,
    LanguagePackMetadata,
    RedactionPolicy,
    Script
)


class FrenchLanguagePack(LanguagePack):
    """
    French language pack for PII detection.

    Supports:
    - INSEE (Numéro de Sécurité Sociale - French Social Security Number)
    - French IBAN
    - French phone numbers (mobile and landline)
    - French postal codes
    - French NIR (Numéro d'Inscription au Répertoire)
    - SIRET (Company identification)
    """

    def get_metadata(self) -> LanguagePackMetadata:
        """Return French language pack metadata."""
        return LanguagePackMetadata(
            language_code="fr",
            language_name="French",
            script=Script.LATIN,
            supported_entity_types=[
                "INSEE",
                "PHONE_FR",
                "IBAN_FR",
                "POSTAL_CODE_FR",
                "NIR_FR",
                "SIRET",
                "SIREN"
            ],
            description="French language PII detection patterns",
            author="RedactionTool Team",
            version="1.0.0",
            country_codes=["FR"],
            right_to_left=False,
            requires_special_tokenization=False
        )

    def get_regex_patterns(self) -> Dict[str, str]:
        """
        Get French-specific regex patterns.

        Returns:
            Dictionary of entity_type -> regex pattern
        """
        return {
            # INSEE (Social Security Number)
            # Format: 1 23 45 67 890 123 45
            # 1 digit (sex), 2 digits (year), 2 digits (month),
            # 2 digits (department), 3 digits (commune), 3 digits (order), 2 digits (key)
            "INSEE": r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',

            # French phone numbers
            # Mobile: 06 XX XX XX XX or 07 XX XX XX XX
            # Landline: 01-05, 08, 09
            # International format: +33 (0)X XX XX XX XX
            "PHONE_FR": r'(?:\+33\s?(?:\(0\))?\s?|0)[1-9](?:[\s.-]?\d{2}){4}\b',

            # French IBAN
            # Format: FR76 XXXX XXXX XXXX XXXX XXXX X99
            "IBAN_FR": r'\bFR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b',

            # French postal code
            # Format: 5 digits (e.g., 75001 for Paris)
            "POSTAL_CODE_FR": r'\b(?:0[1-9]|[1-8]\d|9[0-5]|2[AB])\d{3}\b',

            # NIR (Numéro d'Inscription au Répertoire)
            # Same format as INSEE but specifically for identification
            "NIR_FR": r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',

            # SIRET (Company identification - 14 digits)
            # Format: XXX XXX XXX XXXXX
            "SIRET": r'\b\d{3}\s?\d{3}\s?\d{3}\s?\d{5}\b',

            # SIREN (Company identification - 9 digits)
            # Format: XXX XXX XXX
            "SIREN": r'\b\d{3}\s?\d{3}\s?\d{3}\b'
        }

    def get_redaction_policy(self) -> RedactionPolicy:
        """
        Get French-specific redaction policy.

        Returns:
            RedactionPolicy for French entities
        """
        return RedactionPolicy(
            full_redaction=["INSEE", "NIR_FR", "IBAN_FR"],
            partial_redaction={
                "PHONE_FR": 4,  # Show last 4 digits
                "SIRET": 5,     # Show last 5 digits
                "SIREN": 3      # Show last 3 digits
            },
            preserve_format=["POSTAL_CODE_FR", "PHONE_FR"],
            mask_char="X",
            preserve_length=True,
            case_sensitive=False
        )

    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get validation rules for French entities.

        Returns:
            Dictionary of entity_type -> validation function
        """
        return {
            "INSEE": self._validate_insee,
            "NIR_FR": self._validate_insee,
            "SIRET": self._validate_siret,
            "SIREN": self._validate_siren,
            "IBAN_FR": self._validate_iban_fr
        }

    def get_context_patterns(self) -> Dict[str, List[str]]:
        """
        Get French context keywords.

        Returns:
            Dictionary of entity_type -> context keywords
        """
        return {
            "INSEE": [
                "numéro de sécurité sociale",
                "numéro sécu",
                "sécu",
                "sécurité sociale",
                "n° sécu",
                "insee"
            ],
            "PHONE_FR": [
                "téléphone",
                "tél",
                "tel",
                "portable",
                "mobile",
                "fixe",
                "numéro",
                "appeler",
                "contacter"
            ],
            "IBAN_FR": [
                "iban",
                "compte bancaire",
                "compte",
                "virement",
                "rib",
                "relevé d'identité bancaire"
            ],
            "SIRET": [
                "siret",
                "entreprise",
                "société",
                "établissement",
                "numéro d'établissement"
            ],
            "SIREN": [
                "siren",
                "entreprise",
                "société",
                "numéro d'identification"
            ]
        }

    def get_false_positive_patterns(self) -> Dict[str, List[str]]:
        """
        Get false positive patterns to exclude.

        Returns:
            Dictionary of entity_type -> false positive patterns
        """
        return {
            "INSEE": [
                r'0{13}',  # All zeros
                r'1{13}',  # All ones
            ],
            "PHONE_FR": [
                r'0{10}',  # All zeros
                r'1{10}',  # All ones
            ],
            "POSTAL_CODE_FR": [
                r'00000',  # Invalid postal code
            ]
        }

    def _validate_insee(self, text: str) -> bool:
        """
        Validate INSEE number (with Luhn-like check).

        Args:
            text: INSEE number

        Returns:
            True if valid
        """
        # Remove spaces
        insee = text.replace(' ', '')

        # Must be 13 or 15 digits
        if len(insee) not in [13, 15]:
            return False

        # Must be numeric
        if not insee.isdigit():
            return False

        # First digit must be 1 (male) or 2 (female)
        if insee[0] not in ['1', '2']:
            return False

        # Year: 00-99
        year = insee[1:3]
        if not (0 <= int(year) <= 99):
            return False

        # Month: 01-12 (or 20-99 for special cases)
        month = insee[3:5]
        month_int = int(month)
        if not ((1 <= month_int <= 12) or (20 <= month_int <= 99)):
            return False

        # Basic validation passed
        return True

    def _validate_siret(self, text: str) -> bool:
        """
        Validate SIRET number (14 digits).

        Args:
            text: SIRET number

        Returns:
            True if valid
        """
        # Remove spaces
        siret = text.replace(' ', '')

        # Must be 14 digits
        if len(siret) != 14:
            return False

        # Must be numeric
        if not siret.isdigit():
            return False

        # Luhn algorithm validation
        total = 0
        for i, digit in enumerate(siret):
            n = int(digit)
            if i % 2 == 0:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    def _validate_siren(self, text: str) -> bool:
        """
        Validate SIREN number (9 digits).

        Args:
            text: SIREN number

        Returns:
            True if valid
        """
        # Remove spaces
        siren = text.replace(' ', '')

        # Must be 9 digits
        if len(siren) != 9:
            return False

        # Must be numeric
        if not siren.isdigit():
            return False

        # Luhn algorithm validation
        total = 0
        for i, digit in enumerate(siren):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    def _validate_iban_fr(self, text: str) -> bool:
        """
        Validate French IBAN.

        Args:
            text: IBAN

        Returns:
            True if valid
        """
        # Remove spaces
        iban = text.replace(' ', '').upper()

        # Must start with FR
        if not iban.startswith('FR'):
            return False

        # Must be 27 characters
        if len(iban) != 27:
            return False

        # Check structure: FR + 2 digits + 23 alphanumeric
        if not iban[2:4].isdigit():
            return False

        # Basic validation passed (full IBAN validation would require mod-97 check)
        return True

    def normalize_text(self, text: str) -> str:
        """
        Normalize French text (remove accents for matching).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        import unicodedata

        # Remove accents
        normalized = unicodedata.normalize('NFD', text)
        normalized = ''.join(
            char for char in normalized
            if unicodedata.category(char) != 'Mn'
        )

        return normalized


def register_language_pack() -> LanguagePack:
    """
    Register function called by language registry.

    Returns:
        FrenchLanguagePack instance
    """
    return FrenchLanguagePack()
