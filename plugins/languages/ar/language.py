"""
Arabic Language Pack

Provides Arabic-specific entity detection patterns and redaction policies.
"""

import re
from typing import Dict, List, Any

from app.extensions.interfaces.language_pack import (
    LanguagePack,
    LanguagePackMetadata,
    RedactionPolicy,
    Script
)


class ArabicLanguagePack(LanguagePack):
    """
    Arabic language pack for PII detection.

    Supports:
    - Arabic National ID (various formats)
    - Arabic phone numbers (Saudi, UAE, Egypt, etc.)
    - Arabic postal codes
    - Arabic names (common patterns)
    - IBAN for Arabic countries
    """

    def get_metadata(self) -> LanguagePackMetadata:
        """Return Arabic language pack metadata."""
        return LanguagePackMetadata(
            language_code="ar",
            language_name="Arabic",
            script=Script.ARABIC,
            supported_entity_types=[
                "NATIONAL_ID_SA",  # Saudi Arabia
                "NATIONAL_ID_AE",  # UAE
                "NATIONAL_ID_EG",  # Egypt
                "PHONE_AR",
                "IBAN_AR",
                "PASSPORT_AR",
                "ARABIC_NAME"
            ],
            description="Arabic language PII detection patterns",
            author="RedactionTool Team",
            version="1.0.0",
            country_codes=["SA", "AE", "EG", "KW", "QA", "BH", "OM", "JO", "LB", "IQ"],
            right_to_left=True,
            requires_special_tokenization=True
        )

    def get_regex_patterns(self) -> Dict[str, str]:
        """
        Get Arabic-specific regex patterns.

        Returns:
            Dictionary of entity_type -> regex pattern
        """
        return {
            # Saudi National ID
            # Format: 1 or 2 followed by 9 digits
            "NATIONAL_ID_SA": r'\b[12]\d{9}\b',

            # UAE National ID
            # Format: 784-YYYY-XXXXXXX-X (15 digits)
            "NATIONAL_ID_AE": r'\b784-?\d{4}-?\d{7}-?\d\b',

            # Egyptian National ID
            # Format: 14 digits
            "NATIONAL_ID_EG": r'\b[23]\d{13}\b',

            # Arabic phone numbers (various countries)
            # International format: +966, +971, +20, etc.
            # Local format: 05xxxxxxxx (Saudi), 05xxxxxxxx (UAE)
            "PHONE_AR": r'(?:\+?(?:966|971|20|965|974|973|968|962|961|964))?\s?0?[1-9]\d{7,9}\b',

            # IBAN for Arabic countries
            # Various formats: SA, AE, EG, KW, QA, BH, etc.
            "IBAN_AR": r'\b(?:SA|AE|EG|KW|QA|BH|OM|JO|LB)\d{2}[A-Z0-9\s]{15,30}\b',

            # Arabic passport numbers
            # Format varies by country, typically letter + digits
            "PASSPORT_AR": r'\b[A-Z]\d{7,9}\b',

            # Arabic names (pattern for common Arabic name structure)
            # Matches sequences of Arabic characters that form names
            "ARABIC_NAME": r'[\u0621-\u064A\u0660-\u0669]{2,}\s+[\u0621-\u064A\u0660-\u0669]{2,}(?:\s+[\u0621-\u064A\u0660-\u0669]{2,})?'
        }

    def get_redaction_policy(self) -> RedactionPolicy:
        """
        Get Arabic-specific redaction policy.

        Returns:
            RedactionPolicy for Arabic entities
        """
        return RedactionPolicy(
            full_redaction=[
                "NATIONAL_ID_SA",
                "NATIONAL_ID_AE",
                "NATIONAL_ID_EG",
                "PASSPORT_AR"
            ],
            partial_redaction={
                "PHONE_AR": 4,    # Show last 4 digits
                "IBAN_AR": 4      # Show last 4 characters
            },
            preserve_format=["PHONE_AR"],
            mask_char="X",
            preserve_length=True,
            case_sensitive=False
        )

    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get validation rules for Arabic entities.

        Returns:
            Dictionary of entity_type -> validation function
        """
        return {
            "NATIONAL_ID_SA": self._validate_saudi_id,
            "NATIONAL_ID_AE": self._validate_uae_id,
            "NATIONAL_ID_EG": self._validate_egyptian_id,
            "PHONE_AR": self._validate_arabic_phone
        }

    def get_context_patterns(self) -> Dict[str, List[str]]:
        """
        Get Arabic context keywords (in both Arabic and English).

        Returns:
            Dictionary of entity_type -> context keywords
        """
        return {
            "NATIONAL_ID_SA": [
                "هوية", "رقم هوية", "بطاقة", "هوية وطنية",
                "national id", "id number", "identity"
            ],
            "NATIONAL_ID_AE": [
                "هوية إماراتية", "رقم الهوية",
                "emirates id", "id number"
            ],
            "NATIONAL_ID_EG": [
                "رقم قومي", "بطاقة شخصية",
                "national id", "egyptian id"
            ],
            "PHONE_AR": [
                "هاتف", "جوال", "موبايل", "رقم",
                "phone", "mobile", "tel", "contact"
            ],
            "IBAN_AR": [
                "آيبان", "حساب بنكي", "رقم حساب",
                "iban", "bank account", "account number"
            ],
            "PASSPORT_AR": [
                "جواز", "جواز سفر", "رقم جواز",
                "passport", "passport number"
            ]
        }

    def get_false_positive_patterns(self) -> Dict[str, List[str]]:
        """
        Get false positive patterns to exclude.

        Returns:
            Dictionary of entity_type -> false positive patterns
        """
        return {
            "NATIONAL_ID_SA": [
                r'0{10}',  # All zeros
                r'1{10}',  # All ones
            ],
            "NATIONAL_ID_EG": [
                r'0{14}',  # All zeros
            ],
            "PHONE_AR": [
                r'0{9,10}',  # All zeros
            ]
        }

    def _validate_saudi_id(self, text: str) -> bool:
        """
        Validate Saudi National ID.

        Args:
            text: National ID

        Returns:
            True if valid
        """
        # Remove spaces and dashes
        national_id = text.replace(' ', '').replace('-', '')

        # Must be 10 digits
        if len(national_id) != 10:
            return False

        # Must be numeric
        if not national_id.isdigit():
            return False

        # Must start with 1 (Saudi) or 2 (Resident)
        if national_id[0] not in ['1', '2']:
            return False

        return True

    def _validate_uae_id(self, text: str) -> bool:
        """
        Validate UAE National ID.

        Args:
            text: National ID

        Returns:
            True if valid
        """
        # Remove spaces and dashes
        national_id = text.replace(' ', '').replace('-', '')

        # Must be 15 digits
        if len(national_id) != 15:
            return False

        # Must be numeric
        if not national_id.isdigit():
            return False

        # Must start with 784 (UAE country code)
        if not national_id.startswith('784'):
            return False

        return True

    def _validate_egyptian_id(self, text: str) -> bool:
        """
        Validate Egyptian National ID.

        Args:
            text: National ID

        Returns:
            True if valid
        """
        # Remove spaces
        national_id = text.replace(' ', '')

        # Must be 14 digits
        if len(national_id) != 14:
            return False

        # Must be numeric
        if not national_id.isdigit():
            return False

        # First digit: 2 (born in 1900s) or 3 (born in 2000s)
        if national_id[0] not in ['2', '3']:
            return False

        # Extract and validate date of birth (positions 1-7: YYMMDD)
        year = national_id[1:3]
        month = national_id[3:5]
        day = national_id[5:7]

        try:
            year_int = int(year)
            month_int = int(month)
            day_int = int(day)

            # Basic date validation
            if not (1 <= month_int <= 12):
                return False
            if not (1 <= day_int <= 31):
                return False
        except ValueError:
            return False

        return True

    def _validate_arabic_phone(self, text: str) -> bool:
        """
        Validate Arabic phone number.

        Args:
            text: Phone number

        Returns:
            True if valid
        """
        # Remove common separators
        phone = text.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')

        # Remove country code prefix if present
        if phone.startswith('+'):
            phone = phone[1:]

        # Must be 9-15 digits (varies by country)
        if not (9 <= len(phone) <= 15):
            return False

        # Must be numeric
        if not phone.isdigit():
            return False

        return True

    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        import unicodedata

        # Remove diacritics (tashkeel)
        normalized = ''.join(
            char for char in text
            if unicodedata.category(char) != 'Mn'
        )

        # Normalize Arabic letters
        replacements = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alef variations
            'ة': 'ه',                        # Taa marbuta to haa
            'ى': 'ي'                         # Alef maksura to yaa
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        return normalized

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Arabic text.

        Arabic requires special tokenization due to:
        - Right-to-left writing
        - Connected letters
        - No explicit word boundaries

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Split on whitespace and Arabic punctuation
        import re
        tokens = re.split(r'[\s\u060C\u061B\u061F\u0640]+', text)

        # Remove empty tokens
        tokens = [t for t in tokens if t.strip()]

        return tokens


def register_language_pack() -> LanguagePack:
    """
    Register function called by language registry.

    Returns:
        ArabicLanguagePack instance
    """
    return ArabicLanguagePack()
