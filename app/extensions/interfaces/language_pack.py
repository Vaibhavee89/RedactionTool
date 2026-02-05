"""
Language Pack Interface

This module defines the interface for language-specific detection patterns and policies.
Language packs enable easy addition of multi-language support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class Script(Enum):
    """Writing scripts supported by language packs."""
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    DEVANAGARI = "devanagari"
    CJK = "cjk"  # Chinese, Japanese, Korean
    GREEK = "greek"
    THAI = "thai"
    OTHER = "other"


@dataclass
class LanguagePackMetadata:
    """
    Metadata describing a language pack's capabilities.

    Attributes:
        language_code: ISO 639-1 two-letter language code (e.g., 'fr', 'de', 'ar')
        language_name: Human-readable language name (e.g., 'French', 'German')
        script: Primary writing script used
        supported_entity_types: List of entity types supported by this pack
        description: Description of language pack
        author: Pack author name
        version: Semantic version string
        country_codes: Optional list of country codes (ISO 3166-1 alpha-2)
        right_to_left: Whether language is right-to-left
        requires_special_tokenization: Whether language needs special tokenization
    """
    language_code: str
    language_name: str
    script: Script
    supported_entity_types: List[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    country_codes: List[str] = field(default_factory=list)
    right_to_left: bool = False
    requires_special_tokenization: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.language_code or len(self.language_code) != 2:
            raise ValueError(f"Invalid language code: {self.language_code}")
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.updated_at:
            self.updated_at = datetime.utcnow()


@dataclass
class RedactionPolicy:
    """
    Language-specific redaction policy configuration.

    Attributes:
        full_redaction: Entity types that should be fully redacted (e.g., [REDACTED])
        partial_redaction: Entity types that should be partially redacted (show last N chars)
        preserve_format: Entity types where format should be preserved (e.g., XXX-XX-1234)
        mask_char: Character to use for masking (default: 'X')
        preserve_length: Whether to preserve original length in redaction
        case_sensitive: Whether entity matching is case-sensitive
    """
    full_redaction: List[str] = field(default_factory=list)
    partial_redaction: Dict[str, int] = field(default_factory=dict)  # entity_type -> chars_to_show
    preserve_format: List[str] = field(default_factory=list)
    mask_char: str = "X"
    preserve_length: bool = True
    case_sensitive: bool = False


class LanguagePack(ABC):
    """
    Abstract base class for language-specific detection packs.

    Language packs provide:
    - Regex patterns for language-specific entities
    - Named entity recognition patterns
    - Redaction policies
    - Validation rules
    """

    def __init__(self):
        """Initialize language pack instance."""
        self._initialized = False
        self._metadata = None

    @abstractmethod
    def get_metadata(self) -> LanguagePackMetadata:
        """
        Return language pack metadata.

        Returns:
            LanguagePackMetadata instance
        """
        pass

    @abstractmethod
    def get_regex_patterns(self) -> Dict[str, str]:
        """
        Get regex patterns for entity detection.

        Returns:
            Dictionary mapping entity_type -> regex pattern
            Example:
                {
                    "PHONE_FR": r'\b0[1-9](?:[\s.-]?\d{2}){4}\b',
                    "INSEE": r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',
                    "IBAN_FR": r'\bFR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b'
                }
        """
        pass

    @abstractmethod
    def get_redaction_policy(self) -> RedactionPolicy:
        """
        Get language-specific redaction policy.

        Returns:
            RedactionPolicy instance
        """
        pass

    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get validation rules for detected entities (optional).

        Returns:
            Dictionary mapping entity_type -> validation function
            Example:
                {
                    "INSEE": lambda text: validate_insee_checksum(text),
                    "IBAN_FR": lambda text: validate_iban_checksum(text)
                }
        """
        return {}

    def get_context_patterns(self) -> Dict[str, List[str]]:
        """
        Get context keywords that indicate presence of entities (optional).

        Returns:
            Dictionary mapping entity_type -> list of context keywords
            Example:
                {
                    "PHONE_FR": ["téléphone", "tél.", "portable", "mobile"],
                    "INSEE": ["numéro de sécurité sociale", "INSEE"]
                }
        """
        return {}

    def get_false_positive_patterns(self) -> Dict[str, List[str]]:
        """
        Get patterns that should NOT be detected as entities (optional).

        Returns:
            Dictionary mapping entity_type -> list of false positive patterns
            Example:
                {
                    "PHONE_FR": [r'\b0{10}\b', r'\b1{10}\b']  # All zeros, all ones
                }
        """
        return {}

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for this language (optional).

        Override for language-specific normalization (e.g., remove diacritics,
        convert to lowercase, normalize whitespace).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for this language (optional).

        Override for languages requiring special tokenization (e.g., CJK languages).

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return text.split()

    def validate(self) -> Dict[str, Any]:
        """
        Validate language pack configuration.

        Returns:
            Dictionary with:
                - valid (bool): Whether pack is valid
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings
        """
        errors = []
        warnings = []

        try:
            metadata = self.get_metadata()
            if not metadata.supported_entity_types:
                warnings.append("No supported entity types defined")
        except Exception as e:
            errors.append(f"Failed to get metadata: {str(e)}")

        try:
            patterns = self.get_regex_patterns()
            if not patterns:
                warnings.append("No regex patterns defined")

            # Validate regex patterns compile
            import re
            for entity_type, pattern in patterns.items():
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"Invalid regex for {entity_type}: {str(e)}")
        except Exception as e:
            errors.append(f"Failed to get regex patterns: {str(e)}")

        try:
            policy = self.get_redaction_policy()
            if not policy:
                warnings.append("No redaction policy defined")
        except Exception as e:
            errors.append(f"Failed to get redaction policy: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def initialize(self) -> None:
        """
        Initialize language pack (compile patterns, load resources, etc.).

        Called once after registration. Override if needed.
        """
        self._initialized = True

    def cleanup(self) -> None:
        """
        Cleanup language pack resources.

        Called when pack is disabled or unregistered. Override if needed.
        """
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if language pack is initialized."""
        return self._initialized

    def __repr__(self) -> str:
        """String representation of language pack."""
        metadata = self.get_metadata()
        return f"<{self.__class__.__name__} language='{metadata.language_code}' version='{metadata.version}'>"


class LanguagePackValidationError(Exception):
    """Raised when language pack validation fails."""
    pass
