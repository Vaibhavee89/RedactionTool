"""
Medical Codes Detector Plugin

Detects medical classification codes like ICD-10, CPT, and NDC codes.
"""

import re
from typing import List, Dict, Any, Optional

from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    PluginMetadata,
    PluginType,
    DetectedEntity
)


class MedicalCodesDetectorPlugin(DetectorPlugin):
    """
    Detector for medical classification codes.

    Supports:
    - ICD-10 codes (International Classification of Diseases)
    - CPT codes (Current Procedural Terminology)
    - NDC codes (National Drug Code)
    - LOINC codes (Logical Observation Identifiers Names and Codes)
    - HCPCS codes (Healthcare Common Procedure Coding System)
    """

    # Medical code patterns
    MEDICAL_CODE_PATTERNS = {
        'MEDICAL_ICD10': {
            'pattern': r'\b[A-TV-Z][0-9]{2}\.?[0-9A-TV-Z]{0,4}\b',
            'name': 'ICD-10 Code',
            'confidence': 0.80,
            'description': 'International Classification of Diseases, 10th Revision'
        },
        'MEDICAL_CPT': {
            'pattern': r'\b[0-9]{5}[A-Z]?\b',
            'name': 'CPT Code',
            'confidence': 0.75,
            'description': 'Current Procedural Terminology code'
        },
        'MEDICAL_NDC': {
            'pattern': r'\b[0-9]{4,5}-[0-9]{3,4}-[0-9]{1,2}\b|\b[0-9]{11}\b',
            'name': 'NDC Code',
            'confidence': 0.85,
            'description': 'National Drug Code'
        },
        'MEDICAL_LOINC': {
            'pattern': r'\b[0-9]{4,5}-[0-9]\b',
            'name': 'LOINC Code',
            'confidence': 0.80,
            'description': 'Logical Observation Identifiers Names and Codes'
        },
        'MEDICAL_HCPCS': {
            'pattern': r'\b[A-Z][0-9]{4}\b',
            'name': 'HCPCS Code',
            'confidence': 0.75,
            'description': 'Healthcare Common Procedure Coding System'
        }
    }

    # Context keywords indicating medical codes
    MEDICAL_CONTEXTS = [
        'diagnosis', 'procedure', 'icd', 'cpt', 'code', 'billing',
        'medical', 'health', 'treatment', 'condition', 'disease',
        'surgery', 'medication', 'drug', 'ndc', 'loinc', 'hcpcs',
        'claim', 'insurance', 'patient', 'clinical', 'encounter'
    ]

    # Common false positive patterns to exclude
    FALSE_POSITIVES = {
        'MEDICAL_ICD10': [
            # Exclude codes that look like years
            r'^[12][0-9]{3}$',
            # Exclude single letter codes
            r'^[A-Z]$'
        ],
        'MEDICAL_CPT': [
            # Exclude ZIP codes (5 digits)
            r'^[0-9]{5}$'  # Will be filtered by context
        ]
    }

    def __init__(self):
        """Initialize medical codes detector plugin."""
        super().__init__()
        self.compiled_patterns = {}
        self.false_positive_patterns = {}

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="medical_codes_detector",
            version="1.0.0",
            description="Detects medical classification codes (ICD-10, CPT, NDC, LOINC, HCPCS)",
            author="RedactionTool Team",
            plugin_type=PluginType.DETECTOR,
            supported_entity_types=list(self.MEDICAL_CODE_PATTERNS.keys()),
            supported_languages=["en"],  # Medical codes are primarily English
            priority=4,  # Higher than standard regex
            dependencies=[],
            min_confidence=0.7,
            max_confidence=0.9,
            requires_network=False,
            timeout_seconds=10
        )

    def initialize(self) -> None:
        """Initialize plugin by compiling patterns."""
        super().initialize()

        # Compile detection patterns
        for entity_type, config in self.MEDICAL_CODE_PATTERNS.items():
            self.compiled_patterns[entity_type] = re.compile(config['pattern'])

        # Compile false positive patterns
        for entity_type, patterns in self.FALSE_POSITIVES.items():
            self.false_positive_patterns[entity_type] = [
                re.compile(pattern) for pattern in patterns
            ]

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedEntity]:
        """
        Detect medical codes in text.

        Args:
            text: Text to analyze
            language: Language code
            entity_types: Specific medical code types to detect
            context: Additional context

        Returns:
            List of detected medical codes
        """
        if not text:
            return []

        entities = []

        # Filter patterns by requested entity types
        patterns_to_check = self.MEDICAL_CODE_PATTERNS.items()
        if entity_types:
            patterns_to_check = [
                (et, config) for et, config in patterns_to_check
                if et in entity_types
            ]

        # Detect each medical code type
        for entity_type, config in patterns_to_check:
            pattern = self.compiled_patterns[entity_type]

            for match in pattern.finditer(text):
                code = match.group(0)
                start = match.start()
                end = match.end()

                # Check false positives
                if self._is_false_positive(code, entity_type):
                    continue

                # Validate code format
                if not self._validate_code(code, entity_type):
                    continue

                # Check medical context
                has_context = self._has_medical_context(text, start, end)

                # Adjust confidence based on context
                confidence = config['confidence']
                if has_context:
                    confidence = min(confidence + 0.10, 0.9)
                else:
                    # Lower confidence if no medical context
                    confidence = max(confidence - 0.15, 0.6)

                # Skip if confidence too low
                if confidence < 0.6:
                    continue

                entities.append(DetectedEntity(
                    entity_type=entity_type,
                    text=code,
                    start=start,
                    end=end,
                    confidence=confidence,
                    source="medical_codes_detector",
                    metadata={
                        'code_name': config['name'],
                        'description': config['description'],
                        'has_medical_context': has_context
                    }
                ))

        return entities

    def _is_false_positive(self, code: str, entity_type: str) -> bool:
        """
        Check if code matches false positive patterns.

        Args:
            code: Code string
            entity_type: Entity type

        Returns:
            True if false positive
        """
        if entity_type not in self.false_positive_patterns:
            return False

        for pattern in self.false_positive_patterns[entity_type]:
            if pattern.match(code):
                return True

        return False

    def _validate_code(self, code: str, entity_type: str) -> bool:
        """
        Additional validation for medical codes.

        Args:
            code: Code string
            entity_type: Entity type

        Returns:
            True if code passes validation
        """
        if entity_type == 'MEDICAL_ICD10':
            # ICD-10 codes: Letter + 2 digits + optional subcategory
            # Length: 3-7 characters
            return 3 <= len(code.replace('.', '')) <= 7

        elif entity_type == 'MEDICAL_CPT':
            # CPT codes: 5 digits, optionally followed by modifier letter
            return 5 <= len(code) <= 6

        elif entity_type == 'MEDICAL_NDC':
            # NDC codes: 11 digits or formatted with dashes
            digits_only = code.replace('-', '')
            return len(digits_only) == 11 and digits_only.isdigit()

        elif entity_type == 'MEDICAL_LOINC':
            # LOINC codes: 4-5 digits + dash + check digit
            parts = code.split('-')
            if len(parts) != 2:
                return False
            return 4 <= len(parts[0]) <= 5 and len(parts[1]) == 1

        elif entity_type == 'MEDICAL_HCPCS':
            # HCPCS codes: Letter + 4 digits
            return len(code) == 5 and code[0].isalpha() and code[1:].isdigit()

        return True

    def _has_medical_context(self, text: str, start: int, end: int, window: int = 100) -> bool:
        """
        Check if code has medical context nearby.

        Args:
            text: Full text
            start: Start position
            end: End position
            window: Context window size

        Returns:
            True if medical context found
        """
        # Extract context
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()

        # Check for medical keywords
        return any(keyword in context for keyword in self.MEDICAL_CONTEXTS)

    def validate(self) -> Dict[str, Any]:
        """
        Validate plugin configuration.

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        # Check patterns compile
        for entity_type, config in self.MEDICAL_CODE_PATTERNS.items():
            try:
                re.compile(config['pattern'])
            except re.error as e:
                errors.append(f"Invalid pattern for {entity_type}: {str(e)}")

        # Check false positive patterns
        for entity_type, patterns in self.FALSE_POSITIVES.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"Invalid false positive pattern for {entity_type}: {str(e)}")

        # Check metadata
        try:
            metadata = self.get_metadata()
            if not metadata.supported_entity_types:
                warnings.append("No entity types defined")
        except Exception as e:
            errors.append(f"Failed to get metadata: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


def register_plugin() -> DetectorPlugin:
    """
    Register function called by plugin registry.

    Returns:
        MedicalCodesDetectorPlugin instance
    """
    return MedicalCodesDetectorPlugin()
