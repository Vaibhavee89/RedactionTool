"""
Custom Regex Detector Plugin

Flexible plugin for detecting custom patterns via user-configurable regex patterns.
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    PluginMetadata,
    PluginType,
    DetectedEntity
)


class CustomRegexDetectorPlugin(DetectorPlugin):
    """
    Flexible detector for custom regex patterns.

    Patterns are loaded from a JSON configuration file, allowing users to
    define their own detection patterns without writing code.

    Config file format (custom_patterns.json):
    {
        "patterns": {
            "CUSTOM_EMPLOYEE_ID": {
                "pattern": "EMP-[0-9]{6}",
                "name": "Employee ID",
                "confidence": 0.9,
                "case_sensitive": false
            },
            "CUSTOM_PROJECT_CODE": {
                "pattern": "PROJ-[A-Z]{3}-[0-9]{4}",
                "name": "Project Code",
                "confidence": 0.85
            }
        }
    }
    """

    DEFAULT_CONFIG_PATH = "plugins/detectors/custom_regex_detector/custom_patterns.json"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize custom regex detector.

        Args:
            config_path: Path to config file (uses default if not provided)
        """
        super().__init__()
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.patterns = {}
        self.compiled_patterns = {}

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        # Get entity types from config
        entity_types = list(self.patterns.keys()) if self.patterns else []

        return PluginMetadata(
            name="custom_regex_detector",
            version="1.0.0",
            description="Flexible detector for user-defined regex patterns",
            author="RedactionTool Team",
            plugin_type=PluginType.DETECTOR,
            supported_entity_types=entity_types,
            supported_languages=["*"],  # Can be used for any language
            priority=5,  # Highest priority (user-defined patterns are most specific)
            dependencies=[],
            min_confidence=0.5,
            max_confidence=1.0,
            requires_network=False,
            timeout_seconds=15
        )

    def initialize(self) -> None:
        """Initialize plugin by loading and compiling patterns."""
        super().initialize()

        # Load patterns from config file
        self.patterns = self._load_patterns()

        # Compile regex patterns
        for entity_type, config in self.patterns.items():
            flags = 0 if config.get('case_sensitive', False) else re.IGNORECASE
            self.compiled_patterns[entity_type] = re.compile(
                config['pattern'],
                flags=flags
            )

    def _load_patterns(self) -> Dict[str, Any]:
        """
        Load patterns from JSON config file.

        Returns:
            Dictionary of pattern configurations
        """
        config_file = Path(self.config_path)

        # Create default config if doesn't exist
        if not config_file.exists():
            self._create_default_config(config_file)

        # Load config
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('patterns', {})
        except Exception as e:
            print(f"Error loading custom patterns: {str(e)}")
            return {}

    def _create_default_config(self, config_file: Path) -> None:
        """
        Create default configuration file with examples.

        Args:
            config_file: Path to config file
        """
        default_config = {
            "patterns": {
                "CUSTOM_EMPLOYEE_ID": {
                    "pattern": r"EMP-[0-9]{6}",
                    "name": "Employee ID",
                    "description": "Company employee identifier",
                    "confidence": 0.9,
                    "case_sensitive": False,
                    "examples": ["EMP-123456", "EMP-987654"]
                },
                "CUSTOM_PROJECT_CODE": {
                    "pattern": r"PROJ-[A-Z]{3}-[0-9]{4}",
                    "name": "Project Code",
                    "description": "Internal project identifier",
                    "confidence": 0.85,
                    "case_sensitive": True,
                    "examples": ["PROJ-ABC-1234", "PROJ-XYZ-5678"]
                },
                "CUSTOM_TICKET_NUMBER": {
                    "pattern": r"TICKET-[0-9]{8}",
                    "name": "Support Ticket",
                    "description": "Support ticket number",
                    "confidence": 0.8,
                    "case_sensitive": False,
                    "examples": ["TICKET-12345678"]
                }
            },
            "metadata": {
                "version": "1.0",
                "description": "Custom regex patterns for organization-specific entities",
                "last_updated": "2026-02-05"
            }
        }

        # Create directory if needed
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedEntity]:
        """
        Detect custom patterns in text.

        Args:
            text: Text to analyze
            language: Language code (not used)
            entity_types: Specific entity types to detect
            context: Additional context

        Returns:
            List of detected entities
        """
        if not text or not self.patterns:
            return []

        entities = []

        # Filter patterns by requested entity types
        patterns_to_check = self.patterns.items()
        if entity_types:
            patterns_to_check = [
                (et, config) for et, config in patterns_to_check
                if et in entity_types
            ]

        # Detect each pattern
        for entity_type, config in patterns_to_check:
            pattern = self.compiled_patterns.get(entity_type)
            if not pattern:
                continue

            for match in pattern.finditer(text):
                matched_text = match.group(0)
                start = match.start()
                end = match.end()

                # Get confidence from config
                confidence = config.get('confidence', 0.8)

                entities.append(DetectedEntity(
                    entity_type=entity_type,
                    text=matched_text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    source="custom_regex_detector",
                    metadata={
                        'pattern_name': config.get('name', entity_type),
                        'description': config.get('description', ''),
                        'case_sensitive': config.get('case_sensitive', False)
                    }
                ))

        return entities

    def add_pattern(
        self,
        entity_type: str,
        pattern: str,
        name: str,
        confidence: float = 0.8,
        case_sensitive: bool = False,
        description: str = "",
        examples: Optional[List[str]] = None
    ) -> None:
        """
        Add a new pattern to the detector.

        Args:
            entity_type: Unique entity type identifier
            pattern: Regex pattern string
            name: Human-readable name
            confidence: Detection confidence (0.0-1.0)
            case_sensitive: Whether pattern is case-sensitive
            description: Pattern description
            examples: Example matches

        Raises:
            ValueError: If pattern is invalid or entity_type already exists
        """
        # Validate pattern
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {str(e)}")

        # Check if entity type already exists
        if entity_type in self.patterns:
            raise ValueError(f"Entity type already exists: {entity_type}")

        # Add pattern
        self.patterns[entity_type] = {
            'pattern': pattern,
            'name': name,
            'confidence': confidence,
            'case_sensitive': case_sensitive,
            'description': description,
            'examples': examples or []
        }

        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        self.compiled_patterns[entity_type] = re.compile(pattern, flags=flags)

        # Save to config file
        self._save_patterns()

    def remove_pattern(self, entity_type: str) -> None:
        """
        Remove a pattern from the detector.

        Args:
            entity_type: Entity type to remove
        """
        if entity_type in self.patterns:
            del self.patterns[entity_type]
            del self.compiled_patterns[entity_type]
            self._save_patterns()

    def _save_patterns(self) -> None:
        """Save patterns to config file."""
        config = {
            'patterns': self.patterns,
            'metadata': {
                'version': '1.0',
                'description': 'Custom regex patterns for organization-specific entities',
                'last_updated': '2026-02-05'
            }
        }

        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def get_pattern_info(self, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific pattern.

        Args:
            entity_type: Entity type

        Returns:
            Pattern configuration or None
        """
        return self.patterns.get(entity_type)

    def list_patterns(self) -> List[Dict[str, Any]]:
        """
        List all configured patterns.

        Returns:
            List of pattern configurations
        """
        return [
            {
                'entity_type': entity_type,
                **config
            }
            for entity_type, config in self.patterns.items()
        ]

    def validate(self) -> Dict[str, Any]:
        """
        Validate plugin configuration.

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        # Check config file exists
        if not Path(self.config_path).exists():
            warnings.append(f"Config file not found: {self.config_path}")

        # Check patterns
        if not self.patterns:
            warnings.append("No patterns configured")

        # Validate each pattern
        for entity_type, config in self.patterns.items():
            # Check required fields
            if 'pattern' not in config:
                errors.append(f"Missing pattern for {entity_type}")
                continue

            # Validate regex
            try:
                re.compile(config['pattern'])
            except re.error as e:
                errors.append(f"Invalid pattern for {entity_type}: {str(e)}")

            # Check confidence range
            confidence = config.get('confidence', 0.8)
            if not (0.0 <= confidence <= 1.0):
                errors.append(f"Invalid confidence for {entity_type}: {confidence}")

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
        CustomRegexDetectorPlugin instance
    """
    return CustomRegexDetectorPlugin()
