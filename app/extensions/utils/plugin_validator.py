"""
Plugin Validator

Utilities for validating detector plugins before registration.
"""

import re
import importlib
from typing import Dict, List, Any
import logging

from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    PluginMetadata,
    PluginValidationError
)

logger = logging.getLogger(__name__)


class PluginValidator:
    """Validator for detector plugins."""

    @staticmethod
    def validate_plugin(plugin: DetectorPlugin) -> Dict[str, Any]:
        """
        Comprehensive plugin validation.

        Args:
            plugin: DetectorPlugin instance to validate

        Returns:
            Dictionary with:
                - valid (bool): Whether plugin passes all checks
                - errors (List[str]): Critical validation errors
                - warnings (List[str]): Non-critical warnings
        """
        errors = []
        warnings = []

        # Validate metadata
        try:
            metadata = plugin.get_metadata()
            metadata_validation = PluginValidator._validate_metadata(metadata)
            errors.extend(metadata_validation['errors'])
            warnings.extend(metadata_validation['warnings'])
        except Exception as e:
            errors.append(f"Failed to get metadata: {str(e)}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Validate dependencies
        dependency_validation = PluginValidator._validate_dependencies(metadata)
        errors.extend(dependency_validation['errors'])
        warnings.extend(dependency_validation['warnings'])

        # Validate interface implementation
        interface_validation = PluginValidator._validate_interface(plugin)
        errors.extend(interface_validation['errors'])
        warnings.extend(interface_validation['warnings'])

        # Run plugin's own validation
        try:
            plugin_validation = plugin.validate()
            if not plugin_validation.get('valid', False):
                errors.extend(plugin_validation.get('errors', []))
            warnings.extend(plugin_validation.get('warnings', []))
        except Exception as e:
            errors.append(f"Plugin validation method failed: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    @staticmethod
    def _validate_metadata(metadata: PluginMetadata) -> Dict[str, List[str]]:
        """Validate plugin metadata."""
        errors = []
        warnings = []

        # Validate name
        if not metadata.name:
            errors.append("Plugin name is required")
        elif not re.match(r'^[a-z0-9_]+$', metadata.name):
            errors.append(
                f"Invalid plugin name '{metadata.name}': "
                "must be lowercase alphanumeric with underscores"
            )

        # Validate version
        if not metadata.version:
            errors.append("Plugin version is required")
        elif not re.match(r'^\d+\.\d+\.\d+$', metadata.version):
            warnings.append(
                f"Version '{metadata.version}' does not follow semantic versioning"
            )

        # Validate entity types
        if not metadata.supported_entity_types:
            warnings.append("No supported entity types specified")

        # Validate languages
        if not metadata.supported_languages:
            warnings.append("No supported languages specified")

        # Validate priority
        if metadata.priority < 1 or metadata.priority > 10:
            errors.append(f"Priority must be 1-10, got: {metadata.priority}")

        # Validate confidence range
        if metadata.min_confidence < 0.0 or metadata.min_confidence > 1.0:
            errors.append(f"min_confidence must be 0.0-1.0, got: {metadata.min_confidence}")
        if metadata.max_confidence < 0.0 or metadata.max_confidence > 1.0:
            errors.append(f"max_confidence must be 0.0-1.0, got: {metadata.max_confidence}")
        if metadata.min_confidence > metadata.max_confidence:
            errors.append("min_confidence cannot exceed max_confidence")

        # Validate timeout
        if metadata.timeout_seconds <= 0:
            errors.append(f"timeout_seconds must be positive, got: {metadata.timeout_seconds}")
        elif metadata.timeout_seconds > 300:
            warnings.append(
                f"Long timeout ({metadata.timeout_seconds}s) may affect performance"
            )

        # Validate description
        if not metadata.description:
            warnings.append("No description provided")

        # Validate author
        if not metadata.author:
            warnings.append("No author specified")

        return {"errors": errors, "warnings": warnings}

    @staticmethod
    def _validate_dependencies(metadata: PluginMetadata) -> Dict[str, List[str]]:
        """Validate plugin dependencies."""
        errors = []
        warnings = []

        for dependency in metadata.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                errors.append(f"Missing required dependency: {dependency}")

        if metadata.requires_network:
            warnings.append("Plugin requires network access")

        return {"errors": errors, "warnings": warnings}

    @staticmethod
    def _validate_interface(plugin: DetectorPlugin) -> Dict[str, List[str]]:
        """Validate plugin implements required interface."""
        errors = []
        warnings = []

        # Check required methods
        required_methods = ['get_metadata', 'detect', 'validate']
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                errors.append(f"Missing required method: {method_name}")
            elif not callable(getattr(plugin, method_name)):
                errors.append(f"Method not callable: {method_name}")

        # Check optional methods
        optional_methods = ['initialize', 'cleanup']
        for method_name in optional_methods:
            if hasattr(plugin, method_name) and not callable(getattr(plugin, method_name)):
                warnings.append(f"Optional method not callable: {method_name}")

        return {"errors": errors, "warnings": warnings}

    @staticmethod
    def validate_detection_output(
        entities: List[Any],
        text: str,
        metadata: PluginMetadata
    ) -> Dict[str, Any]:
        """
        Validate plugin detection output.

        Args:
            entities: Detected entities from plugin
            text: Original text that was analyzed
            metadata: Plugin metadata

        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []

        if not isinstance(entities, list):
            errors.append(f"Expected list of entities, got: {type(entities)}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        for i, entity in enumerate(entities):
            # Check entity has required attributes
            required_attrs = ['entity_type', 'text', 'start', 'end', 'confidence']
            for attr in required_attrs:
                if not hasattr(entity, attr):
                    errors.append(f"Entity {i} missing required attribute: {attr}")
                    continue

            # Validate entity type
            if entity.entity_type not in metadata.supported_entity_types:
                warnings.append(
                    f"Entity {i} type '{entity.entity_type}' not in supported types"
                )

            # Validate positions
            if entity.start < 0 or entity.end < 0:
                errors.append(f"Entity {i} has negative position")
            elif entity.start >= entity.end:
                errors.append(f"Entity {i} has invalid position range")
            elif entity.end > len(text):
                errors.append(f"Entity {i} position exceeds text length")
            else:
                # Validate extracted text matches positions
                extracted = text[entity.start:entity.end]
                if extracted != entity.text:
                    warnings.append(
                        f"Entity {i} text mismatch: "
                        f"expected '{extracted}', got '{entity.text}'"
                    )

            # Validate confidence
            if entity.confidence < metadata.min_confidence:
                warnings.append(
                    f"Entity {i} confidence {entity.confidence} below min {metadata.min_confidence}"
                )
            elif entity.confidence > metadata.max_confidence:
                warnings.append(
                    f"Entity {i} confidence {entity.confidence} exceeds max {metadata.max_confidence}"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
