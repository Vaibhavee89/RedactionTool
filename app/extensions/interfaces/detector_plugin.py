"""
Detector Plugin Interface

This module defines the base interface for all detector plugins in the RedactionTool
extensibility framework. Plugins enable custom entity detection without modifying core code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from datetime import datetime


class PluginType(Enum):
    """Types of plugins supported by the framework."""
    DETECTOR = "detector"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    LLM_PROVIDER = "llm_provider"


@dataclass
class PluginMetadata:
    """
    Metadata describing a plugin's capabilities and requirements.

    Attributes:
        name: Unique plugin identifier (lowercase, alphanumeric + underscore)
        version: Semantic version string (e.g., "1.0.0")
        description: Human-readable plugin description
        author: Plugin author name
        plugin_type: Type of plugin (default: DETECTOR)
        supported_entity_types: List of entity types this plugin detects
        supported_languages: List of language codes or ["*"] for all languages
        priority: Plugin priority for conflict resolution (1-10, higher = higher priority)
        dependencies: List of required Python packages
        min_confidence: Minimum confidence score this plugin returns (0.0-1.0)
        max_confidence: Maximum confidence score this plugin returns (0.0-1.0)
        requires_network: Whether plugin needs network access
        timeout_seconds: Maximum execution time before timeout
        enabled: Whether plugin is currently enabled
    """
    name: str
    version: str
    description: str = ""
    author: str = ""
    plugin_type: PluginType = PluginType.DETECTOR
    supported_entity_types: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(default_factory=lambda: ["*"])
    priority: int = 3  # Same as regex default
    dependencies: List[str] = field(default_factory=list)
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    requires_network: bool = False
    timeout_seconds: int = 30
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name or not self.name.replace('_', '').isalnum():
            raise ValueError(f"Invalid plugin name: {self.name}")
        if self.priority < 1 or self.priority > 10:
            raise ValueError(f"Priority must be 1-10, got: {self.priority}")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be 0.0-1.0, got: {self.min_confidence}")
        if not (0.0 <= self.max_confidence <= 1.0):
            raise ValueError(f"max_confidence must be 0.0-1.0, got: {self.max_confidence}")
        if self.min_confidence > self.max_confidence:
            raise ValueError("min_confidence cannot exceed max_confidence")
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.updated_at:
            self.updated_at = datetime.utcnow()

    def supports_language(self, language: str) -> bool:
        """Check if plugin supports a given language."""
        return "*" in self.supported_languages or language in self.supported_languages

    def supports_entity_type(self, entity_type: str) -> bool:
        """Check if plugin supports a given entity type."""
        return entity_type in self.supported_entity_types


@dataclass
class DetectedEntity:
    """
    Represents a detected entity with standardized format.

    Attributes:
        entity_type: Type of entity (e.g., "EMAIL", "PHONE", "CUSTOM_TYPE")
        text: The actual text of the detected entity
        start: Start position in original text
        end: End position in original text
        confidence: Detection confidence (0.0-1.0)
        source: Name of plugin that detected this entity
        metadata: Additional plugin-specific metadata
    """
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entity after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got: {self.confidence}")
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end positions must be non-negative")
        if self.start >= self.end:
            raise ValueError("Start position must be less than end position")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary format."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }


class DetectorPlugin(ABC):
    """
    Abstract base class for all detector plugins.

    Plugins must implement:
    - get_metadata(): Return plugin metadata
    - detect(): Perform entity detection
    - validate(): Validate plugin configuration

    Optional:
    - initialize(): Setup plugin (load models, compile patterns, etc.)
    - cleanup(): Cleanup resources
    """

    def __init__(self):
        """Initialize plugin instance."""
        self._initialized = False
        self._metadata = None

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Return plugin metadata describing capabilities.

        Returns:
            PluginMetadata instance
        """
        pass

    @abstractmethod
    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedEntity]:
        """
        Detect entities in text.

        Args:
            text: Text to analyze
            language: Language code (e.g., 'en', 'fr', 'de')
            entity_types: Specific entity types to detect (None = all)
            context: Additional context for detection (optional)

        Returns:
            List of DetectedEntity instances

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If detection fails
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Validate plugin configuration and dependencies.

        Returns:
            Dictionary with:
                - valid (bool): Whether plugin is valid
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings
        """
        pass

    def initialize(self) -> None:
        """
        Initialize plugin (load models, compile patterns, etc.).

        Called once after registration. Override if needed.
        """
        self._initialized = True

    def cleanup(self) -> None:
        """
        Cleanup plugin resources.

        Called when plugin is disabled or unregistered. Override if needed.
        """
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    def get_supported_entity_types(self) -> Set[str]:
        """Get set of supported entity types."""
        if not self._metadata:
            self._metadata = self.get_metadata()
        return set(self._metadata.supported_entity_types)

    def get_supported_languages(self) -> Set[str]:
        """Get set of supported languages."""
        if not self._metadata:
            self._metadata = self.get_metadata()
        return set(self._metadata.supported_languages)

    def __repr__(self) -> str:
        """String representation of plugin."""
        metadata = self.get_metadata()
        return f"<{self.__class__.__name__} name='{metadata.name}' version='{metadata.version}'>"


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass


class PluginExecutionError(Exception):
    """Raised when plugin execution fails."""
    pass


class PluginTimeoutError(Exception):
    """Raised when plugin execution exceeds timeout."""
    pass
