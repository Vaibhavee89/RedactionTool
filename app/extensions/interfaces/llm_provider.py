"""
LLM Provider Interface

This module defines the interface for LLM-based entity detection and validation.
LLM providers enable context-aware, intelligent PII detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime


class SensitivityLevel(Enum):
    """Sensitivity levels for detected entities."""
    LOW = "low"           # Public information, low risk
    MEDIUM = "medium"     # Sensitive information, moderate risk
    HIGH = "high"         # Highly sensitive information, high risk
    CRITICAL = "critical" # Critical information, maximum risk


@dataclass
class LLMProviderMetadata:
    """
    Metadata describing an LLM provider's capabilities.

    Attributes:
        name: Provider name (e.g., 'openai', 'anthropic', 'ollama')
        version: Provider version
        model_name: Specific model name (e.g., 'gpt-4', 'claude-3-opus')
        description: Human-readable description
        supports_streaming: Whether provider supports streaming responses
        supports_function_calling: Whether provider supports function calling
        max_context_length: Maximum context length in tokens
        requires_api_key: Whether provider requires API key
        cost_per_1k_tokens: Approximate cost per 1000 tokens (USD)
        rate_limit: Maximum requests per minute
        local: Whether provider runs locally (no network)
    """
    name: str
    version: str
    model_name: str
    description: str = ""
    supports_streaming: bool = False
    supports_function_calling: bool = False
    max_context_length: int = 4096
    requires_api_key: bool = True
    cost_per_1k_tokens: float = 0.0
    rate_limit: int = 60
    local: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Provider name is required")
        if not self.model_name:
            raise ValueError("Model name is required")
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.updated_at:
            self.updated_at = datetime.utcnow()


@dataclass
class LLMDetectionResult:
    """
    Result from LLM-based entity detection.

    Attributes:
        entity_type: Type of detected entity
        text: Extracted entity text
        start: Start position in text
        end: End position in text
        confidence: Detection confidence (0.0-1.0)
        sensitivity: Sensitivity level of entity
        reasoning: LLM's reasoning for detection
        context: Surrounding context used for detection
        suggestions: LLM's suggestions (e.g., redaction strategy)
    """
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    sensitivity: SensitivityLevel
    reasoning: str = ""
    context: str = ""
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "sensitivity": self.sensitivity.value,
            "reasoning": self.reasoning,
            "context": self.context,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


@dataclass
class LLMValidationResult:
    """
    Result from LLM-based entity validation.

    Attributes:
        is_valid: Whether entity is valid
        confidence: Validation confidence (0.0-1.0)
        reasoning: LLM's reasoning for validation
        corrected_text: Corrected entity text (if applicable)
        suggestions: LLM's suggestions for correction
    """
    is_valid: bool
    confidence: float
    reasoning: str = ""
    corrected_text: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "corrected_text": self.corrected_text,
            "suggestions": self.suggestions
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Providers must implement:
    - get_metadata(): Return provider metadata
    - detect_entities(): Detect entities using LLM
    - validate_entity(): Validate entity using LLM
    - classify_sensitivity(): Classify entity sensitivity

    Optional:
    - initialize(): Setup provider (configure client, load API keys, etc.)
    - cleanup(): Cleanup resources
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize LLM provider.

        Args:
            api_key: API key for remote providers (optional for local)
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
        self._initialized = False
        self._metadata = None

    @abstractmethod
    def get_metadata(self) -> LLMProviderMetadata:
        """
        Return provider metadata.

        Returns:
            LLMProviderMetadata instance
        """
        pass

    @abstractmethod
    def detect_entities(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context_window: int = 100,
        **kwargs
    ) -> List[LLMDetectionResult]:
        """
        Detect entities in text using LLM.

        Args:
            text: Text to analyze
            language: Language code (e.g., 'en', 'fr')
            entity_types: Specific entity types to detect (None = all)
            context_window: Characters of context before/after entities
            **kwargs: Additional provider-specific parameters

        Returns:
            List of LLMDetectionResult instances

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If detection fails
        """
        pass

    @abstractmethod
    def validate_entity(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str] = None,
        **kwargs
    ) -> LLMValidationResult:
        """
        Validate a detected entity using LLM.

        Args:
            entity_text: Text of entity to validate
            entity_type: Type of entity (e.g., 'EMAIL', 'PHONE')
            context: Surrounding context for validation
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMValidationResult instance

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If validation fails
        """
        pass

    @abstractmethod
    def classify_sensitivity(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str] = None,
        **kwargs
    ) -> SensitivityLevel:
        """
        Classify entity sensitivity level using LLM.

        Args:
            entity_text: Text of entity to classify
            entity_type: Type of entity
            context: Surrounding context for classification
            **kwargs: Additional provider-specific parameters

        Returns:
            SensitivityLevel enum value

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If classification fails
        """
        pass

    def initialize(self) -> None:
        """
        Initialize provider (configure client, validate API key, etc.).

        Called once after registration. Override if needed.
        """
        metadata = self.get_metadata()
        if metadata.requires_api_key and not self.api_key:
            raise ValueError(f"API key required for provider: {metadata.name}")
        self._initialized = True

    def cleanup(self) -> None:
        """
        Cleanup provider resources.

        Called when provider is disabled or unregistered. Override if needed.
        """
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    def validate(self) -> Dict[str, Any]:
        """
        Validate provider configuration.

        Returns:
            Dictionary with:
                - valid (bool): Whether provider is valid
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings
        """
        errors = []
        warnings = []

        try:
            metadata = self.get_metadata()
            if metadata.requires_api_key and not self.api_key:
                errors.append("API key required but not provided")
        except Exception as e:
            errors.append(f"Failed to get metadata: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def get_cost_estimate(self, text_length: int) -> float:
        """
        Estimate cost for processing text.

        Args:
            text_length: Length of text in characters

        Returns:
            Estimated cost in USD
        """
        metadata = self.get_metadata()
        # Rough estimate: 1 token ~= 4 characters
        tokens = text_length / 4
        return (tokens / 1000) * metadata.cost_per_1k_tokens

    def __repr__(self) -> str:
        """String representation of provider."""
        metadata = self.get_metadata()
        return f"<{self.__class__.__name__} name='{metadata.name}' model='{metadata.model_name}'>"


class LLMProviderError(Exception):
    """Raised when LLM provider operation fails."""
    pass


class LLMProviderTimeoutError(Exception):
    """Raised when LLM provider operation exceeds timeout."""
    pass


class LLMProviderRateLimitError(Exception):
    """Raised when LLM provider rate limit is exceeded."""
    pass
