"""
Anthropic LLM Provider

Context-aware PII detection using Anthropic's Claude models.
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging

from app.extensions.interfaces.llm_provider import (
    LLMProvider,
    LLMProviderMetadata,
    LLMDetectionResult,
    LLMValidationResult,
    SensitivityLevel,
    LLMProviderError
)
from app.extensions.utils.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider for context-aware PII detection.

    Uses Claude models for:
    - Context-aware entity detection
    - Entity validation
    - Sensitivity classification
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229", **kwargs):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use (claude-3-opus, claude-3-sonnet, claude-3-haiku)
            **kwargs: Additional configuration
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

        self.model = model
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.0)
        self.client = None

        # Get cache manager
        self.cache_manager = get_cache_manager()

    def get_metadata(self) -> LLMProviderMetadata:
        """Return Anthropic provider metadata."""
        # Cost per 1K tokens (approximate, varies by model)
        costs = {
            "claude-3-opus-20240229": 0.015,
            "claude-3-sonnet-20240229": 0.003,
            "claude-3-haiku-20240307": 0.00025
        }

        return LLMProviderMetadata(
            name="anthropic",
            version="1.0.0",
            model_name=self.model,
            description="Anthropic Claude provider for context-aware PII detection",
            supports_streaming=True,
            supports_function_calling=False,
            max_context_length=200000,  # Claude 3 has 200K context window
            requires_api_key=True,
            cost_per_1k_tokens=costs.get(self.model, 0.003),
            rate_limit=60,
            local=False
        )

    def initialize(self) -> None:
        """Initialize Anthropic client."""
        super().initialize()

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Anthropic provider initialized with model: {self.model}")
        except ImportError:
            raise LLMProviderError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Anthropic client: {str(e)}")

    def detect_entities(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context_window: int = 100,
        **kwargs
    ) -> List[LLMDetectionResult]:
        """
        Detect entities using Claude with context awareness.

        Args:
            text: Text to analyze
            language: Language code
            entity_types: Specific entity types to detect
            context_window: Context window size
            **kwargs: Additional parameters

        Returns:
            List of detected entities
        """
        if not self.client:
            raise LLMProviderError("Provider not initialized")

        # Check cache
        cache_key = self.cache_manager.generate_key(
            "detect", text, language, entity_types
        )
        cached_result = self.cache_manager.cache.get(cache_key)
        if cached_result:
            logger.debug("Cache hit for entity detection")
            return cached_result

        # Build prompt
        prompt = self._build_detection_prompt(text, language, entity_types)

        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._get_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            content = response.content[0].text

            # Extract JSON from response (Claude might add explanation)
            json_str = self._extract_json(content)
            result_data = json.loads(json_str)

            # Convert to detection results
            entities = self._parse_detection_response(result_data, text)

            # Cache result
            self.cache_manager.cache.put(cache_key, entities)

            logger.info(f"Detected {len(entities)} entities with Claude")
            return entities

        except Exception as e:
            logger.error(f"Anthropic detection failed: {str(e)}")
            raise LLMProviderError(f"Detection failed: {str(e)}")

    def validate_entity(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str] = None,
        **kwargs
    ) -> LLMValidationResult:
        """
        Validate entity using Claude.

        Args:
            entity_text: Entity text to validate
            entity_type: Entity type
            context: Surrounding context
            **kwargs: Additional parameters

        Returns:
            Validation result
        """
        if not self.client:
            raise LLMProviderError("Provider not initialized")

        # Check cache
        cache_key = self.cache_manager.generate_key(
            "validate", entity_text, entity_type, context
        )
        cached_result = self.cache_manager.cache.get(cache_key)
        if cached_result:
            logger.debug("Cache hit for entity validation")
            return cached_result

        # Build prompt
        prompt = self._build_validation_prompt(entity_text, entity_type, context)

        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=self.temperature,
                system=self._get_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            content = response.content[0].text
            json_str = self._extract_json(content)
            result_data = json.loads(json_str)

            # Convert to validation result
            result = LLMValidationResult(
                is_valid=result_data.get("is_valid", True),
                confidence=result_data.get("confidence", 0.5),
                reasoning=result_data.get("reasoning", ""),
                corrected_text=result_data.get("corrected_text"),
                suggestions=result_data.get("suggestions", [])
            )

            # Cache result
            self.cache_manager.cache.put(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Anthropic validation failed: {str(e)}")
            raise LLMProviderError(f"Validation failed: {str(e)}")

    def classify_sensitivity(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str] = None,
        **kwargs
    ) -> SensitivityLevel:
        """
        Classify entity sensitivity using Claude.

        Args:
            entity_text: Entity text
            entity_type: Entity type
            context: Surrounding context
            **kwargs: Additional parameters

        Returns:
            Sensitivity level
        """
        if not self.client:
            raise LLMProviderError("Provider not initialized")

        # Check cache
        cache_key = self.cache_manager.generate_key(
            "sensitivity", entity_text, entity_type, context
        )
        cached_result = self.cache_manager.cache.get(cache_key)
        if cached_result:
            logger.debug("Cache hit for sensitivity classification")
            return cached_result

        # Build prompt
        prompt = self._build_sensitivity_prompt(entity_text, entity_type, context)

        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=self.temperature,
                system=self._get_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            content = response.content[0].text
            json_str = self._extract_json(content)
            result_data = json.loads(json_str)

            # Get sensitivity level
            sensitivity_str = result_data.get("sensitivity", "medium").lower()
            sensitivity_map = {
                "low": SensitivityLevel.LOW,
                "medium": SensitivityLevel.MEDIUM,
                "high": SensitivityLevel.HIGH,
                "critical": SensitivityLevel.CRITICAL
            }
            sensitivity = sensitivity_map.get(sensitivity_str, SensitivityLevel.MEDIUM)

            # Cache result
            self.cache_manager.cache.put(cache_key, sensitivity)

            return sensitivity

        except Exception as e:
            logger.error(f"Anthropic sensitivity classification failed: {str(e)}")
            return SensitivityLevel.MEDIUM

    def _get_system_prompt(self) -> str:
        """Get system prompt for PII detection."""
        return """You are an expert at identifying personally identifiable information (PII) in text.
Your task is to detect sensitive information while understanding context and nuance.
Always respond with valid JSON format enclosed in ```json``` code blocks.
Be conservative - when in doubt, flag potential PII."""

    def _build_detection_prompt(
        self,
        text: str,
        language: str,
        entity_types: Optional[List[str]]
    ) -> str:
        """Build prompt for entity detection."""
        entity_filter = ""
        if entity_types:
            entity_filter = f"\nFocus on these entity types: {', '.join(entity_types)}"

        return f"""Analyze the following text and detect all personally identifiable information (PII).{entity_filter}
Language: {language}

Text:
{text}

Return a JSON object with this structure:
```json
{{
  "entities": [
    {{
      "entity_type": "EMAIL|PHONE|SSN|etc",
      "text": "the actual entity text",
      "start": start_position,
      "end": end_position,
      "confidence": 0.0-1.0,
      "sensitivity": "low|medium|high|critical",
      "reasoning": "why this is PII",
      "context": "surrounding text"
    }}
  ]
}}
```"""

    def _build_validation_prompt(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str]
    ) -> str:
        """Build prompt for entity validation."""
        context_str = f"\nContext: {context}" if context else ""

        return f"""Validate if this is a real {entity_type}:{context_str}

Entity: {entity_text}

Return a JSON object in ```json``` code block:
```json
{{
  "is_valid": true|false,
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "corrected_text": "corrected version if invalid",
  "suggestions": ["list of suggestions"]
}}
```"""

    def _build_sensitivity_prompt(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str]
    ) -> str:
        """Build prompt for sensitivity classification."""
        context_str = f"\nContext: {context}" if context else ""

        return f"""Classify the sensitivity level of this {entity_type}:{context_str}

Entity: {entity_text}

Consider:
- Low: Public information, minimal risk if exposed
- Medium: Personal but not highly sensitive
- High: Sensitive personal information
- Critical: Highly sensitive (SSN, financial, health records)

Return a JSON object in ```json``` code block:
```json
{{
  "sensitivity": "low|medium|high|critical",
  "reasoning": "explanation"
}}
```"""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from Claude's response (may include markdown)."""
        import re

        # Try to find JSON in code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no JSON found, return original text and hope for the best
        return text

    def _parse_detection_response(
        self,
        response_data: Dict[str, Any],
        original_text: str
    ) -> List[LLMDetectionResult]:
        """Parse Claude response into detection results."""
        entities = []

        for entity_data in response_data.get("entities", []):
            try:
                # Map sensitivity string to enum
                sensitivity_str = entity_data.get("sensitivity", "medium").lower()
                sensitivity_map = {
                    "low": SensitivityLevel.LOW,
                    "medium": SensitivityLevel.MEDIUM,
                    "high": SensitivityLevel.HIGH,
                    "critical": SensitivityLevel.CRITICAL
                }
                sensitivity = sensitivity_map.get(sensitivity_str, SensitivityLevel.MEDIUM)

                entities.append(LLMDetectionResult(
                    entity_type=entity_data["entity_type"],
                    text=entity_data["text"],
                    start=entity_data["start"],
                    end=entity_data["end"],
                    confidence=entity_data.get("confidence", 0.8),
                    sensitivity=sensitivity,
                    reasoning=entity_data.get("reasoning", ""),
                    context=entity_data.get("context", ""),
                    suggestions=entity_data.get("suggestions", [])
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse entity: {str(e)}")
                continue

        return entities


def create_provider(api_key: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to create Anthropic provider.

    Args:
        api_key: Anthropic API key
        **kwargs: Additional configuration

    Returns:
        AnthropicProvider instance
    """
    return AnthropicProvider(api_key=api_key, **kwargs)
