"""
Ollama LLM Provider

Context-aware PII detection using local Ollama models.
Privacy-focused, no API key required, runs entirely locally.
"""

import json
import requests
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


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local model-based PII detection.

    Uses locally-running Ollama models for:
    - Privacy-preserving PII detection
    - No API costs
    - No network dependency (after model download)
    - Full control over data

    Recommended models:
    - llama2:13b - Good balance of speed and quality
    - mistral:7b - Fast, good for simple cases
    - mixtral:8x7b - High quality, slower
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama2:13b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama provider.

        Args:
            api_key: Not required for Ollama (kept for interface compatibility)
            model: Ollama model name
            base_url: Ollama API base URL
            **kwargs: Additional configuration
        """
        super().__init__(api_key=None, **kwargs)

        self.model = model
        self.base_url = base_url
        self.timeout = kwargs.get("timeout", 60)

        # Get cache manager
        self.cache_manager = get_cache_manager()

    def get_metadata(self) -> LLMProviderMetadata:
        """Return Ollama provider metadata."""
        return LLMProviderMetadata(
            name="ollama",
            version="1.0.0",
            model_name=self.model,
            description="Local Ollama provider for privacy-preserving PII detection",
            supports_streaming=True,
            supports_function_calling=False,
            max_context_length=4096,  # Varies by model
            requires_api_key=False,
            cost_per_1k_tokens=0.0,  # Free, running locally
            rate_limit=0,  # No limit (local)
            local=True
        )

    def initialize(self) -> None:
        """Initialize Ollama provider (check connectivity)."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]

            if self.model not in model_names:
                logger.warning(
                    f"Model {self.model} not found. Available models: {model_names}"
                )
                logger.warning(f"Pull model with: ollama pull {self.model}")

            logger.info(f"Ollama provider initialized with model: {self.model}")
            self._initialized = True

        except requests.exceptions.RequestException as e:
            raise LLMProviderError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {str(e)}"
            )

    def detect_entities(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context_window: int = 100,
        **kwargs
    ) -> List[LLMDetectionResult]:
        """
        Detect entities using local Ollama model.

        Args:
            text: Text to analyze
            language: Language code
            entity_types: Specific entity types to detect
            context_window: Context window size
            **kwargs: Additional parameters

        Returns:
            List of detected entities
        """
        if not self._initialized:
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
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "system": self._get_system_prompt(),
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 1000
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            content = response.json().get("response", "")

            # Extract JSON
            json_str = self._extract_json(content)
            result_data = json.loads(json_str)

            # Convert to detection results
            entities = self._parse_detection_response(result_data, text)

            # Cache result
            self.cache_manager.cache.put(cache_key, entities)

            logger.info(f"Detected {len(entities)} entities with Ollama")
            return entities

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            raise LLMProviderError(f"Detection failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama response: {str(e)}")
            # Return empty list on parse failure
            return []

    def validate_entity(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str] = None,
        **kwargs
    ) -> LLMValidationResult:
        """
        Validate entity using Ollama.

        Args:
            entity_text: Entity text to validate
            entity_type: Entity type
            context: Surrounding context
            **kwargs: Additional parameters

        Returns:
            Validation result
        """
        if not self._initialized:
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
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "system": self._get_system_prompt(),
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 500
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            content = response.json().get("response", "")
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
            logger.error(f"Ollama validation failed: {str(e)}")
            # Return default validation result
            return LLMValidationResult(
                is_valid=True,
                confidence=0.5,
                reasoning="Validation unavailable"
            )

    def classify_sensitivity(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[str] = None,
        **kwargs
    ) -> SensitivityLevel:
        """
        Classify entity sensitivity using Ollama.

        Args:
            entity_text: Entity text
            entity_type: Entity type
            context: Surrounding context
            **kwargs: Additional parameters

        Returns:
            Sensitivity level
        """
        if not self._initialized:
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
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "system": self._get_system_prompt(),
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 200
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            content = response.json().get("response", "")
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
            logger.error(f"Ollama sensitivity classification failed: {str(e)}")
            return SensitivityLevel.MEDIUM

    def _get_system_prompt(self) -> str:
        """Get system prompt for PII detection."""
        return """You are an expert at identifying personally identifiable information (PII) in text.
Your task is to detect sensitive information while understanding context and nuance.
Always respond with valid JSON format only, no additional text.
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

Return ONLY a JSON object (no other text) with this structure:
{{
  "entities": [
    {{
      "entity_type": "EMAIL|PHONE|SSN|etc",
      "text": "the actual entity text",
      "start": start_position,
      "end": end_position,
      "confidence": 0.8,
      "sensitivity": "low|medium|high|critical",
      "reasoning": "why this is PII",
      "context": "surrounding text"
    }}
  ]
}}"""

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

Return ONLY a JSON object (no other text):
{{
  "is_valid": true,
  "confidence": 0.8,
  "reasoning": "explanation",
  "corrected_text": null,
  "suggestions": []
}}"""

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

Levels:
- low: Public information
- medium: Personal but not highly sensitive
- high: Sensitive personal information
- critical: Highly sensitive (SSN, financial, health)

Return ONLY a JSON object (no other text):
{{
  "sensitivity": "medium",
  "reasoning": "explanation"
}}"""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from Ollama's response."""
        import re

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no JSON found, return original and hope for the best
        return text

    def _parse_detection_response(
        self,
        response_data: Dict[str, Any],
        original_text: str
    ) -> List[LLMDetectionResult]:
        """Parse Ollama response into detection results."""
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
                    confidence=entity_data.get("confidence", 0.7),
                    sensitivity=sensitivity,
                    reasoning=entity_data.get("reasoning", ""),
                    context=entity_data.get("context", ""),
                    suggestions=entity_data.get("suggestions", [])
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse entity: {str(e)}")
                continue

        return entities


def create_provider(model: str = "llama2:13b", **kwargs) -> LLMProvider:
    """
    Factory function to create Ollama provider.

    Args:
        model: Ollama model name
        **kwargs: Additional configuration

    Returns:
        OllamaProvider instance
    """
    return OllamaProvider(model=model, **kwargs)
