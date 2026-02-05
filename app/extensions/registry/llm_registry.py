"""
LLM Provider Registry

This module manages LLM providers for context-aware PII detection and validation.
"""

import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from app.extensions.interfaces.llm_provider import (
    LLMProvider,
    LLMProviderMetadata,
    LLMProviderError
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for LLM API calls."""

    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = threading.Lock()

    def check_rate_limit(self) -> bool:
        """
        Check if request is within rate limit.

        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=self.time_window)

            # Remove old requests
            self.requests = [req_time for req_time in self.requests if req_time > cutoff]

            # Check limit
            if len(self.requests) >= self.max_requests:
                return False

            # Record request
            self.requests.append(now)
            return True

    def get_wait_time(self) -> float:
        """
        Get wait time until next request is allowed.

        Returns:
            Wait time in seconds
        """
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0

            oldest = min(self.requests)
            wait_until = oldest + timedelta(seconds=self.time_window)
            now = datetime.utcnow()

            if wait_until <= now:
                return 0.0

            return (wait_until - now).total_seconds()


class LLMProviderRegistry:
    """
    Registry for managing LLM providers.

    Provides thread-safe provider registration, rate limiting, and access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for thread-safe registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize LLM provider registry."""
        if self._initialized:
            return

        self._providers: Dict[str, LLMProvider] = {}
        self._metadata_cache: Dict[str, LLMProviderMetadata] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._enabled_providers: set = set()
        self._default_provider: Optional[str] = None
        self._lock = threading.RLock()
        self._initialized = True

        logger.info("LLM provider registry initialized")

    def register(
        self,
        provider: LLMProvider,
        auto_enable: bool = True,
        set_as_default: bool = False
    ) -> str:
        """
        Register an LLM provider instance.

        Args:
            provider: LLMProvider instance
            auto_enable: Whether to enable provider automatically
            set_as_default: Whether to set as default provider

        Returns:
            Provider name

        Raises:
            LLMProviderError: If validation fails
            ValueError: If provider already registered
        """
        with self._lock:
            # Get metadata
            try:
                metadata = provider.get_metadata()
            except Exception as e:
                raise LLMProviderError(f"Failed to get provider metadata: {str(e)}")

            # Check if already registered
            if metadata.name in self._providers:
                raise ValueError(f"Provider already registered: {metadata.name}")

            # Validate provider
            validation = provider.validate()
            if not validation.get('valid', False):
                errors = validation.get('errors', [])
                raise LLMProviderError(f"Provider validation failed: {', '.join(errors)}")

            # Initialize provider
            try:
                provider.initialize()
            except Exception as e:
                raise LLMProviderError(f"Provider initialization failed: {str(e)}")

            # Create rate limiter
            rate_limiter = RateLimiter(
                max_requests=metadata.rate_limit,
                time_window=60
            )

            # Register provider
            self._providers[metadata.name] = provider
            self._metadata_cache[metadata.name] = metadata
            self._rate_limiters[metadata.name] = rate_limiter

            if auto_enable:
                self._enabled_providers.add(metadata.name)

            if set_as_default or not self._default_provider:
                self._default_provider = metadata.name

            logger.info(f"Registered LLM provider: {metadata.name} ({metadata.model_name})")
            return metadata.name

    def unregister(self, provider_name: str) -> None:
        """
        Unregister an LLM provider.

        Args:
            provider_name: Name of provider to unregister
        """
        with self._lock:
            if provider_name not in self._providers:
                raise ValueError(f"Provider not registered: {provider_name}")

            # Cleanup provider
            try:
                self._providers[provider_name].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up provider {provider_name}: {str(e)}")

            # Remove from registry
            del self._providers[provider_name]
            del self._metadata_cache[provider_name]
            del self._rate_limiters[provider_name]
            self._enabled_providers.discard(provider_name)

            # Update default if needed
            if self._default_provider == provider_name:
                self._default_provider = next(iter(self._enabled_providers), None)

            logger.info(f"Unregistered LLM provider: {provider_name}")

    def enable_provider(self, provider_name: str) -> None:
        """
        Enable a provider.

        Args:
            provider_name: Name of provider to enable
        """
        with self._lock:
            if provider_name not in self._providers:
                raise ValueError(f"Provider not registered: {provider_name}")

            self._enabled_providers.add(provider_name)
            logger.info(f"Enabled LLM provider: {provider_name}")

    def disable_provider(self, provider_name: str) -> None:
        """
        Disable a provider.

        Args:
            provider_name: Name of provider to disable
        """
        with self._lock:
            if provider_name not in self._providers:
                raise ValueError(f"Provider not registered: {provider_name}")

            self._enabled_providers.discard(provider_name)

            # Update default if needed
            if self._default_provider == provider_name:
                self._default_provider = next(iter(self._enabled_providers), None)

            logger.info(f"Disabled LLM provider: {provider_name}")

    def set_default_provider(self, provider_name: str) -> None:
        """
        Set default LLM provider.

        Args:
            provider_name: Name of provider to set as default
        """
        with self._lock:
            if provider_name not in self._providers:
                raise ValueError(f"Provider not registered: {provider_name}")

            if provider_name not in self._enabled_providers:
                raise ValueError(f"Provider not enabled: {provider_name}")

            self._default_provider = provider_name
            logger.info(f"Set default LLM provider: {provider_name}")

    def get_provider(self, provider_name: Optional[str] = None) -> Optional[LLMProvider]:
        """
        Get provider by name or default.

        Args:
            provider_name: Name of provider (None = default)

        Returns:
            LLMProvider instance or None
        """
        if provider_name:
            return self._providers.get(provider_name)
        elif self._default_provider:
            return self._providers.get(self._default_provider)
        return None

    def get_all_providers(self) -> Dict[str, LLMProvider]:
        """Get all registered providers."""
        return self._providers.copy()

    def get_enabled_providers(self) -> Dict[str, LLMProvider]:
        """Get all enabled providers."""
        return {
            name: provider
            for name, provider in self._providers.items()
            if name in self._enabled_providers
        }

    def get_metadata(self, provider_name: str) -> Optional[LLMProviderMetadata]:
        """
        Get provider metadata.

        Args:
            provider_name: Name of provider

        Returns:
            LLMProviderMetadata instance or None
        """
        return self._metadata_cache.get(provider_name)

    def is_enabled(self, provider_name: str) -> bool:
        """
        Check if provider is enabled.

        Args:
            provider_name: Name of provider

        Returns:
            True if enabled, False otherwise
        """
        return provider_name in self._enabled_providers

    def check_rate_limit(self, provider_name: Optional[str] = None) -> bool:
        """
        Check if request is within rate limit.

        Args:
            provider_name: Name of provider (None = default)

        Returns:
            True if request is allowed, False otherwise
        """
        if not provider_name:
            provider_name = self._default_provider

        if not provider_name or provider_name not in self._rate_limiters:
            return False

        return self._rate_limiters[provider_name].check_rate_limit()

    def get_wait_time(self, provider_name: Optional[str] = None) -> float:
        """
        Get wait time until next request is allowed.

        Args:
            provider_name: Name of provider (None = default)

        Returns:
            Wait time in seconds
        """
        if not provider_name:
            provider_name = self._default_provider

        if not provider_name or provider_name not in self._rate_limiters:
            return 0.0

        return self._rate_limiters[provider_name].get_wait_time()

    def get_stats(self) -> Dict[str, any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_providers": len(self._providers),
            "enabled_providers": len(self._enabled_providers),
            "default_provider": self._default_provider,
            "providers": [
                {
                    "name": metadata.name,
                    "model": metadata.model_name,
                    "enabled": name in self._enabled_providers,
                    "is_default": name == self._default_provider,
                    "local": metadata.local,
                    "cost_per_1k_tokens": metadata.cost_per_1k_tokens,
                    "rate_limit": metadata.rate_limit
                }
                for name, metadata in self._metadata_cache.items()
            ]
        }

    def clear(self) -> None:
        """Clear all registered providers."""
        with self._lock:
            for provider in self._providers.values():
                try:
                    provider.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up provider: {str(e)}")

            self._providers.clear()
            self._metadata_cache.clear()
            self._rate_limiters.clear()
            self._enabled_providers.clear()
            self._default_provider = None

            logger.info("LLM provider registry cleared")


# Global registry instance
_registry = LLMProviderRegistry()


def get_llm_registry() -> LLMProviderRegistry:
    """Get global LLM provider registry instance."""
    return _registry
