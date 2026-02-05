"""
Pattern Cache - Intelligent caching for regex patterns and NER models.

Features:
- LRU cache for compiled regex patterns
- Model caching (spaCy, transformers)
- Cache statistics and hit rate tracking
- Configurable cache size
- Memory-efficient storage
"""

import re
import hashlib
from functools import lru_cache
from typing import Dict, Any, Optional, Pattern, List
from collections import OrderedDict
from datetime import datetime
import logging


class PatternCache:
    """
    Intelligent cache for compiled regex patterns and detection models.

    Uses LRU (Least Recently Used) eviction policy for memory efficiency.
    """

    def __init__(self, max_patterns: int = 1000, max_models: int = 10):
        """
        Initialize PatternCache.

        Args:
            max_patterns: Maximum number of cached regex patterns
            max_models: Maximum number of cached ML models
        """
        self.max_patterns = max_patterns
        self.max_models = max_models

        # Pattern cache (regex)
        self._pattern_cache: OrderedDict[str, Pattern] = OrderedDict()

        # Model cache (spaCy, transformers)
        self._model_cache: OrderedDict[str, Any] = OrderedDict()

        # Statistics
        self._stats = {
            "pattern_hits": 0,
            "pattern_misses": 0,
            "model_hits": 0,
            "model_misses": 0,
            "total_patterns_cached": 0,
            "total_models_cached": 0
        }

        self.logger = logging.getLogger(__name__)

    def get_pattern(
        self,
        pattern: str,
        flags: int = 0,
        cache_key: Optional[str] = None
    ) -> Pattern:
        """
        Get compiled regex pattern from cache or compile and cache it.

        Args:
            pattern: Regex pattern string
            flags: Regex flags (re.IGNORECASE, etc.)
            cache_key: Optional custom cache key

        Returns:
            Compiled regex pattern
        """
        # Generate cache key
        if cache_key is None:
            cache_key = self._generate_pattern_key(pattern, flags)

        # Check cache
        if cache_key in self._pattern_cache:
            # Move to end (most recently used)
            self._pattern_cache.move_to_end(cache_key)
            self._stats["pattern_hits"] += 1
            return self._pattern_cache[cache_key]

        # Cache miss - compile pattern
        self._stats["pattern_misses"] += 1

        try:
            compiled_pattern = re.compile(pattern, flags)

            # Add to cache
            self._pattern_cache[cache_key] = compiled_pattern
            self._stats["total_patterns_cached"] += 1

            # Evict oldest if cache is full
            if len(self._pattern_cache) > self.max_patterns:
                oldest_key = next(iter(self._pattern_cache))
                del self._pattern_cache[oldest_key]

            return compiled_pattern

        except re.error as e:
            self.logger.error(f"Invalid regex pattern: {pattern} - {e}")
            raise

    def _generate_pattern_key(self, pattern: str, flags: int) -> str:
        """
        Generate unique cache key for pattern + flags combination.

        Args:
            pattern: Regex pattern
            flags: Regex flags

        Returns:
            Cache key (hash)
        """
        key_str = f"{pattern}:{flags}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_model(
        self,
        model_name: str,
        load_func: Optional[callable] = None
    ) -> Any:
        """
        Get model from cache or load and cache it.

        Args:
            model_name: Name/identifier of the model
            load_func: Function to load the model if not cached

        Returns:
            Loaded model
        """
        # Check cache
        if model_name in self._model_cache:
            self._model_cache.move_to_end(model_name)
            self._stats["model_hits"] += 1
            return self._model_cache[model_name]

        # Cache miss
        self._stats["model_misses"] += 1

        if load_func is None:
            raise ValueError(f"Model '{model_name}' not in cache and no load function provided")

        # Load model
        self.logger.info(f"Loading model: {model_name}")
        model = load_func()

        # Add to cache
        self._model_cache[model_name] = model
        self._stats["total_models_cached"] += 1

        # Evict oldest if cache is full
        if len(self._model_cache) > self.max_models:
            oldest_key = next(iter(self._model_cache))
            self.logger.info(f"Evicting model from cache: {oldest_key}")
            del self._model_cache[oldest_key]

        return model

    def preload_patterns(self, patterns: List[tuple]) -> int:
        """
        Preload multiple patterns into cache.

        Args:
            patterns: List of (pattern_str, flags) tuples

        Returns:
            Number of patterns loaded
        """
        loaded = 0
        for pattern_str, flags in patterns:
            try:
                self.get_pattern(pattern_str, flags)
                loaded += 1
            except Exception as e:
                self.logger.error(f"Failed to preload pattern {pattern_str}: {e}")

        self.logger.info(f"Preloaded {loaded}/{len(patterns)} patterns")
        return loaded

    def clear_patterns(self):
        """Clear all cached patterns."""
        count = len(self._pattern_cache)
        self._pattern_cache.clear()
        self.logger.info(f"Cleared {count} cached patterns")

    def clear_models(self):
        """Clear all cached models."""
        count = len(self._model_cache)
        self._model_cache.clear()
        self.logger.info(f"Cleared {count} cached models")

    def clear_all(self):
        """Clear all caches."""
        self.clear_patterns()
        self.clear_models()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        pattern_total = self._stats["pattern_hits"] + self._stats["pattern_misses"]
        model_total = self._stats["model_hits"] + self._stats["model_misses"]

        return {
            "patterns": {
                "cached": len(self._pattern_cache),
                "max_size": self.max_patterns,
                "hits": self._stats["pattern_hits"],
                "misses": self._stats["pattern_misses"],
                "hit_rate": self._stats["pattern_hits"] / pattern_total if pattern_total > 0 else 0,
                "total_cached_lifetime": self._stats["total_patterns_cached"]
            },
            "models": {
                "cached": len(self._model_cache),
                "max_size": self.max_models,
                "hits": self._stats["model_hits"],
                "misses": self._stats["model_misses"],
                "hit_rate": self._stats["model_hits"] / model_total if model_total > 0 else 0,
                "total_cached_lifetime": self._stats["total_models_cached"]
            }
        }

    def get_cache_summary(self) -> str:
        """
        Get human-readable cache summary.

        Returns:
            Summary string
        """
        stats = self.get_statistics()

        summary = [
            "Pattern Cache Statistics:",
            f"  - Cached patterns: {stats['patterns']['cached']}/{stats['patterns']['max_size']}",
            f"  - Hit rate: {stats['patterns']['hit_rate']:.1%}",
            f"  - Hits: {stats['patterns']['hits']}, Misses: {stats['patterns']['misses']}",
            "",
            "Model Cache Statistics:",
            f"  - Cached models: {stats['models']['cached']}/{stats['models']['max_size']}",
            f"  - Hit rate: {stats['models']['hit_rate']:.1%}",
            f"  - Hits: {stats['models']['hits']}, Misses: {stats['models']['misses']}"
        ]

        return "\n".join(summary)


# Singleton instance for global caching
_global_cache: Optional[PatternCache] = None


def get_cache(max_patterns: int = 1000, max_models: int = 10) -> PatternCache:
    """
    Get global PatternCache singleton.

    Args:
        max_patterns: Maximum patterns to cache
        max_models: Maximum models to cache

    Returns:
        PatternCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = PatternCache(max_patterns=max_patterns, max_models=max_models)
    return _global_cache


# Decorator for caching function results
def cache_result(maxsize: int = 128):
    """
    Decorator to cache function results using LRU cache.

    Args:
        maxsize: Maximum cache size

    Returns:
        Decorated function
    """
    return lru_cache(maxsize=maxsize)
