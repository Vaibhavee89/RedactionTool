"""
Cache Manager

LRU cache implementation for LLM provider responses to reduce API costs.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache.

    Stores key-value pairs with automatic eviction of least recently used items
    when capacity is reached.
    """

    def __init__(self, capacity: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to store
            ttl_seconds: Time-to-live for cache entries in seconds (None = no expiry)
        """
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        logger.info(f"LRU cache initialized: capacity={capacity}, ttl={ttl_seconds}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Check TTL
            if self.ttl_seconds:
                timestamp = self._timestamps.get(key)
                if timestamp:
                    age = (datetime.utcnow() - timestamp).total_seconds()
                    if age > self.ttl_seconds:
                        # Expired
                        del self._cache[key]
                        del self._timestamps[key]
                        self._misses += 1
                        return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache.move_to_end(key)
            else:
                # Add new
                if len(self._cache) >= self.capacity:
                    # Evict least recently used
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._timestamps.pop(oldest_key, None)
                    logger.debug(f"Evicted cache entry: {oldest_key}")

            self._cache[key] = value
            self._timestamps[key] = datetime.utcnow()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0

            return {
                "capacity": self.capacity,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds
            }

    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self._cache


class CacheManager:
    """
    Manager for LLM response caching.

    Provides method decorators and utilities for caching LLM API responses.
    """

    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache manager.

        Args:
            capacity: Maximum number of cached responses
            ttl_seconds: Cache entry time-to-live (default: 1 hour)
        """
        self.cache = LRUCache(capacity=capacity, ttl_seconds=ttl_seconds)
        logger.info(f"Cache manager initialized: capacity={capacity}, ttl={ttl_seconds}s")

    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key (hash of arguments)
        """
        # Create stable representation
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def cached(self, func: Callable) -> Callable:
        """
        Decorator for caching function results.

        Usage:
            @cache_manager.cached
            def expensive_llm_call(text, prompt):
                return llm.generate(text, prompt)

        Args:
            func: Function to cache

        Returns:
            Wrapped function with caching
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.generate_key(*args, **kwargs)

            # Check cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Call function
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)

            # Store in cache
            self.cache.put(cache_key, result)

            return result

        return wrapper

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


# Global cache manager instance
_cache_manager = None


def get_cache_manager(
    capacity: int = 1000,
    ttl_seconds: int = 3600
) -> CacheManager:
    """
    Get global cache manager instance.

    Args:
        capacity: Maximum cache capacity (only used on first call)
        ttl_seconds: Cache TTL in seconds (only used on first call)

    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(capacity=capacity, ttl_seconds=ttl_seconds)
    return _cache_manager
