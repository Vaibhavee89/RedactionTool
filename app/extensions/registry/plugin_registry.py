"""
Plugin Registry

This module manages the lifecycle of detector plugins including discovery,
registration, validation, and execution.
"""

import os
import sys
import importlib
import importlib.util
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import logging

from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    PluginMetadata,
    PluginValidationError,
    PluginExecutionError,
    PluginTimeoutError,
    DetectedEntity
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Registry for managing detector plugins.

    Provides thread-safe plugin registration, discovery, and execution.
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
        """Initialize plugin registry."""
        if self._initialized:
            return

        self._plugins: Dict[str, DetectorPlugin] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._enabled_plugins: Set[str] = set()
        self._plugin_paths: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._initialized = True

        logger.info("Plugin registry initialized")

    def discover_plugins(self, plugin_dir: str = "plugins/detectors") -> List[str]:
        """
        Discover plugins from directory.

        Scans the plugin directory for valid plugin modules and attempts to
        register them automatically.

        Args:
            plugin_dir: Directory containing plugins

        Returns:
            List of discovered plugin names

        Example directory structure:
            plugins/detectors/
                crypto_detector/
                    plugin.py       # Must implement register_plugin()
                    __init__.py
                medical_codes_detector/
                    plugin.py
        """
        discovered = []
        plugin_path = Path(plugin_dir)

        if not plugin_path.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return discovered

        logger.info(f"Discovering plugins in: {plugin_dir}")

        for plugin_folder in plugin_path.iterdir():
            if not plugin_folder.is_dir() or plugin_folder.name.startswith(('_', '.')):
                continue

            plugin_file = plugin_folder / "plugin.py"
            if not plugin_file.exists():
                logger.debug(f"Skipping {plugin_folder.name}: no plugin.py found")
                continue

            try:
                plugin_name = self._load_plugin_from_file(str(plugin_file), plugin_folder.name)
                if plugin_name:
                    discovered.append(plugin_name)
                    self._plugin_paths[plugin_name] = str(plugin_file)
                    logger.info(f"Discovered plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_folder.name}: {str(e)}")

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def _load_plugin_from_file(self, file_path: str, plugin_folder: str) -> Optional[str]:
        """
        Load plugin from file.

        Args:
            file_path: Path to plugin.py file
            plugin_folder: Name of plugin folder

        Returns:
            Plugin name if successful, None otherwise
        """
        try:
            # Create module spec
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_folder}",
                file_path
            )
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec for {file_path}")
                return None

            # Load module
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Get plugin instance
            if not hasattr(module, 'register_plugin'):
                logger.error(f"Plugin {plugin_folder} missing register_plugin() function")
                return None

            plugin_instance = module.register_plugin()
            if not isinstance(plugin_instance, DetectorPlugin):
                logger.error(f"Plugin {plugin_folder} did not return DetectorPlugin instance")
                return None

            # Register plugin
            return self.register(plugin_instance)

        except Exception as e:
            logger.error(f"Error loading plugin from {file_path}: {str(e)}")
            return None

    def register(self, plugin: DetectorPlugin, auto_enable: bool = True) -> str:
        """
        Register a plugin instance.

        Args:
            plugin: DetectorPlugin instance
            auto_enable: Whether to enable plugin automatically

        Returns:
            Plugin name

        Raises:
            PluginValidationError: If plugin validation fails
            ValueError: If plugin already registered
        """
        with self._lock:
            # Get metadata
            try:
                metadata = plugin.get_metadata()
            except Exception as e:
                raise PluginValidationError(f"Failed to get plugin metadata: {str(e)}")

            # Check if already registered
            if metadata.name in self._plugins:
                raise ValueError(f"Plugin already registered: {metadata.name}")

            # Validate plugin
            validation = plugin.validate()
            if not validation.get('valid', False):
                errors = validation.get('errors', [])
                raise PluginValidationError(f"Plugin validation failed: {', '.join(errors)}")

            # Validate dependencies
            self._validate_dependencies(metadata)

            # Initialize plugin
            try:
                plugin.initialize()
            except Exception as e:
                raise PluginValidationError(f"Plugin initialization failed: {str(e)}")

            # Register plugin
            self._plugins[metadata.name] = plugin
            self._metadata_cache[metadata.name] = metadata

            if auto_enable:
                self._enabled_plugins.add(metadata.name)

            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return metadata.name

    def _validate_dependencies(self, metadata: PluginMetadata) -> None:
        """
        Validate plugin dependencies.

        Args:
            metadata: Plugin metadata

        Raises:
            PluginValidationError: If dependencies are missing
        """
        missing = []
        for dependency in metadata.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                missing.append(dependency)

        if missing:
            raise PluginValidationError(
                f"Missing dependencies for plugin {metadata.name}: {', '.join(missing)}"
            )

    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin.

        Args:
            plugin_name: Name of plugin to unregister
        """
        with self._lock:
            if plugin_name not in self._plugins:
                raise ValueError(f"Plugin not registered: {plugin_name}")

            # Cleanup plugin
            try:
                self._plugins[plugin_name].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin_name}: {str(e)}")

            # Remove from registry
            del self._plugins[plugin_name]
            del self._metadata_cache[plugin_name]
            self._enabled_plugins.discard(plugin_name)
            self._plugin_paths.pop(plugin_name, None)

            logger.info(f"Unregistered plugin: {plugin_name}")

    def enable_plugin(self, plugin_name: str) -> None:
        """
        Enable a plugin.

        Args:
            plugin_name: Name of plugin to enable
        """
        with self._lock:
            if plugin_name not in self._plugins:
                raise ValueError(f"Plugin not registered: {plugin_name}")

            self._enabled_plugins.add(plugin_name)
            logger.info(f"Enabled plugin: {plugin_name}")

    def disable_plugin(self, plugin_name: str) -> None:
        """
        Disable a plugin.

        Args:
            plugin_name: Name of plugin to disable
        """
        with self._lock:
            if plugin_name not in self._plugins:
                raise ValueError(f"Plugin not registered: {plugin_name}")

            self._enabled_plugins.discard(plugin_name)
            logger.info(f"Disabled plugin: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> Optional[DetectorPlugin]:
        """
        Get plugin by name.

        Args:
            plugin_name: Name of plugin

        Returns:
            DetectorPlugin instance or None
        """
        return self._plugins.get(plugin_name)

    def get_all_plugins(self) -> Dict[str, DetectorPlugin]:
        """Get all registered plugins."""
        return self._plugins.copy()

    def get_enabled_plugins(self) -> Dict[str, DetectorPlugin]:
        """Get all enabled plugins."""
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if name in self._enabled_plugins
        }

    def get_plugins_for_language(self, language: str) -> List[DetectorPlugin]:
        """
        Get enabled plugins that support a language.

        Args:
            language: Language code (e.g., 'en', 'fr')

        Returns:
            List of DetectorPlugin instances
        """
        plugins = []
        for name in self._enabled_plugins:
            plugin = self._plugins.get(name)
            metadata = self._metadata_cache.get(name)
            if plugin and metadata and metadata.supports_language(language):
                plugins.append(plugin)

        # Sort by priority (higher first)
        plugins.sort(
            key=lambda p: self._metadata_cache[p.get_metadata().name].priority,
            reverse=True
        )
        return plugins

    def get_plugins_for_entity_type(self, entity_type: str) -> List[DetectorPlugin]:
        """
        Get enabled plugins that support an entity type.

        Args:
            entity_type: Entity type (e.g., 'EMAIL', 'PHONE')

        Returns:
            List of DetectorPlugin instances
        """
        plugins = []
        for name in self._enabled_plugins:
            plugin = self._plugins.get(name)
            metadata = self._metadata_cache.get(name)
            if plugin and metadata and metadata.supports_entity_type(entity_type):
                plugins.append(plugin)

        # Sort by priority (higher first)
        plugins.sort(
            key=lambda p: self._metadata_cache[p.get_metadata().name].priority,
            reverse=True
        )
        return plugins

    def get_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata.

        Args:
            plugin_name: Name of plugin

        Returns:
            PluginMetadata instance or None
        """
        return self._metadata_cache.get(plugin_name)

    def is_enabled(self, plugin_name: str) -> bool:
        """
        Check if plugin is enabled.

        Args:
            plugin_name: Name of plugin

        Returns:
            True if enabled, False otherwise
        """
        return plugin_name in self._enabled_plugins

    def get_stats(self) -> Dict[str, any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_plugins": len(self._plugins),
            "enabled_plugins": len(self._enabled_plugins),
            "disabled_plugins": len(self._plugins) - len(self._enabled_plugins),
            "plugins": [
                {
                    "name": metadata.name,
                    "version": metadata.version,
                    "enabled": name in self._enabled_plugins,
                    "entity_types": metadata.supported_entity_types,
                    "languages": metadata.supported_languages,
                    "priority": metadata.priority
                }
                for name, metadata in self._metadata_cache.items()
            ]
        }

    def clear(self) -> None:
        """Clear all registered plugins."""
        with self._lock:
            for plugin in self._plugins.values():
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin: {str(e)}")

            self._plugins.clear()
            self._metadata_cache.clear()
            self._enabled_plugins.clear()
            self._plugin_paths.clear()

            logger.info("Plugin registry cleared")


# Global registry instance
_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get global plugin registry instance."""
    return _registry
