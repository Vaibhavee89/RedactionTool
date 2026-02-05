"""
Language Pack Registry

This module manages language packs for multi-language PII detection support.
"""

import os
import sys
import importlib
import importlib.util
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from app.extensions.interfaces.language_pack import (
    LanguagePack,
    LanguagePackMetadata,
    LanguagePackValidationError,
    RedactionPolicy
)

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """
    Registry for managing language packs.

    Provides thread-safe language pack registration, discovery, and access.
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
        """Initialize language registry."""
        if self._initialized:
            return

        self._language_packs: Dict[str, LanguagePack] = {}
        self._metadata_cache: Dict[str, LanguagePackMetadata] = {}
        self._pack_paths: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._initialized = True

        logger.info("Language pack registry initialized")

    def discover_language_packs(self, language_dir: str = "plugins/languages") -> List[str]:
        """
        Discover language packs from directory.

        Args:
            language_dir: Directory containing language packs

        Returns:
            List of discovered language codes

        Example directory structure:
            plugins/languages/
                fr/
                    language.py     # Must implement register_language_pack()
                    __init__.py
                de/
                    language.py
        """
        discovered = []
        language_path = Path(language_dir)

        if not language_path.exists():
            logger.warning(f"Language directory not found: {language_dir}")
            return discovered

        logger.info(f"Discovering language packs in: {language_dir}")

        for lang_folder in language_path.iterdir():
            if not lang_folder.is_dir() or lang_folder.name.startswith(('_', '.')):
                continue

            language_file = lang_folder / "language.py"
            if not language_file.exists():
                logger.debug(f"Skipping {lang_folder.name}: no language.py found")
                continue

            try:
                language_code = self._load_language_pack_from_file(
                    str(language_file),
                    lang_folder.name
                )
                if language_code:
                    discovered.append(language_code)
                    self._pack_paths[language_code] = str(language_file)
                    logger.info(f"Discovered language pack: {language_code}")
            except Exception as e:
                logger.error(f"Failed to load language pack from {lang_folder.name}: {str(e)}")

        logger.info(f"Discovered {len(discovered)} language packs")
        return discovered

    def _load_language_pack_from_file(
        self,
        file_path: str,
        lang_folder: str
    ) -> Optional[str]:
        """
        Load language pack from file.

        Args:
            file_path: Path to language.py file
            lang_folder: Name of language folder

        Returns:
            Language code if successful, None otherwise
        """
        try:
            # Create module spec
            spec = importlib.util.spec_from_file_location(
                f"language_{lang_folder}",
                file_path
            )
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec for {file_path}")
                return None

            # Load module
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Get language pack instance
            if not hasattr(module, 'register_language_pack'):
                logger.error(f"Language pack {lang_folder} missing register_language_pack() function")
                return None

            pack_instance = module.register_language_pack()
            if not isinstance(pack_instance, LanguagePack):
                logger.error(f"Language pack {lang_folder} did not return LanguagePack instance")
                return None

            # Register language pack
            return self.register(pack_instance)

        except Exception as e:
            logger.error(f"Error loading language pack from {file_path}: {str(e)}")
            return None

    def register(self, language_pack: LanguagePack) -> str:
        """
        Register a language pack instance.

        Args:
            language_pack: LanguagePack instance

        Returns:
            Language code

        Raises:
            LanguagePackValidationError: If validation fails
            ValueError: If language pack already registered
        """
        with self._lock:
            # Get metadata
            try:
                metadata = language_pack.get_metadata()
            except Exception as e:
                raise LanguagePackValidationError(f"Failed to get metadata: {str(e)}")

            # Check if already registered
            if metadata.language_code in self._language_packs:
                raise ValueError(f"Language pack already registered: {metadata.language_code}")

            # Validate language pack
            validation = language_pack.validate()
            if not validation.get('valid', False):
                errors = validation.get('errors', [])
                raise LanguagePackValidationError(
                    f"Language pack validation failed: {', '.join(errors)}"
                )

            # Initialize language pack
            try:
                language_pack.initialize()
            except Exception as e:
                raise LanguagePackValidationError(f"Language pack initialization failed: {str(e)}")

            # Register language pack
            self._language_packs[metadata.language_code] = language_pack
            self._metadata_cache[metadata.language_code] = metadata

            logger.info(f"Registered language pack: {metadata.language_code} ({metadata.language_name})")
            return metadata.language_code

    def unregister(self, language_code: str) -> None:
        """
        Unregister a language pack.

        Args:
            language_code: Language code to unregister
        """
        with self._lock:
            if language_code not in self._language_packs:
                raise ValueError(f"Language pack not registered: {language_code}")

            # Cleanup language pack
            try:
                self._language_packs[language_code].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up language pack {language_code}: {str(e)}")

            # Remove from registry
            del self._language_packs[language_code]
            del self._metadata_cache[language_code]
            self._pack_paths.pop(language_code, None)

            logger.info(f"Unregistered language pack: {language_code}")

    def get_language_pack(self, language_code: str) -> Optional[LanguagePack]:
        """
        Get language pack by code.

        Args:
            language_code: Language code (e.g., 'fr', 'de')

        Returns:
            LanguagePack instance or None
        """
        return self._language_packs.get(language_code)

    def get_all_language_packs(self) -> Dict[str, LanguagePack]:
        """Get all registered language packs."""
        return self._language_packs.copy()

    def get_metadata(self, language_code: str) -> Optional[LanguagePackMetadata]:
        """
        Get language pack metadata.

        Args:
            language_code: Language code

        Returns:
            LanguagePackMetadata instance or None
        """
        return self._metadata_cache.get(language_code)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self._language_packs.keys())

    def get_regex_patterns(self, language_code: str) -> Optional[Dict[str, str]]:
        """
        Get regex patterns for a language.

        Args:
            language_code: Language code

        Returns:
            Dictionary of entity_type -> pattern, or None
        """
        pack = self._language_packs.get(language_code)
        return pack.get_regex_patterns() if pack else None

    def get_redaction_policy(self, language_code: str) -> Optional[RedactionPolicy]:
        """
        Get redaction policy for a language.

        Args:
            language_code: Language code

        Returns:
            RedactionPolicy instance or None
        """
        pack = self._language_packs.get(language_code)
        return pack.get_redaction_policy() if pack else None

    def get_stats(self) -> Dict[str, any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_packs": len(self._language_packs),
            "supported_languages": self.get_supported_languages(),
            "language_packs": [
                {
                    "code": metadata.language_code,
                    "name": metadata.language_name,
                    "script": metadata.script.value,
                    "entity_types": metadata.supported_entity_types,
                    "right_to_left": metadata.right_to_left
                }
                for metadata in self._metadata_cache.values()
            ]
        }

    def clear(self) -> None:
        """Clear all registered language packs."""
        with self._lock:
            for pack in self._language_packs.values():
                try:
                    pack.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up language pack: {str(e)}")

            self._language_packs.clear()
            self._metadata_cache.clear()
            self._pack_paths.clear()

            logger.info("Language pack registry cleared")


# Global registry instance
_registry = LanguageRegistry()


def get_language_registry() -> LanguageRegistry:
    """Get global language registry instance."""
    return _registry
