"""
Enhanced Ensemble PII Detector with Plugin and LLM Support

Extends the base EnsembleDetector to support:
- Plugin-based custom detectors
- LLM-powered context-aware detection
- Language pack integration
- Backward compatible with existing code
"""

from typing import List, Dict, Any, Optional
import logging
import signal
from contextlib import contextmanager

from app.services.pii.ensemble_detector import EnsembleDetector
from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    DetectedEntity,
    PluginExecutionError,
    PluginTimeoutError
)
from app.extensions.interfaces.llm_provider import (
    LLMProvider,
    LLMDetectionResult,
    SensitivityLevel
)
from app.extensions.registry.plugin_registry import get_plugin_registry
from app.extensions.registry.llm_registry import get_llm_registry
from app.extensions.registry.language_registry import get_language_registry

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager for timing out operations.

    Args:
        seconds: Timeout in seconds
    """
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class EnhancedEnsembleDetector(EnsembleDetector):
    """
    Enhanced ensemble detector with plugin and LLM support.

    Extends EnsembleDetector to add:
    - Plugin-based detection
    - LLM-powered detection
    - Language pack integration
    - Enhanced conflict resolution

    Maintains 100% backward compatibility with base EnsembleDetector.
    """

    def __init__(
        self,
        use_ner: bool = True,
        use_regex: bool = True,
        use_presidio: bool = True,
        load_hindi: bool = True,
        enable_plugins: bool = False,
        enable_llm: bool = False,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize enhanced ensemble detector.

        Args:
            use_ner: Enable NER provider (inherited)
            use_regex: Enable regex provider (inherited)
            use_presidio: Enable Presidio provider (inherited)
            load_hindi: Load Hindi/multilingual models (inherited)
            enable_plugins: Enable plugin-based detection (NEW)
            enable_llm: Enable LLM-based detection (NEW)
            llm_provider: Specific LLM provider name (None = default)
        """
        # Initialize base detector
        super().__init__(use_ner, use_regex, use_presidio, load_hindi)

        # Extension features
        self.enable_plugins = enable_plugins
        self.enable_llm = enable_llm
        self.llm_provider_name = llm_provider

        # Get registries
        self.plugin_registry = get_plugin_registry() if enable_plugins else None
        self.llm_registry = get_llm_registry() if enable_llm else None
        self.language_registry = get_language_registry()

        # Enhanced provider priority (includes plugins)
        self.ENHANCED_PROVIDER_PRIORITY = {
            'regex': 3,
            'presidio': 2,
            'ner': 1,
            'plugin': 3,  # Same as regex by default
            'llm': 4      # Highest priority (context-aware)
        }

        logger.info(
            f"Enhanced ensemble detector initialized: "
            f"plugins={enable_plugins}, llm={enable_llm}"
        )

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        use_plugins: Optional[bool] = None,
        use_llm: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect PII using enhanced ensemble (base + plugins + LLM).

        Args:
            text: Input text
            language: Language code
            entity_types: Filter by specific entity types
            min_confidence: Minimum confidence threshold
            use_plugins: Override enable_plugins setting
            use_llm: Override enable_llm setting

        Returns:
            List of detected entities with enhanced conflict resolution
        """
        # Use override if provided
        use_plugins_flag = use_plugins if use_plugins is not None else self.enable_plugins
        use_llm_flag = use_llm if use_llm is not None else self.enable_llm

        # Get base detector results
        all_results = []

        # Collect results from base providers
        if self.use_regex and self.regex_provider:
            regex_results = self.regex_provider.detect(text, entity_types)
            all_results.extend(regex_results)

        if self.use_presidio and self.presidio_provider:
            presidio_results = self.presidio_provider.detect(text, language, entity_types)
            all_results.extend(presidio_results)

        if self.use_ner and self.ner_provider:
            ner_results = self.ner_provider.detect(text, language, entity_types)
            all_results.extend(ner_results)

        # Add plugin results
        if use_plugins_flag and self.plugin_registry:
            plugin_results = self._detect_with_plugins(text, language, entity_types)
            all_results.extend(plugin_results)

        # Add LLM results
        if use_llm_flag and self.llm_registry:
            llm_results = self._detect_with_llm(text, language, entity_types)
            all_results.extend(llm_results)

        # Apply enhanced conflict resolution
        merged_results = self._resolve_conflicts_enhanced(all_results)

        # Filter by confidence
        if min_confidence > 0:
            merged_results = [r for r in merged_results if r['confidence'] >= min_confidence]

        # Sort by position
        return sorted(merged_results, key=lambda x: x['start'])

    def _detect_with_plugins(
        self,
        text: str,
        language: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run detection using registered plugins.

        Args:
            text: Input text
            language: Language code
            entity_types: Filter by entity types

        Returns:
            List of detected entities from plugins
        """
        if not self.plugin_registry:
            return []

        results = []

        # Get plugins for language
        plugins = self.plugin_registry.get_plugins_for_language(language)

        logger.debug(f"Running {len(plugins)} plugins for language '{language}'")

        for plugin in plugins:
            try:
                metadata = plugin.get_metadata()

                # Check if plugin is enabled
                if not self.plugin_registry.is_enabled(metadata.name):
                    continue

                # Filter by entity types if specified
                if entity_types:
                    supported = set(metadata.supported_entity_types)
                    requested = set(entity_types)
                    if not supported.intersection(requested):
                        continue

                # Run plugin detection with timeout
                try:
                    with timeout(metadata.timeout_seconds):
                        entities = plugin.detect(
                            text=text,
                            language=language,
                            entity_types=entity_types
                        )
                except TimeoutError:
                    logger.error(f"Plugin {metadata.name} timed out after {metadata.timeout_seconds}s")
                    raise PluginTimeoutError(f"Plugin {metadata.name} timed out")

                # Convert plugin entities to standard format
                for entity in entities:
                    results.append({
                        'entity_type': entity.entity_type,
                        'text': entity.text,
                        'start': entity.start,
                        'end': entity.end,
                        'confidence': entity.confidence,
                        'source': f'plugin:{metadata.name}',
                        'plugin_priority': metadata.priority,
                        'metadata': entity.metadata
                    })

                logger.debug(f"Plugin {metadata.name} detected {len(entities)} entities")

            except Exception as e:
                logger.error(f"Plugin {metadata.name} failed: {str(e)}")
                # Continue with other plugins

        return results

    def _detect_with_llm(
        self,
        text: str,
        language: str,
        entity_types: Optional[List[str]] = None,
        context_window: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Run detection using LLM provider.

        Args:
            text: Input text
            language: Language code
            entity_types: Filter by entity types
            context_window: Context window size

        Returns:
            List of detected entities from LLM
        """
        if not self.llm_registry:
            return []

        results = []

        # Get LLM provider
        provider = self.llm_registry.get_provider(self.llm_provider_name)
        if not provider:
            logger.warning("No LLM provider available")
            return results

        # Check rate limit
        if not self.llm_registry.check_rate_limit(self.llm_provider_name):
            wait_time = self.llm_registry.get_wait_time(self.llm_provider_name)
            logger.warning(f"LLM rate limit exceeded. Wait {wait_time:.1f}s")
            return results

        try:
            # Run LLM detection
            llm_entities = provider.detect_entities(
                text=text,
                language=language,
                entity_types=entity_types,
                context_window=context_window
            )

            # Convert LLM entities to standard format
            for entity in llm_entities:
                results.append({
                    'entity_type': entity.entity_type,
                    'text': entity.text,
                    'start': entity.start,
                    'end': entity.end,
                    'confidence': entity.confidence,
                    'source': 'llm',
                    'sensitivity': entity.sensitivity.value,
                    'reasoning': entity.reasoning,
                    'metadata': entity.metadata
                })

            logger.debug(f"LLM detected {len(llm_entities)} entities")

        except Exception as e:
            logger.error(f"LLM detection failed: {str(e)}")

        return results

    def _resolve_conflicts_enhanced(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhanced conflict resolution with plugin and LLM support.

        Args:
            entities: List of detected entities

        Returns:
            List of resolved entities
        """
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end']))

        # Find overlapping groups
        overlap_groups = self._find_overlap_groups(sorted_entities)

        # Resolve each group with enhanced scoring
        resolved = []
        for group in overlap_groups:
            if len(group) == 1:
                resolved.append(group[0])
            else:
                best = self._choose_best_entity_enhanced(group)
                resolved.append(best)

        return sorted(resolved, key=lambda x: x['start'])

    def _choose_best_entity_enhanced(
        self,
        group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Choose best entity with enhanced scoring for plugins and LLM.

        Args:
            group: List of overlapping entities

        Returns:
            Best entity
        """
        scored = []
        for entity in group:
            score = self._calculate_entity_score_enhanced(entity)
            scored.append((score, entity))

        # Sort by score (descending) and return best
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _calculate_entity_score_enhanced(self, entity: Dict[str, Any]) -> float:
        """
        Calculate enhanced score considering plugins and LLM.

        Args:
            entity: Entity dictionary

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Provider priority
        source = entity.get('source', 'unknown')

        # Handle plugin sources
        if source.startswith('plugin:'):
            # Use plugin-specific priority if available
            plugin_priority = entity.get('plugin_priority', 3)
            score += plugin_priority * 10
        else:
            # Use standard provider priority
            base_source = source.split(':')[0]
            score += self.ENHANCED_PROVIDER_PRIORITY.get(base_source, 0) * 10

        # Confidence (0-1 -> 0-10)
        confidence = entity.get('confidence', 0.5)
        score += confidence * 10

        # Length bonus
        length = entity['end'] - entity['start']
        length_score = min(length / 50.0, 1.0) * 5
        score += length_score

        # Entity type priority
        entity_type = entity.get('entity_type', '')
        score += self._get_entity_type_priority(entity_type)

        # LLM sensitivity bonus
        if 'sensitivity' in entity:
            sensitivity = entity['sensitivity']
            sensitivity_bonus = {
                'critical': 5.0,
                'high': 3.0,
                'medium': 1.0,
                'low': 0.0
            }
            score += sensitivity_bonus.get(sensitivity, 0.0)

        return score

    def detect_with_provenance_enhanced(
        self,
        text: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Detect PII with enhanced provenance including plugins and LLM.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary with results and metadata
        """
        results_by_source = {
            'regex': [],
            'presidio': [],
            'ner': [],
            'plugins': {},
            'llm': []
        }

        # Base providers
        if self.use_regex and self.regex_provider:
            results_by_source['regex'] = self.regex_provider.detect(text)

        if self.use_presidio and self.presidio_provider:
            results_by_source['presidio'] = self.presidio_provider.detect(text, language)

        if self.use_ner and self.ner_provider:
            results_by_source['ner'] = self.ner_provider.detect(text, language)

        # Plugins
        if self.enable_plugins and self.plugin_registry:
            plugins = self.plugin_registry.get_plugins_for_language(language)
            for plugin in plugins:
                metadata = plugin.get_metadata()
                try:
                    entities = plugin.detect(text, language)
                    results_by_source['plugins'][metadata.name] = [
                        e.to_dict() for e in entities
                    ]
                except Exception as e:
                    logger.error(f"Plugin {metadata.name} failed: {str(e)}")

        # LLM
        if self.enable_llm and self.llm_registry:
            llm_results = self._detect_with_llm(text, language)
            results_by_source['llm'] = llm_results

        # Get merged results
        all_results = []
        all_results.extend(results_by_source['regex'])
        all_results.extend(results_by_source['presidio'])
        all_results.extend(results_by_source['ner'])
        for plugin_results in results_by_source['plugins'].values():
            all_results.extend(plugin_results)
        all_results.extend(results_by_source['llm'])

        merged = self._resolve_conflicts_enhanced(all_results)

        return {
            'merged_results': merged,
            'results_by_source': results_by_source,
            'statistics': self._calculate_statistics_enhanced(results_by_source, merged),
            'text': text,
            'language': language
        }

    def _calculate_statistics_enhanced(
        self,
        by_source: Dict[str, Any],
        merged: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate enhanced statistics including plugins and LLM.

        Args:
            by_source: Results grouped by source
            merged: Merged results

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_entities': len(merged),
            'by_source': {
                'regex': len(by_source.get('regex', [])),
                'presidio': len(by_source.get('presidio', [])),
                'ner': len(by_source.get('ner', [])),
                'plugins': sum(
                    len(results) for results in by_source.get('plugins', {}).values()
                ),
                'llm': len(by_source.get('llm', []))
            },
            'by_plugin': {
                plugin_name: len(results)
                for plugin_name, results in by_source.get('plugins', {}).items()
            },
            'by_type': {},
            'confidence_distribution': {
                'high': 0,     # >= 0.8
                'medium': 0,   # >= 0.5
                'low': 0       # < 0.5
            }
        }

        # Count by entity type
        for entity in merged:
            entity_type = entity['entity_type']
            stats['by_type'][entity_type] = stats['by_type'].get(entity_type, 0) + 1

            # Confidence distribution
            conf = entity['confidence']
            if conf >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif conf >= 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1

        return stats

    def get_extension_info(self) -> Dict[str, Any]:
        """
        Get information about enabled extensions.

        Returns:
            Dictionary with extension information
        """
        info = {
            'plugins_enabled': self.enable_plugins,
            'llm_enabled': self.enable_llm,
            'plugins': [],
            'language_packs': [],
            'llm_providers': []
        }

        if self.plugin_registry:
            plugin_stats = self.plugin_registry.get_stats()
            info['plugins'] = plugin_stats.get('plugins', [])

        if self.language_registry:
            lang_stats = self.language_registry.get_stats()
            info['language_packs'] = lang_stats.get('language_packs', [])

        if self.llm_registry:
            llm_stats = self.llm_registry.get_stats()
            info['llm_providers'] = llm_stats.get('providers', [])

        return info
