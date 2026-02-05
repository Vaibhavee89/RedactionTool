"""
Ensemble PII Detector with multi-layered detection and conflict resolution.
Combines NER, Regex, and Presidio for maximum accuracy and recall.
"""

from typing import List, Dict, Any, Optional, Tuple
from .enhanced_ner_provider import EnhancedNERProvider
from .enhanced_regex_provider import EnhancedRegexProvider
from .enhanced_presidio_provider import EnhancedPresidioProvider


class EnsembleDetector:
    """
    Ensemble detector combining multiple PII detection methods:
    - NER-based detection (spaCy)
    - Rule-based detection (Regex)
    - Presidio integration (ML-based)

    Features:
    - Multi-layered detection
    - Conflict resolution
    - Priority ordering
    - Confidence-based merging
    - High recall + precision
    """

    # Priority order for conflict resolution (higher = more priority)
    PROVIDER_PRIORITY = {
        'regex': 3,      # Highest priority (most specific)
        'presidio': 2,   # Medium priority (ML-based)
        'ner': 1         # Lowest priority (most general)
    }

    # Entity type aliases for merging
    ENTITY_ALIASES = {
        'PHONE_IN': 'PHONE_NUMBER',
        'PHONE_US': 'PHONE_NUMBER',
        'PHONE_INTL': 'PHONE_NUMBER',
        'PHONE': 'PHONE_NUMBER',

        'DATE_DMY': 'DATE',
        'DATE_MDY': 'DATE',
        'DATE_YMD': 'DATE',
        'DATE_TEXT': 'DATE',

        'ORG': 'ORGANIZATION',
        'GPE': 'LOCATION',
        'LOC': 'LOCATION',

        'CREDIT_CARD': 'CREDIT_CARD_NUMBER',
        'SSN': 'US_SSN',
    }

    def __init__(
        self,
        use_ner: bool = True,
        use_regex: bool = True,
        use_presidio: bool = True,
        load_hindi: bool = True
    ):
        """
        Initialize ensemble detector.

        Args:
            use_ner: Enable NER provider
            use_regex: Enable regex provider
            use_presidio: Enable Presidio provider
            load_hindi: Load Hindi/multilingual models
        """
        self.use_ner = use_ner
        self.use_regex = use_regex
        self.use_presidio = use_presidio

        # Initialize providers
        self.ner_provider = EnhancedNERProvider(load_hindi=load_hindi) if use_ner else None
        self.regex_provider = EnhancedRegexProvider() if use_regex else None
        self.presidio_provider = EnhancedPresidioProvider() if use_presidio else None

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detect PII using ensemble of all providers.

        Args:
            text: Input text
            language: Language code
            entity_types: Filter by specific entity types
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected entities with conflict resolution
        """
        all_results = []

        # Collect results from each provider
        if self.use_regex and self.regex_provider:
            regex_results = self.regex_provider.detect(text, entity_types)
            all_results.extend(regex_results)

        if self.use_presidio and self.presidio_provider:
            presidio_results = self.presidio_provider.detect(text, language, entity_types)
            all_results.extend(presidio_results)

        if self.use_ner and self.ner_provider:
            ner_results = self.ner_provider.detect(text, language, entity_types)
            all_results.extend(ner_results)

        # Apply conflict resolution
        merged_results = self._resolve_conflicts(all_results)

        # Filter by confidence
        if min_confidence > 0:
            merged_results = [r for r in merged_results if r['confidence'] >= min_confidence]

        # Sort by position
        return sorted(merged_results, key=lambda x: x['start'])

    def detect_with_provenance(
        self,
        text: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Detect PII and return results with provenance information.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary with results and metadata
        """
        # Get results from each provider separately
        results_by_provider = {
            'regex': [],
            'presidio': [],
            'ner': []
        }

        if self.use_regex and self.regex_provider:
            results_by_provider['regex'] = self.regex_provider.detect(text)

        if self.use_presidio and self.presidio_provider:
            results_by_provider['presidio'] = self.presidio_provider.detect(text, language)

        if self.use_ner and self.ner_provider:
            results_by_provider['ner'] = self.ner_provider.detect(text, language)

        # Get merged results
        all_results = []
        for provider_results in results_by_provider.values():
            all_results.extend(provider_results)

        merged = self._resolve_conflicts(all_results)

        return {
            'merged_results': merged,
            'results_by_provider': results_by_provider,
            'statistics': self._calculate_statistics(results_by_provider, merged),
            'text': text,
            'language': language
        }

    def _resolve_conflicts(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve conflicts between overlapping entities.

        Strategy:
        1. Group overlapping entities
        2. Apply priority ordering
        3. Choose best entity per overlap group
        4. Merge adjacent/nested entities when appropriate

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

        # Resolve each group
        resolved = []
        for group in overlap_groups:
            if len(group) == 1:
                resolved.append(group[0])
            else:
                best = self._choose_best_entity(group)
                resolved.append(best)

        return sorted(resolved, key=lambda x: x['start'])

    def _find_overlap_groups(self, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group overlapping entities together.

        Args:
            entities: Sorted list of entities

        Returns:
            List of overlap groups
        """
        if not entities:
            return []

        groups = []
        current_group = [entities[0]]

        for entity in entities[1:]:
            # Check if overlaps with any entity in current group
            overlaps = any(
                self._entities_overlap(entity, group_entity)
                for group_entity in current_group
            )

            if overlaps:
                current_group.append(entity)
            else:
                groups.append(current_group)
                current_group = [entity]

        groups.append(current_group)
        return groups

    def _entities_overlap(self, e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
        """
        Check if two entities overlap.

        Args:
            e1: First entity
            e2: Second entity

        Returns:
            True if entities overlap
        """
        return not (e1['end'] <= e2['start'] or e2['end'] <= e1['start'])

    def _choose_best_entity(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Choose the best entity from a group of overlapping entities.

        Priority factors:
        1. Provider priority (regex > presidio > ner)
        2. Confidence score
        3. Specificity (longer match preferred)
        4. Entity type priority

        Args:
            group: List of overlapping entities

        Returns:
            Best entity
        """
        # Score each entity
        scored = []
        for entity in group:
            score = self._calculate_entity_score(entity)
            scored.append((score, entity))

        # Sort by score (descending) and return best
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _calculate_entity_score(self, entity: Dict[str, Any]) -> float:
        """
        Calculate overall score for entity selection.

        Args:
            entity: Entity dictionary

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Provider priority (0-3)
        source = entity.get('source', 'unknown')
        score += self.PROVIDER_PRIORITY.get(source, 0) * 10

        # Confidence (0-1 -> 0-10)
        confidence = entity.get('confidence', 0.5)
        score += confidence * 10

        # Length bonus (longer matches often more specific)
        length = entity['end'] - entity['start']
        length_score = min(length / 50.0, 1.0) * 5  # Max 5 points
        score += length_score

        # Entity type priority
        entity_type = entity.get('entity_type', '')
        score += self._get_entity_type_priority(entity_type)

        return score

    def _get_entity_type_priority(self, entity_type: str) -> float:
        """
        Get priority score for entity type.

        Args:
            entity_type: Entity type string

        Returns:
            Priority score
        """
        # Specific IDs have higher priority
        high_priority = ['PAN', 'AADHAAR', 'SSN', 'CREDIT_CARD', 'PASSPORT',
                        'DRIVING_LICENSE', 'VOTER_ID']
        if entity_type in high_priority:
            return 3.0

        # Contact info medium priority
        medium_priority = ['EMAIL', 'PHONE_NUMBER', 'PHONE_IN', 'IFSC_CODE']
        if entity_type in medium_priority:
            return 2.0

        # General entities lower priority
        low_priority = ['PERSON', 'ORGANIZATION', 'LOCATION']
        if entity_type in low_priority:
            return 1.0

        return 0.5

    def _normalize_entity_type(self, entity_type: str) -> str:
        """
        Normalize entity type using aliases.

        Args:
            entity_type: Original entity type

        Returns:
            Normalized entity type
        """
        return self.ENTITY_ALIASES.get(entity_type, entity_type)

    def _calculate_statistics(
        self,
        by_provider: Dict[str, List[Dict[str, Any]]],
        merged: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate detection statistics.

        Args:
            by_provider: Results grouped by provider
            merged: Merged results

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_entities': len(merged),
            'by_provider': {
                'regex': len(by_provider.get('regex', [])),
                'presidio': len(by_provider.get('presidio', [])),
                'ner': len(by_provider.get('ner', []))
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

    def get_supported_entity_types(self) -> Dict[str, List[str]]:
        """
        Get supported entity types for each provider.

        Returns:
            Dictionary mapping provider to entity types
        """
        supported = {}

        if self.regex_provider:
            supported['regex'] = list(self.regex_provider.patterns.keys())

        if self.presidio_provider:
            supported['presidio'] = self.presidio_provider.get_supported_entities()

        if self.ner_provider:
            supported['ner'] = self.ner_provider.get_supported_entity_types('en')

        return supported

    def benchmark_providers(
        self,
        text: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Benchmark performance of each provider.

        Args:
            text: Input text
            language: Language code

        Returns:
            Benchmark results
        """
        import time

        results = {}

        # Benchmark regex
        if self.regex_provider:
            start = time.time()
            regex_results = self.regex_provider.detect(text)
            regex_time = time.time() - start
            results['regex'] = {
                'time_ms': regex_time * 1000,
                'count': len(regex_results)
            }

        # Benchmark Presidio
        if self.presidio_provider:
            start = time.time()
            presidio_results = self.presidio_provider.detect(text, language)
            presidio_time = time.time() - start
            results['presidio'] = {
                'time_ms': presidio_time * 1000,
                'count': len(presidio_results)
            }

        # Benchmark NER
        if self.ner_provider:
            start = time.time()
            ner_results = self.ner_provider.detect(text, language)
            ner_time = time.time() - start
            results['ner'] = {
                'time_ms': ner_time * 1000,
                'count': len(ner_results)
            }

        return results
