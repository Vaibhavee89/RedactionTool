"""
Enhanced ensemble detector with Hindi-specific patterns and improved detection.
"""

from typing import List, Dict, Any, Optional
from .ensemble_detector import EnsembleDetector
from .hindi_regex_provider import HindiRegexProvider


class HindiEnsembleDetector(EnsembleDetector):
    """
    Enhanced ensemble detector for Hindi text.

    Includes:
    - Base ensemble detection (NER + Regex + Presidio)
    - Hindi-specific regex patterns
    - Hindi language model support
    - Context-aware detection for Hindi
    """

    def __init__(
        self,
        use_ner: bool = True,
        use_regex: bool = True,
        use_presidio: bool = True,
        use_hindi_regex: bool = True,
        load_hindi: bool = True
    ):
        """
        Initialize Hindi ensemble detector.

        Args:
            use_ner: Enable NER provider
            use_regex: Enable regex provider
            use_presidio: Enable Presidio provider
            use_hindi_regex: Enable Hindi-specific regex patterns
            load_hindi: Load Hindi/multilingual NER model
        """
        super().__init__(
            use_ner=use_ner,
            use_regex=use_regex,
            use_presidio=use_presidio,
            load_hindi=load_hindi
        )

        # Add Hindi-specific regex provider
        self.hindi_regex_provider = HindiRegexProvider() if use_hindi_regex else None
        self.use_hindi_regex = use_hindi_regex

        # Update provider priority to include Hindi regex
        if use_hindi_regex:
            self.PROVIDER_PRIORITY['hindi_regex'] = 4  # Highest priority for Hindi context

    def detect(
        self,
        text: str,
        language: str = 'hi',  # Default to Hindi
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detect PII with Hindi-specific patterns.

        Args:
            text: Input text (Hindi or mixed)
            language: Language code ('hi' for Hindi, 'en' for English)
            entity_types: Filter by specific entity types
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected entities with conflict resolution
        """
        all_results = []

        # Collect results from base providers
        if self.use_regex and self.regex_provider:
            regex_results = self.regex_provider.detect(text, entity_types)
            all_results.extend(regex_results)

        # Add Hindi-specific regex results (highest priority)
        if self.use_hindi_regex and self.hindi_regex_provider:
            hindi_results = self.hindi_regex_provider.detect(text)

            # Filter by entity types if specified
            if entity_types:
                hindi_results = [r for r in hindi_results if r['entity_type'] in entity_types]

            all_results.extend(hindi_results)

        if self.use_presidio and self.presidio_provider:
            # Presidio supports 'hi' for Hindi
            presidio_results = self.presidio_provider.detect(text, language, entity_types)
            all_results.extend(presidio_results)

        if self.use_ner and self.ner_provider:
            # NER with Hindi model
            ner_results = self.ner_provider.detect(text, language, entity_types)
            all_results.extend(ner_results)

        # Apply conflict resolution
        merged_results = self._resolve_conflicts(all_results)

        # Filter by confidence
        if min_confidence > 0:
            merged_results = [r for r in merged_results if r['confidence'] >= min_confidence]

        return sorted(merged_results, key=lambda x: x['start'])

    def detect_multilingual(
        self,
        text: str,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Detect PII in mixed Hindi-English text.

        Args:
            text: Mixed language text
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with detected entities and language detection
        """
        # Try both Hindi and English detection
        hindi_results = self.detect(text, language='hi', min_confidence=min_confidence)
        english_results = self.detect(text, language='en', min_confidence=min_confidence)

        # Merge results
        all_results = hindi_results + english_results

        # Deduplicate
        merged = self._resolve_conflicts(all_results)

        # Detect language distribution
        lang_dist = self._detect_language_distribution(text)

        return {
            'entities': merged,
            'language_distribution': lang_dist,
            'total_entities': len(merged),
            'statistics': self._calculate_statistics({
                'hindi': hindi_results,
                'english': english_results,
                'merged': merged
            }, merged)
        }

    def _detect_language_distribution(self, text: str) -> Dict[str, float]:
        """
        Detect language distribution in text.

        Args:
            text: Input text

        Returns:
            Dictionary with language percentages
        """
        # Count Devanagari vs Latin characters
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        latin_count = sum(1 for c in text if c.isalpha() and c.isascii())
        total_alpha = devanagari_count + latin_count

        if total_alpha == 0:
            return {'hindi': 0.0, 'english': 0.0, 'mixed': 0.0}

        hindi_pct = (devanagari_count / total_alpha) * 100
        english_pct = (latin_count / total_alpha) * 100

        # Determine category
        if hindi_pct > 70:
            category = 'hindi'
        elif english_pct > 70:
            category = 'english'
        else:
            category = 'mixed'

        return {
            'hindi': hindi_pct,
            'english': english_pct,
            'category': category
        }

    def get_hindi_entity_types(self) -> List[str]:
        """
        Get list of Hindi-specific entity types.

        Returns:
            List of Hindi entity type labels
        """
        if self.hindi_regex_provider:
            return list(self.hindi_regex_provider.patterns.keys())
        return []
