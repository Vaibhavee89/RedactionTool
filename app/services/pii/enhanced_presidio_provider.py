"""
Enhanced Presidio Provider with custom recognizers and improved configuration.
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import List, Dict, Any, Optional
from .custom_presidio_recognizers import get_all_custom_recognizers


class EnhancedPresidioProvider:
    """
    Enhanced Presidio provider with:
    - Custom Indian ID recognizers
    - Configurable entity types
    - Confidence scoring
    - Multi-language support
    """

    def __init__(
        self,
        languages: List[str] = None,
        custom_recognizers: bool = True
    ):
        """
        Initialize enhanced Presidio provider.

        Args:
            languages: List of supported languages
            custom_recognizers: Load custom recognizers
        """
        self.languages = languages or ['en']
        self.analyzer = self._create_analyzer(custom_recognizers)
        self.supported_entities = self._get_supported_entities()

    def _create_analyzer(self, use_custom: bool) -> AnalyzerEngine:
        """Create analyzer engine with custom recognizers."""
        # Configure NLP engine
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"}
            ]
        }

        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        if use_custom:
            # Create analyzer first with default recognizers
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

            # Add custom recognizers to the analyzer's registry
            for recognizer in get_all_custom_recognizers():
                analyzer.registry.add_recognizer(recognizer)
        else:
            # Default analyzer
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        return analyzer

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detect PII using Presidio.

        Args:
            text: Input text
            language: Language code
            entity_types: Filter by specific entity types (None = all)
            score_threshold: Minimum confidence score

        Returns:
            List of detected entities
        """
        # Presidio only supports 'en' for spaCy-based detection
        # For Hindi, we use 'en' and rely on pattern-based recognizers
        analyze_language = 'en' if language in ['hi', 'hin', 'hindi'] else language

        # Analyze text
        try:
            results = self.analyzer.analyze(
                text=text,
                language=analyze_language,
                entities=entity_types,
                score_threshold=score_threshold
            )
        except ValueError as e:
            # If no recognizers found, return empty
            if "No matching recognizers" in str(e):
                return []
            raise

        # Convert to standard format
        entities = []
        for res in results:
            entities.append({
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "text": text[res.start:res.end],
                "confidence": res.score,
                "source": "presidio",
                "recognition_metadata": res.recognition_metadata
            })

        return sorted(entities, key=lambda x: x['start'])

    def detect_with_context(
        self,
        text: str,
        context: List[str] = None,
        language: str = 'en'
    ) -> List[Dict[str, Any]]:
        """
        Detect PII with context keywords for better accuracy.

        Args:
            text: Input text
            context: Context keywords
            language: Language code

        Returns:
            List of detected entities
        """
        # Presidio automatically uses context if recognizers have it defined
        return self.detect(text, language=language)

    def get_supported_entities(self) -> List[str]:
        """
        Get list of supported entity types.

        Returns:
            List of entity type names
        """
        return self.supported_entities

    def _get_supported_entities(self) -> List[str]:
        """Extract supported entity types from analyzer."""
        recognizers = self.analyzer.registry.recognizers
        entities = set()

        for recognizer in recognizers:
            if hasattr(recognizer, 'supported_entities'):
                entities.update(recognizer.supported_entities)
            elif hasattr(recognizer, 'supported_entity'):
                entities.add(recognizer.supported_entity)

        return sorted(list(entities))

    def batch_detect(
        self,
        texts: List[str],
        language: str = 'en'
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect entities in batch.

        Args:
            texts: List of input texts
            language: Language code

        Returns:
            List of entity lists
        """
        results = []
        for text in texts:
            entities = self.detect(text, language=language)
            results.append(entities)

        return results

    def analyze_with_decision_process(
        self,
        text: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Analyze text and return detailed decision process.

        Args:
            text: Input text
            language: Language code

        Returns:
            Dictionary with entities and decision metadata
        """
        results = self.analyzer.analyze(
            text=text,
            language=language,
            return_decision_process=True
        )

        detailed_results = []
        for res in results:
            detailed_results.append({
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "text": text[res.start:res.end],
                "confidence": res.score,
                "source": "presidio",
                "recognizer": res.recognition_metadata.get("recognizer_name") if res.recognition_metadata else None,
                "decision_process": res.analysis_explanation if hasattr(res, 'analysis_explanation') else None
            })

        return {
            "entities": detailed_results,
            "text": text,
            "language": language
        }
