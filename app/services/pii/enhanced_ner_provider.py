"""
Enhanced NER Provider with multilingual support and custom entity types.
"""

import spacy
from typing import List, Dict, Any, Optional
import subprocess
import sys
from app.core.config import Config


class EnhancedNERProvider:
    """
    Enhanced NER provider with:
    - Multilingual support (English, Hindi)
    - Custom entity types
    - Confidence scoring
    - Entity type mapping
    """

    # Entity type mappings for standardization
    ENTITY_MAPPING = {
        # spaCy entities
        'PERSON': 'PERSON',
        'PER': 'PERSON',
        'ORG': 'ORGANIZATION',
        'GPE': 'LOCATION',
        'LOC': 'LOCATION',
        'DATE': 'DATE',
        'TIME': 'TIME',
        'MONEY': 'MONEY',
        'CARDINAL': 'NUMBER',
        'ORDINAL': 'NUMBER',

        # Hindi/Multilingual entities
        'व्यक्ति': 'PERSON',  # Person in Hindi
        'संगठन': 'ORGANIZATION',  # Organization in Hindi
        'स्थान': 'LOCATION',  # Location in Hindi
    }

    def __init__(self, load_hindi: bool = True):
        """
        Initialize NER provider.

        Args:
            load_hindi: Load Hindi/multilingual model
        """
        self.nlp_en = self._load_model(Config.SPACY_MODEL_EN)
        self.nlp_multi = None

        if load_hindi:
            try:
                self.nlp_multi = self._load_model(Config.SPACY_MODEL_MULTI)
            except Exception as e:
                print(f"Warning: Could not load multilingual model: {e}")

    def _load_model(self, model_name: str):
        """Load spaCy model with fallback."""
        try:
            return spacy.load(model_name)
        except OSError:
            try:
                import importlib
                module = importlib.import_module(model_name)
                return module.load()
            except (ImportError, AttributeError):
                print(f"Downloading model {model_name}...")
                subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
                return spacy.load(model_name)

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect entities using NER.

        Args:
            text: Input text
            language: Language code ('en', 'hi', 'multi')
            entity_types: Filter by specific entity types (None = all)

        Returns:
            List of detected entities
        """
        # Select appropriate model
        if language == 'en':
            nlp = self.nlp_en
        elif language in ['hi', 'hin', 'hindi', 'multi']:
            nlp = self.nlp_multi if self.nlp_multi else self.nlp_en
        else:
            nlp = self.nlp_en

        # Process text
        doc = nlp(text)

        entities = []
        for ent in doc.ents:
            # Map entity type
            mapped_type = self._map_entity_type(ent.label_)

            # Filter by entity types if specified
            if entity_types and mapped_type not in entity_types:
                continue

            # Calculate confidence score
            confidence = self._calculate_confidence(ent, doc)

            entities.append({
                "entity_type": mapped_type,
                "original_label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
                "confidence": confidence,
                "source": "ner",
                "language": language
            })

        return entities

    def detect_multilingual(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect entities with both English and multilingual models.

        Args:
            text: Input text

        Returns:
            Combined results from both models
        """
        results_en = self.detect(text, language='en')
        results_multi = []

        if self.nlp_multi:
            results_multi = self.detect(text, language='multi')

        # Combine and deduplicate
        all_results = results_en + results_multi
        return self._deduplicate_entities(all_results)

    def _map_entity_type(self, label: str) -> str:
        """Map entity label to standardized type."""
        return self.ENTITY_MAPPING.get(label, label)

    def _calculate_confidence(self, ent, doc) -> float:
        """
        Calculate confidence score for entity.

        Factors:
        - Entity length
        - Capitalization
        - Position in sentence
        - Entity type
        """
        confidence = 0.8  # Base confidence

        # Bonus for proper capitalization (names should be capitalized)
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            if ent.text[0].isupper():
                confidence += 0.1

        # Bonus for reasonable length
        if 2 <= len(ent.text.split()) <= 5:
            confidence += 0.05

        # Penalty for very short entities
        if len(ent.text) < 3:
            confidence -= 0.1

        # Bonus for dates and times (usually high accuracy)
        if ent.label_ in ['DATE', 'TIME']:
            confidence += 0.15

        return min(confidence, 1.0)

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on position and text."""
        seen = set()
        unique = []

        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)

        for ent in sorted_entities:
            key = (ent['start'], ent['end'], ent['text'])
            if key not in seen:
                seen.add(key)
                unique.append(ent)

        # Sort by position
        return sorted(unique, key=lambda x: x['start'])

    def get_supported_entity_types(self, language: str = 'en') -> List[str]:
        """
        Get list of supported entity types for a language.

        Args:
            language: Language code

        Returns:
            List of entity type labels
        """
        nlp = self.nlp_en if language == 'en' else (self.nlp_multi if self.nlp_multi else self.nlp_en)
        return list(nlp.get_pipe("ner").labels)

    def batch_detect(
        self,
        texts: List[str],
        language: str = 'en'
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect entities in batch for better performance.

        Args:
            texts: List of input texts
            language: Language code

        Returns:
            List of entity lists
        """
        nlp = self.nlp_en if language == 'en' else (self.nlp_multi if self.nlp_multi else self.nlp_en)

        results = []
        for doc in nlp.pipe(texts):
            entities = []
            for ent in doc.ents:
                mapped_type = self._map_entity_type(ent.label_)
                confidence = self._calculate_confidence(ent, doc)

                entities.append({
                    "entity_type": mapped_type,
                    "original_label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "confidence": confidence,
                    "source": "ner",
                    "language": language
                })
            results.append(entities)

        return results
