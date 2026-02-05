"""
Multilingual PII Detector with Code-Mixed Support.

Integrates:
- Language detection
- Script detection
- Transliteration
- PII detection across languages
"""

from typing import Dict, Any, List, Optional
from .language_detector import LanguageDetector, Script
from .transliterator import Transliterator, HinglishNormalizer
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MultilingualPIIDetector:
    """
    PII Detector with multilingual and code-mixed support.

    Features:
    - Automatic language detection
    - Language-aware PII detection
    - Transliteration-aware matching
    - Code-mixed text (Hinglish) support
    - Per-paragraph language adaptation
    """

    def __init__(
        self,
        use_ensemble: bool = True,
        use_hindi: bool = True,
        use_transliteration: bool = True
    ):
        """
        Initialize multilingual PII detector.

        Args:
            use_ensemble: Use ensemble detector
            use_hindi: Load Hindi-specific detectors
            use_transliteration: Enable transliteration-aware matching
        """
        self.language_detector = LanguageDetector()
        self.transliterator = Transliterator() if use_transliteration else None
        self.hinglish_normalizer = HinglishNormalizer() if use_transliteration else None

        # Load PII detectors
        self.use_ensemble = use_ensemble
        self.use_hindi = use_hindi
        self.use_transliteration = use_transliteration

        # Initialize detectors lazily
        self._ensemble_detector = None
        self._hindi_detector = None

    def _get_ensemble_detector(self):
        """Lazy load ensemble detector."""
        if self._ensemble_detector is None:
            try:
                from app.services.pii.ensemble_detector import EnsembleDetector
                self._ensemble_detector = EnsembleDetector()
            except ImportError:
                print("Warning: Ensemble detector not available")
        return self._ensemble_detector

    def _get_hindi_detector(self):
        """Lazy load Hindi ensemble detector."""
        if self._hindi_detector is None:
            try:
                from app.services.pii.hindi_ensemble_detector import HindiEnsembleDetector
                self._hindi_detector = HindiEnsembleDetector(
                    use_hindi_regex=True,
                    load_hindi=self.use_hindi
                )
            except ImportError:
                print("Warning: Hindi detector not available")
        return self._hindi_detector

    def detect(
        self,
        text: str,
        language: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect PII with automatic language detection.

        Args:
            text: Input text
            language: Language code (None for auto-detect)
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with PII findings and language information
        """
        # Detect language if not specified
        if language is None or language == 'auto':
            lang_info = self.language_detector.detect_with_details(text)
            language = lang_info['primary_language']
        else:
            lang_info = self.language_detector.detect_with_details(text)

        # Detect PII based on language
        if language in ['hi', 'mr', 'ne']:  # Devanagari-based languages
            findings = self._detect_hindi(text, min_confidence)
        elif lang_info.get('is_code_mixed'):
            findings = self._detect_code_mixed(text, min_confidence)
        else:
            findings = self._detect_english(text, min_confidence)

        # Add transliteration variants if enabled
        if self.use_transliteration and lang_info.get('script') == Script.DEVANAGARI.value:
            findings = self._add_transliteration_variants(text, findings)

        return {
            'entities': findings,
            'language_info': lang_info,
            'total_entities': len(findings),
            'statistics': self._calculate_statistics(findings)
        }

    def detect_with_language_analysis(
        self,
        text: str,
        analyze_paragraphs: bool = True,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect PII with comprehensive language analysis.

        Args:
            text: Input text
            analyze_paragraphs: Whether to analyze per paragraph
            min_confidence: Minimum confidence threshold

        Returns:
            Comprehensive analysis results
        """
        # Document-level analysis
        doc_analysis = self.language_detector.analyze_document(text)

        # Detect PII
        pii_result = self.detect(text, language=doc_analysis['document_language'], min_confidence=min_confidence)

        result = {
            'document_analysis': doc_analysis,
            'pii_findings': pii_result['entities'],
            'total_pii': len(pii_result['entities']),
            'pii_by_language': self._group_by_language(pii_result['entities']),
            'statistics': pii_result['statistics']
        }

        # Paragraph-level PII detection if requested
        if analyze_paragraphs:
            para_results = []

            for para_info in doc_analysis['paragraphs']:
                para_text = para_info.get('text', '')

                if len(para_text) > 20:
                    para_pii = self.detect(
                        para_text,
                        language=para_info.get('language'),
                        min_confidence=min_confidence
                    )

                    para_results.append({
                        'paragraph_index': para_info['paragraph_index'],
                        'language': para_info['language'],
                        'pii_count': len(para_pii['entities']),
                        'entities': para_pii['entities']
                    })

            result['paragraph_analysis'] = para_results

        return result

    def _detect_english(self, text: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Detect PII in English text."""
        detector = self._get_ensemble_detector()

        if detector:
            return detector.detect(text, language='en', min_confidence=min_confidence)

        return []

    def _detect_hindi(self, text: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Detect PII in Hindi text."""
        detector = self._get_hindi_detector()

        if detector:
            return detector.detect(text, language='hi', min_confidence=min_confidence)

        # Fallback to ensemble
        return self._detect_english(text, min_confidence)

    def _detect_code_mixed(self, text: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Detect PII in code-mixed (Hinglish) text."""
        detector = self._get_hindi_detector()

        if detector:
            # Use multilingual detection
            result = detector.detect_multilingual(text, min_confidence=min_confidence)
            return result.get('entities', [])

        return self._detect_english(text, min_confidence)

    def _add_transliteration_variants(
        self,
        text: str,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add transliteration-aware variants for better matching.

        Args:
            text: Original text
            findings: PII findings

        Returns:
            Enhanced findings with transliteration info
        """
        if not self.transliterator:
            return findings

        enhanced_findings = []

        for finding in findings:
            # Add transliteration info
            trans_info = self.transliterator.detect_transliteration_type(finding['text'])

            finding['transliteration_info'] = trans_info

            # Generate variants for matching
            variants = self.transliterator.normalize_for_matching(finding['text'])
            finding['variants'] = variants

            enhanced_findings.append(finding)

        return enhanced_findings

    def _group_by_language(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group PII findings by detected language."""
        lang_counts = {}

        for finding in findings:
            # Try to detect language of the PII text
            lang = self.language_detector.detect_language(finding.get('text', ''))

            if lang != 'unknown':
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return lang_counts

    def _calculate_statistics(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate PII detection statistics."""
        stats = {
            'total': len(findings),
            'by_type': {},
            'by_source': {},
            'by_confidence': {
                'high': 0,    # >= 0.8
                'medium': 0,  # >= 0.5
                'low': 0      # < 0.5
            }
        }

        for finding in findings:
            # By type
            entity_type = finding.get('entity_type', 'unknown')
            stats['by_type'][entity_type] = stats['by_type'].get(entity_type, 0) + 1

            # By source
            source = finding.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1

            # By confidence
            conf = finding.get('confidence', 0.5)
            if conf >= 0.8:
                stats['by_confidence']['high'] += 1
            elif conf >= 0.5:
                stats['by_confidence']['medium'] += 1
            else:
                stats['by_confidence']['low'] += 1

        return stats

    def detect_romanized_hindi_pii(
        self,
        text: str,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect PII in romanized Hindi (Hinglish) text.

        Args:
            text: Romanized Hindi text
            min_confidence: Minimum confidence threshold

        Returns:
            List of PII findings
        """
        # Check if text is romanized Hindi
        if self.transliterator and self.transliterator.is_romanized_hindi(text):
            # Normalize for better matching
            if self.hinglish_normalizer:
                normalized = self.hinglish_normalizer.normalize(text)
            else:
                normalized = text

            # Detect PII in normalized text
            findings = self._detect_english(normalized, min_confidence)

            # Add romanized Hindi flag
            for finding in findings:
                finding['is_romanized_hindi'] = True
                finding['original_text'] = text[finding['start']:finding['end']]

            return findings

        return self._detect_english(text, min_confidence)


# Convenience function

def detect_multilingual_pii(text: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick multilingual PII detection.

    Args:
        text: Input text
        language: Language code (None for auto-detect)

    Returns:
        PII detection results with language info
    """
    detector = MultilingualPIIDetector()
    return detector.detect(text, language=language)
