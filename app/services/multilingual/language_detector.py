"""
Enhanced Language Detection for Multilingual Support.

Supports:
- Document-level language detection
- Paragraph-level language detection
- Sentence-level language detection
- Code-mixed text detection (Hinglish, etc.)
- Script detection (Devanagari, Latin, etc.)
- Language confidence scoring
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from langdetect import detect, detect_langs, LangDetectException
import re
from enum import Enum


class Script(Enum):
    """Script types for Indian languages."""
    LATIN = "latin"
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"
    TAMIL = "tamil"
    TELUGU = "telugu"
    GUJARATI = "gujarati"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    GURMUKHI = "gurmukhi"
    ORIYA = "oriya"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LanguageDetector:
    """
    Advanced language detector with support for:
    - Multiple Indian languages
    - Code-mixed text (Hinglish)
    - Script detection
    - Paragraph and sentence-level detection
    """

    # Unicode ranges for Indian scripts
    SCRIPT_RANGES = {
        Script.DEVANAGARI: (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit, Nepali
        Script.BENGALI: (0x0980, 0x09FF),      # Bengali, Assamese
        Script.GURMUKHI: (0x0A00, 0x0A7F),     # Punjabi
        Script.GUJARATI: (0x0A80, 0x0AFF),     # Gujarati
        Script.ORIYA: (0x0B00, 0x0B7F),        # Oriya
        Script.TAMIL: (0x0B80, 0x0BFF),        # Tamil
        Script.TELUGU: (0x0C00, 0x0C7F),       # Telugu
        Script.KANNADA: (0x0C80, 0x0CFF),      # Kannada
        Script.MALAYALAM: (0x0D00, 0x0D7F),    # Malayalam
    }

    # Language code mappings
    LANGUAGE_NAMES = {
        'en': 'English',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'te': 'Telugu',
        'mr': 'Marathi',
        'ta': 'Tamil',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'ur': 'Urdu'
    }

    def __init__(self, min_text_length: int = 20):
        """
        Initialize language detector.

        Args:
            min_text_length: Minimum text length for reliable detection
        """
        self.min_text_length = min_text_length

    def detect_language(
        self,
        text: str,
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Detect primary language of text.

        Args:
            text: Input text
            return_confidence: Whether to return confidence score

        Returns:
            Language code (e.g., 'en', 'hi') or tuple of (language, confidence)
        """
        if not text or len(text.strip()) < 3:
            return ('unknown', 0.0) if return_confidence else 'unknown'

        try:
            if return_confidence:
                langs = detect_langs(text)
                if langs:
                    return (langs[0].lang, langs[0].prob)
                return ('unknown', 0.0)
            else:
                return detect(text)
        except LangDetectException:
            return ('unknown', 0.0) if return_confidence else 'unknown'

    def detect_with_details(self, text: str) -> Dict[str, Any]:
        """
        Detect language with detailed information.

        Args:
            text: Input text

        Returns:
            Dictionary with language details
        """
        if not text or len(text.strip()) < 3:
            return {
                'primary_language': 'unknown',
                'confidence': 0.0,
                'all_languages': [],
                'script': Script.UNKNOWN.value,
                'is_code_mixed': False
            }

        try:
            # Detect all possible languages
            langs = detect_langs(text)

            primary_lang = langs[0].lang if langs else 'unknown'
            primary_conf = langs[0].prob if langs else 0.0

            # Detect script
            script = self.detect_script(text)

            # Check if code-mixed
            is_code_mixed = self.is_code_mixed(text)

            return {
                'primary_language': primary_lang,
                'primary_language_name': self.LANGUAGE_NAMES.get(primary_lang, primary_lang),
                'confidence': primary_conf,
                'all_languages': [
                    {
                        'language': lang.lang,
                        'language_name': self.LANGUAGE_NAMES.get(lang.lang, lang.lang),
                        'probability': lang.prob
                    }
                    for lang in langs
                ],
                'script': script.value,
                'is_code_mixed': is_code_mixed,
                'text_length': len(text)
            }

        except LangDetectException:
            return {
                'primary_language': 'unknown',
                'confidence': 0.0,
                'all_languages': [],
                'script': self.detect_script(text).value,
                'is_code_mixed': False,
                'text_length': len(text)
            }

    def detect_script(self, text: str) -> Script:
        """
        Detect the script used in text.

        Args:
            text: Input text

        Returns:
            Script enum value
        """
        if not text:
            return Script.UNKNOWN

        # Count characters in each script
        script_counts = {}

        for char in text:
            code_point = ord(char)

            # Check Indian scripts
            for script, (start, end) in self.SCRIPT_RANGES.items():
                if start <= code_point <= end:
                    script_counts[script] = script_counts.get(script, 0) + 1
                    break
            else:
                # Check if Latin
                if char.isalpha() and char.isascii():
                    script_counts[Script.LATIN] = script_counts.get(Script.LATIN, 0) + 1

        if not script_counts:
            return Script.UNKNOWN

        # Find dominant script
        max_script = max(script_counts, key=script_counts.get)
        max_count = script_counts[max_script]

        # Check if mixed
        total_script_chars = sum(script_counts.values())
        if len(script_counts) > 1 and max_count / total_script_chars < 0.8:
            return Script.MIXED

        return max_script

    def is_code_mixed(self, text: str) -> bool:
        """
        Detect if text is code-mixed (e.g., Hinglish).

        Args:
            text: Input text

        Returns:
            True if text appears to be code-mixed
        """
        if not text or len(text.strip()) < 10:
            return False

        # Count script types
        script = self.detect_script(text)

        if script == Script.MIXED:
            return True

        # Additional heuristic: Check if multiple languages detected with high probability
        try:
            langs = detect_langs(text)

            if len(langs) >= 2:
                # Check if second language has significant probability
                if langs[1].prob > 0.2:
                    return True

        except LangDetectException:
            pass

        return False

    def detect_paragraph_languages(
        self,
        text: str,
        min_paragraph_length: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Detect language for each paragraph.

        Args:
            text: Input text with multiple paragraphs
            min_paragraph_length: Minimum paragraph length for detection

        Returns:
            List of paragraph language information
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        results = []

        for i, para in enumerate(paragraphs):
            para = para.strip()

            if len(para) < min_paragraph_length:
                results.append({
                    'paragraph_index': i,
                    'text': para,
                    'language': 'unknown',
                    'confidence': 0.0,
                    'reason': 'too_short'
                })
                continue

            lang_info = self.detect_with_details(para)

            results.append({
                'paragraph_index': i,
                'text': para[:100] + '...' if len(para) > 100 else para,
                'language': lang_info['primary_language'],
                'language_name': lang_info['primary_language_name'],
                'confidence': lang_info['confidence'],
                'script': lang_info['script'],
                'is_code_mixed': lang_info['is_code_mixed'],
                'all_languages': lang_info['all_languages']
            })

        return results

    def detect_sentence_languages(
        self,
        text: str,
        min_sentence_length: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Detect language for each sentence.

        Args:
            text: Input text
            min_sentence_length: Minimum sentence length for detection

        Returns:
            List of sentence language information
        """
        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'[.!?।॥]+', text)

        results = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()

            if len(sent) < min_sentence_length:
                continue

            lang_info = self.detect_with_details(sent)

            results.append({
                'sentence_index': i,
                'text': sent[:100] + '...' if len(sent) > 100 else sent,
                'language': lang_info['primary_language'],
                'language_name': lang_info['primary_language_name'],
                'confidence': lang_info['confidence'],
                'script': lang_info['script'],
                'is_code_mixed': lang_info['is_code_mixed']
            })

        return results

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive document language analysis.

        Args:
            text: Full document text

        Returns:
            Complete language analysis
        """
        # Document-level detection
        doc_lang = self.detect_with_details(text)

        # Paragraph-level detection
        para_langs = self.detect_paragraph_languages(text)

        # Calculate language distribution
        lang_counts = {}
        for para in para_langs:
            lang = para.get('language', 'unknown')
            if lang != 'unknown':
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        total_paras = len([p for p in para_langs if p['language'] != 'unknown'])
        lang_distribution = {
            lang: (count / total_paras * 100) if total_paras > 0 else 0
            for lang, count in lang_counts.items()
        }

        # Check if multilingual document
        is_multilingual = len(lang_counts) > 1

        return {
            'document_language': doc_lang['primary_language'],
            'document_language_name': doc_lang['primary_language_name'],
            'document_confidence': doc_lang['confidence'],
            'document_script': doc_lang['script'],
            'is_code_mixed': doc_lang['is_code_mixed'],
            'is_multilingual': is_multilingual,
            'language_distribution': lang_distribution,
            'paragraph_count': len(para_langs),
            'paragraphs': para_langs,
            'statistics': {
                'total_characters': len(text),
                'total_paragraphs': len(para_langs),
                'languages_detected': list(lang_counts.keys()),
                'primary_script': doc_lang['script']
            }
        }

    def get_language_name(self, language_code: str) -> str:
        """
        Get language name from code.

        Args:
            language_code: ISO 639-1 language code

        Returns:
            Language name
        """
        return self.LANGUAGE_NAMES.get(language_code, language_code)


# Convenience functions

def detect_language(text: str) -> str:
    """Quick language detection."""
    detector = LanguageDetector()
    return detector.detect_language(text)


def detect_script(text: str) -> str:
    """Quick script detection."""
    detector = LanguageDetector()
    return detector.detect_script(text).value


def is_code_mixed(text: str) -> bool:
    """Check if text is code-mixed."""
    detector = LanguageDetector()
    return detector.is_code_mixed(text)
