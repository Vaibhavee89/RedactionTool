"""
Multilingual and Code-Mixed Support Module.

Provides:
- Language detection (document, paragraph, sentence level)
- Script detection (Devanagari, Latin, etc.)
- Transliteration (Hindi ↔ Roman)
- Code-mixed text handling (Hinglish)
- Multilingual PII detection
"""

from .language_detector import LanguageDetector, Script, detect_language, detect_script, is_code_mixed
from .transliterator import Transliterator, HinglishNormalizer
from .multilingual_detector import MultilingualPIIDetector

__all__ = [
    'LanguageDetector',
    'Script',
    'detect_language',
    'detect_script',
    'is_code_mixed',
    'Transliterator',
    'HinglishNormalizer',
    'MultilingualPIIDetector'
]
