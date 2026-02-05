"""
Transliteration Support for Indian Languages.

Handles:
- Hindi (Devanagari) ↔ Latin (Romanized)
- Script conversion
- Romanized Hindi detection (Hinglish)
- Transliteration for PII matching
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class Transliterator:
    """
    Transliterator for Indian scripts (primarily Hindi/Devanagari).

    Supports:
    - Devanagari to Latin (Hindi to English romanization)
    - Latin to Devanagari (English to Hindi)
    - Romanized Hindi (Hinglish) detection
    """

    # Devanagari to Latin mapping (ISO 15919 based)
    DEVANAGARI_TO_LATIN = {
        # Vowels
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
        'ऋ': 'ri', 'ॠ': 'rii', 'ऌ': 'li', 'ॡ': 'lii',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',

        # Consonants
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'ळ': 'l', 'व': 'v', 'w': 'w',
        'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',

        # Additional
        'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',

        # Vowel marks (matras)
        'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
        'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
        '्': '',  # Halant (virama)
        'ं': 'n', 'ः': 'h', 'ँ': 'n',

        # Numbers
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    }

    # Common Romanized Hindi words (for detection)
    ROMANIZED_HINDI_WORDS = {
        'naam', 'naan', 'kya', 'hai', 'hain', 'aap', 'aapka', 'mera', 'tera',
        'yeh', 'woh', 'kahan', 'kaise', 'kyun', 'abhi', 'phir', 'kal',
        'aaj', 'raat', 'din', 'subah', 'shaam', 'paisa', 'rupaya', 'number',
        'phone', 'email', 'address', 'pata', 'ghar', 'shahar', 'gaon',
        'bahut', 'thoda', 'sab', 'kuch', 'koi', 'kaun', 'kahaan'
    }

    # Common Hindi name patterns (romanized)
    HINDI_NAME_PATTERNS = [
        r'\b(kumar|kumari|singh|sharma|verma|gupta|patel|reddy|rao|shah)\b',
        r'\b(raj|rajesh|amit|anil|sunil|deepak|vijay|sanjay|manoj|prakash)\b',
        r'\b(priya|pooja|neha|anjali|kavita|geeta|nisha|rani|sunita)\b'
    ]

    def __init__(self):
        """Initialize transliterator."""
        # Build reverse mapping (dict, not method)
        self.latin_to_dev_map = {v: k for k, v in self.DEVANAGARI_TO_LATIN.items()
                                  if v and k not in ['्']}

    def devanagari_to_latin(self, text: str) -> str:
        """
        Convert Devanagari text to Latin script.

        Args:
            text: Devanagari text

        Returns:
            Romanized (Latin) text
        """
        result = []

        i = 0
        while i < len(text):
            # Check for multi-character combinations
            if i < len(text) - 1:
                two_char = text[i:i+2]
                if two_char in self.DEVANAGARI_TO_LATIN:
                    result.append(self.DEVANAGARI_TO_LATIN[two_char])
                    i += 2
                    continue

            # Single character
            char = text[i]
            if char in self.DEVANAGARI_TO_LATIN:
                result.append(self.DEVANAGARI_TO_LATIN[char])
            else:
                result.append(char)
            i += 1

        return ''.join(result)

    def latin_to_devanagari(self, text: str, strict: bool = False) -> str:
        """
        Convert romanized text to Devanagari (best effort).

        Args:
            text: Romanized text
            strict: If True, only convert known patterns

        Returns:
            Devanagari text (or original if conversion fails)
        """
        # This is a simplified conversion
        # For production, use a proper transliteration library like indic-transliteration

        result = []
        i = 0

        text_lower = text.lower()

        while i < len(text_lower):
            matched = False

            # Try matching longer sequences first
            for length in [3, 2, 1]:
                if i + length <= len(text_lower):
                    substr = text_lower[i:i+length]
                    if substr in self.latin_to_dev_map:
                        result.append(self.latin_to_dev_map[substr])
                        i += length
                        matched = True
                        break

            if not matched:
                if strict:
                    result.append('')  # Skip unknown characters in strict mode
                else:
                    result.append(text[i])  # Keep original character
                i += 1

        return ''.join(result)

    def is_romanized_hindi(self, text: str) -> bool:
        """
        Detect if text is romanized Hindi (Hinglish).

        Args:
            text: Input text

        Returns:
            True if text appears to be romanized Hindi
        """
        if not text:
            return False

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        if not words:
            return False

        # Count Hindi words
        hindi_word_count = sum(1 for word in words if word in self.ROMANIZED_HINDI_WORDS)

        # Check for Hindi name patterns
        has_hindi_names = any(re.search(pattern, text_lower, re.IGNORECASE)
                             for pattern in self.HINDI_NAME_PATTERNS)

        # Heuristic: If >20% words are Hindi or has Hindi names
        if len(words) > 0:
            hindi_ratio = hindi_word_count / len(words)
            return hindi_ratio > 0.2 or has_hindi_names

        return False

    def detect_transliteration_type(self, text: str) -> Dict[str, Any]:
        """
        Detect the type of transliteration in text.

        Args:
            text: Input text

        Returns:
            Dictionary with transliteration information
        """
        # Count character types
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        latin_count = sum(1 for c in text if c.isalpha() and c.isascii())
        total_alpha = devanagari_count + latin_count

        if total_alpha == 0:
            return {
                'type': 'none',
                'script': 'unknown',
                'is_mixed': False
            }

        devanagari_pct = (devanagari_count / total_alpha) * 100
        latin_pct = (latin_count / total_alpha) * 100

        # Determine script type
        if devanagari_pct > 80:
            script_type = 'devanagari'
            is_mixed = False
        elif latin_pct > 80:
            # Check if romanized Hindi
            is_romanized = self.is_romanized_hindi(text)
            script_type = 'romanized_hindi' if is_romanized else 'latin'
            is_mixed = False
        else:
            script_type = 'mixed'
            is_mixed = True

        return {
            'type': script_type,
            'script': 'devanagari' if devanagari_pct > 50 else 'latin',
            'is_mixed': is_mixed,
            'devanagari_percent': devanagari_pct,
            'latin_percent': latin_pct,
            'is_romanized_hindi': self.is_romanized_hindi(text) if latin_pct > 20 else False
        }

    def normalize_for_matching(self, text: str) -> List[str]:
        """
        Generate normalized versions of text for PII matching.

        Useful for matching PII across different scripts.

        Args:
            text: Input text

        Returns:
            List of normalized versions
        """
        variants = [text]

        # Detect script
        trans_info = self.detect_transliteration_type(text)

        # If Devanagari, add romanized version
        if trans_info['devanagari_percent'] > 20:
            romanized = self.devanagari_to_latin(text)
            if romanized != text:
                variants.append(romanized)

        # If romanized Hindi, try to convert to Devanagari
        if trans_info.get('is_romanized_hindi'):
            devanagari = self.latin_to_devanagari(text)
            if devanagari != text:
                variants.append(devanagari)

        return list(set(variants))  # Remove duplicates

    def transliterate_pii_patterns(self, patterns: List[str]) -> List[str]:
        """
        Generate transliterated versions of PII patterns.

        Args:
            patterns: List of regex patterns or text patterns

        Returns:
            Extended list with transliterated versions
        """
        extended_patterns = patterns.copy()

        for pattern in patterns:
            # Generate variants
            variants = self.normalize_for_matching(pattern)
            extended_patterns.extend([v for v in variants if v not in extended_patterns])

        return extended_patterns


class HinglishNormalizer:
    """
    Normalizer for Hinglish (code-mixed Hindi-English) text.

    Handles common variations and spellings in romanized Hindi.
    """

    # Common variations in romanized Hindi
    VARIATIONS = {
        'ph': 'f',   # phone → fone
        'kh': 'k',   # khana → kana
        'gh': 'g',   # ghar → gar
        'th': 't',   # thik → tik
        'dh': 'd',   # dhyan → dyan
        'bh': 'b',   # bhai → bai
        'chh': 'ch', # chhe → che
    }

    def __init__(self):
        """Initialize normalizer."""
        self.transliterator = Transliterator()

    def normalize(self, text: str) -> str:
        """
        Normalize Hinglish text for better matching.

        Args:
            text: Hinglish text

        Returns:
            Normalized text
        """
        # Convert to lowercase
        normalized = text.lower()

        # Apply common variations
        for variant, standard in self.VARIATIONS.items():
            normalized = normalized.replace(variant, standard)

        return normalized

    def generate_variants(self, text: str) -> List[str]:
        """
        Generate common spelling variants of Hinglish text.

        Args:
            text: Input text

        Returns:
            List of variants
        """
        variants = [text, text.lower()]

        # Add normalized version
        normalized = self.normalize(text)
        variants.append(normalized)

        # Add transliterated versions
        trans_variants = self.transliterator.normalize_for_matching(text)
        variants.extend(trans_variants)

        return list(set(variants))
