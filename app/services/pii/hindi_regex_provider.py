"""
Hindi-specific regex patterns for PII detection.
Handles Hindi text with context-aware patterns.
"""

import re
from typing import List, Dict, Any


class HindiRegexProvider:
    """
    Hindi-specific regex patterns for PII detection.

    Detects:
    - Hindi names with prefixes (श्री, कुमार, etc.)
    - Hindi addresses
    - Hindi dates
    - Context-aware phone, email, PAN, Aadhaar with Hindi labels
    """

    def __init__(self):
        self.patterns = {
            # Hindi name patterns (common prefixes/titles)
            "HINDI_NAME": r'\b(श्री|श्रीमती|कुमार|कुमारी|डॉ|डॉक्टर|प्रो|प्रोफेसर)\s+[\u0900-\u097F]+(\s+[\u0900-\u097F]+)?',

            # Hindi address keywords
            "HINDI_ADDRESS": r'(पता|निवास|घर|मकान|गली|रोड|मार्ग)\s*[:：]\s*[\u0900-\u097F\sa-zA-Z\d,/-]+',

            # Hindi date patterns
            "HINDI_DATE": r'\d{1,2}\s+(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर)\s+\d{4}',

            # Phone with Hindi context
            "HINDI_PHONE": r'(फ़ोन|फोन|मोबाइल|संपर्क|दूरभाष)\s*(नंबर|संख्या)?\s*[:：]?\s*[+\d\s-]{10,}',

            # Email with Hindi context
            "HINDI_EMAIL": r'(ईमेल|इ-मेल|ई-मेल|इमेल)\s*[:：]?\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',

            # PAN with Hindi context
            "HINDI_PAN": r'(पैन|पैन\s*नंबर|पैन\s*संख्या|पैन\s*कार्ड)\s*[:：]?\s*[A-Z]{5}[0-9]{4}[A-Z]{1}',

            # Aadhaar with Hindi context
            "HINDI_AADHAAR": r'(आधार|आधार\s*नंबर|आधार\s*संख्या|आधार\s*कार्ड)\s*[:：]?\s*\d{4}\s*\d{4}\s*\d{4}',

            # Voter ID with Hindi context
            "HINDI_VOTER_ID": r'(मतदाता\s*पहचान|वोटर\s*आईडी|मतदाता\s*कार्ड)\s*[:：]?\s*[A-Z]{3}[0-9]{7}',

            # Driving License with Hindi context
            "HINDI_DL": r'(ड्राइविंग\s*लाइसेंस|चालक\s*लाइसेंस|डीएल)\s*[:：]?\s*[A-Z]{2}[-]?\d{2}[-]?\d{4}[-]?\d{7}',

            # Passport with Hindi context
            "HINDI_PASSPORT": r'(पासपोर्ट|पासपोर्ट\s*नंबर)\s*[:：]?\s*[A-Z]\d{7}',

            # Bank Account with Hindi context
            "HINDI_BANK_ACCOUNT": r'(बैंक\s*खाता|खाता\s*संख्या|खाता\s*नंबर)\s*[:：]?\s*\d{9,18}',

            # IFSC with Hindi context
            "HINDI_IFSC": r'(आईएफएससी|IFSC|आईएफएससी\s*कोड)\s*[:：]?\s*[A-Z]{4}0[A-Z0-9]{6}',

            # Devanagari numbers (for completeness)
            "DEVANAGARI_PHONE": r'(फ़ोन|मोबाइल)\s*[:：]?\s*[०-९\s-]{10,}',
        }

        # Compile patterns with UNICODE flag
        self.compiled_patterns = {
            label: re.compile(pattern, re.UNICODE | re.IGNORECASE)
            for label, pattern in self.patterns.items()
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect Hindi PII entities.

        Args:
            text: Text containing Hindi PII

        Returns:
            List of detected entities
        """
        entities = []

        for label, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                matched_text = match.group()

                # Extract actual value (remove Hindi label)
                value = self.extract_entity_value(matched_text, label)

                entities.append({
                    "entity_type": label,
                    "start": match.start(),
                    "end": match.end(),
                    "text": matched_text,
                    "value": value,  # Cleaned value
                    "confidence": self._calculate_confidence(label, matched_text),
                    "source": "hindi_regex",
                    "language": "hindi"
                })

        return sorted(entities, key=lambda x: x['start'])

    def extract_entity_value(self, text: str, entity_type: str) -> str:
        """
        Extract the actual value from Hindi context.

        Example:
            "पैन: ABCDE1234F" -> "ABCDE1234F"
            "नाम: राजेश कुमार" -> "राजेश कुमार"

        Args:
            text: Matched text with Hindi context
            entity_type: Entity type label

        Returns:
            Cleaned entity value
        """
        # Split by colon or similar
        if ':' in text or '：' in text:
            parts = re.split(r'[:：]', text, maxsplit=1)
            if len(parts) > 1:
                return parts[-1].strip()

        # For names, remove title/prefix
        if entity_type == "HINDI_NAME":
            # Remove common prefixes
            prefixes = ['श्री', 'श्रीमती', 'कुमार', 'कुमारी', 'डॉ', 'डॉक्टर', 'प्रो', 'प्रोफेसर']
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text[len(prefix):].strip()

        return text.strip()

    def _calculate_confidence(self, label: str, text: str) -> float:
        """
        Calculate confidence score for Hindi entity.

        Args:
            label: Entity type label
            text: Matched text

        Returns:
            Confidence score (0-1)
        """
        # High confidence for structured IDs with Hindi context
        high_confidence = ['HINDI_PAN', 'HINDI_AADHAAR', 'HINDI_IFSC', 'HINDI_VOTER_ID']
        if label in high_confidence:
            return 0.90

        # Medium-high for contact info
        medium_high = ['HINDI_PHONE', 'HINDI_EMAIL', 'HINDI_DL', 'HINDI_PASSPORT']
        if label in medium_high:
            return 0.85

        # Medium for names and addresses (can be ambiguous)
        medium = ['HINDI_NAME', 'HINDI_ADDRESS']
        if label in medium:
            return 0.75

        # Lower for dates and bank accounts
        if label in ['HINDI_DATE', 'HINDI_BANK_ACCOUNT']:
            return 0.70

        return 0.80  # Default

    def normalize_devanagari_numbers(self, text: str) -> str:
        """
        Convert Devanagari numerals to Arabic numerals.

        Args:
            text: Text with Devanagari numbers (०-९)

        Returns:
            Text with Arabic numbers (0-9)
        """
        devanagari_to_arabic = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }

        for dev, arab in devanagari_to_arabic.items():
            text = text.replace(dev, arab)

        return text
