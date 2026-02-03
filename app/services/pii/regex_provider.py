import re
from typing import List, Dict, Any

class RegexProvider:
    def __init__(self):
        self.patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(\+?91[\s-]?)?[6-9]\d{9}\b',  # Indian Phone Number
            "PAN": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',    # Indian PAN Card
            "AADHAAR": r'\b\d{4}\s\d{4}\s\d{4}\b'      # Indian Aadhaar (simple check)
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "entity_type": label,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "confidence": 1.0,
                    "source": "regex"
                })
        return entities
