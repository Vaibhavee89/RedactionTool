from presidio_analyzer import AnalyzerEngine
from typing import List, Dict, Any

class PresidioProvider:
    def __init__(self):
        # Initialize the engine (loads models)
        self.analyzer = AnalyzerEngine()

    def detect(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        results = self.analyzer.analyze(text=text, language=language)
        
        entities = []
        for res in results:
            entities.append({
                "entity_type": res.entity_type,
                "start": res.start,
                "end": res.end,
                "text": text[res.start:res.end],
                "confidence": res.score,
                "source": "presidio"
            })
        return entities
