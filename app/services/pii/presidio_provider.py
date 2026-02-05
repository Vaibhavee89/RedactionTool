from presidio_analyzer import AnalyzerEngine
from typing import List, Dict, Any

class PresidioProvider:
    def __init__(self):
        # Initialize the engine (loads models)
        # Explicitly configure to use the small model to avoid downloading 'en_core_web_lg' (800MB)
        # which crashes Streamlit Cloud Free tier.
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

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
