import spacy
from typing import List, Dict, Any, Tuple
import subprocess
import sys
from app.core.config import Config

class NERProvider:
    def __init__(self):
        self.nlp_en = self._load_model(Config.SPACY_MODEL_EN)
        self.nlp_multi = self._load_model(Config.SPACY_MODEL_MULTI)

    def _load_model(self, model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading model {model_name}...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
            return spacy.load(model_name)

    def detect(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        nlp = self.nlp_en if language == 'en' else self.nlp_multi
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "entity_type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
                "confidence": 1.0, # spaCy doesn't provide confidence by default
                "source": "spacy_ner"
            })
        return entities
