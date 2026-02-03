from typing import List, Dict, Any
from .ner_provider import NERProvider
from .regex_provider import RegexProvider
from .presidio_provider import PresidioProvider

class DetectorEngine:
    def __init__(self):
        self.ner_provider = NERProvider()
        self.regex_provider = RegexProvider()
        self.presidio_provider = PresidioProvider()

    def detect(self, text: str, language: str = 'en') -> List[Dict[str, Any]]:
        # Collect results from all providers
        regex_results = self.regex_provider.detect(text)
        ner_results = self.ner_provider.detect(text, language)
        presidio_results = self.presidio_provider.detect(text, language)
        
        all_results = regex_results + presidio_results + ner_results
        
        # Conflict resolution and merging
        return self._merge_results(all_results)

    def _merge_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Basic merging: Sort by start position
        # In a real enterprise system, we would have complex overlap resolution here
        # For now, we'll just sort and filter complete duplicates
        
        sorted_results = sorted(results, key=lambda x: (x['start'], -x['end']))
        merged = []
        
        if not sorted_results:
            return []

        # Simple greedy strategy: if overlap, keep the one with higher priority or confidence
        # Priority: Regex > Presidio > NER
        
        # This is a simplified de-duplication
        # A proper implementation requires an Interval Tree or similar structure
        
        # Let's simple filter out objects that are fully contained in another
        # Or simple exact duplicates
        
        unique_results = []
        seen = set()
        
        for res in sorted_results:
            key = (res['start'], res['end'], res['entity_type'])
            if key not in seen:
                seen.add(key)
                unique_results.append(res)
                
        return unique_results
