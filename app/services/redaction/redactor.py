from typing import List, Dict, Any

class Redactor:
    def redact_text(self, text: str, findings: List[Dict[str, Any]], policy: Dict[str, str] = None) -> str:
        """
        Redact text based on findings and policy.
        
        Args:
            text: Original text.
            findings: List of dicts with 'start', 'end', 'entity_type'.
            policy: Dict mapping entity_type to action (e.g., {'PERSON': 'mask', 'PAN': 'block'}).
                    Default action is 'block' (full redaction).
        """
        if not findings:
            return text
            
        # Sort findings by start index in descending order to avoid offset issues
        sorted_findings = sorted(findings, key=lambda x: x['start'], reverse=True)
        
        redacted_text = text
        
        for finding in sorted_findings:
            start = finding['start']
            end = finding['end']
            entity_type = finding['entity_type']
            original_value = text[start:end]
            
            action = 'block'
            if policy and entity_type in policy:
                action = policy[entity_type]
            
            replacement = self._generate_replacement(original_value, entity_type, action)
            
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            
        return redacted_text

    def _generate_replacement(self, value: str, entity_type: str, action: str) -> str:
        if action == 'mask':
            # Show last 4 chars if long enough, else mask all
            if len(value) > 4:
                return '*' * (len(value) - 4) + value[-4:]
            else:
                return '*' * len(value)
        elif action == 'label':
            return f"[{entity_type}]"
        else:
            # Default 'block'
            return "█" * len(value)
