"""
Enhanced Redactor with advanced redaction strategies and policy support.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional, Callable
from .policy_manager import PolicyManager, RedactionPolicy


class EnhancedRedactor:
    """
    Advanced redactor with multiple redaction strategies and policy support.

    Supported Actions:
    - block: Full redaction with █ blocks
    - mask: Partial masking (e.g., show last 4 digits)
    - partial_mask: Custom partial masking with patterns
    - label: Replace with [ENTITY_TYPE] label
    - hash: Replace with SHA-256 hash (reversible with salt)
    - tokenize: Replace with unique token (e.g., TOKEN_001)
    - allow: No redaction
    """

    def __init__(self, policy_manager: Optional[PolicyManager] = None):
        """
        Initialize enhanced redactor.

        Args:
            policy_manager: PolicyManager instance (creates new if None)
        """
        self.policy_manager = policy_manager or PolicyManager()
        self.token_counter = 0
        self.token_mapping: Dict[str, str] = {}  # Original -> Token

    def redact_text(
        self,
        text: str,
        findings: List[Dict[str, Any]],
        policy: Optional[str] = None,
        custom_rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        Redact text based on findings and policy.

        Args:
            text: Original text
            findings: List of detected entities
            policy: Policy name (None for current policy)
            custom_rules: Optional custom rules to override policy

        Returns:
            Redacted text
        """
        if not findings:
            return text

        # Get active policy
        active_policy = None
        if policy:
            active_policy = self.policy_manager.get_policy(policy)
        elif self.policy_manager.current_policy:
            active_policy = self.policy_manager.current_policy

        # Sort findings by start index in descending order
        sorted_findings = sorted(findings, key=lambda x: x['start'], reverse=True)

        redacted_text = text

        for finding in sorted_findings:
            start = finding['start']
            end = finding['end']
            entity_type = finding['entity_type']
            confidence = finding.get('confidence', 1.0)
            original_value = text[start:end]

            # Get rule from custom rules or policy
            rule = None
            if custom_rules and entity_type in custom_rules:
                rule = custom_rules[entity_type]
            elif active_policy:
                # Check if should redact based on policy
                if not active_policy.should_redact(entity_type, confidence):
                    continue  # Skip redaction
                rule = active_policy.get_rule(entity_type)

            # Default rule if no policy
            if rule is None:
                rule = {'action': 'block'}

            # Generate replacement
            action = rule.get('action', 'block')
            replacement = self._apply_action(
                original_value,
                entity_type,
                action,
                rule
            )

            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]

        return redacted_text

    def _apply_action(
        self,
        value: str,
        entity_type: str,
        action: str,
        rule: Dict[str, Any]
    ) -> str:
        """
        Apply redaction action to value.

        Args:
            value: Original value
            entity_type: Entity type
            action: Redaction action
            rule: Full rule configuration

        Returns:
            Redacted value
        """
        if action == 'allow':
            return value

        elif action == 'block':
            return self._block(value, rule)

        elif action == 'mask':
            return self._mask(value, rule)

        elif action == 'partial_mask':
            return self._partial_mask(value, rule)

        elif action == 'label':
            return self._label(entity_type, rule)

        elif action == 'hash':
            return self._hash(value, rule)

        elif action == 'tokenize':
            return self._tokenize(value, entity_type, rule)

        else:
            # Default to block
            return self._block(value, rule)

    def _block(self, value: str, rule: Dict[str, Any]) -> str:
        """Full redaction with blocks."""
        char = rule.get('char', '█')
        preserve_length = rule.get('preserve_length', True)

        if preserve_length:
            return char * len(value)
        else:
            return char * rule.get('fixed_length', 10)

    def _mask(self, value: str, rule: Dict[str, Any]) -> str:
        """
        Partial masking showing first/last characters.

        Rule options:
        - show_first: Number of characters to show at start
        - show_last: Number of characters to show at end
        - mask_char: Character to use for masking (default: *)
        """
        show_first = rule.get('show_first', 0)
        show_last = rule.get('show_last', 4)
        mask_char = rule.get('mask_char', '*')

        if len(value) <= show_first + show_last:
            # Value too short, mask all
            return mask_char * len(value)

        masked_length = len(value) - show_first - show_last
        masked = value[:show_first] + mask_char * masked_length + value[-show_last:]

        return masked

    def _partial_mask(self, value: str, rule: Dict[str, Any]) -> str:
        """
        Advanced partial masking with patterns.

        Rule options:
        - pattern: Masking pattern (e.g., "XXXX-XXXX-1234" for credit card)
        - preserve_format: Keep formatting characters (spaces, dashes, etc.)
        - mask_positions: List of character positions to mask
        """
        preserve_format = rule.get('preserve_format', True)
        pattern = rule.get('pattern')
        mask_char = rule.get('mask_char', 'X')

        if pattern:
            # Use explicit pattern
            return pattern

        # Auto-detect format and preserve it
        if preserve_format:
            result = []
            for i, char in enumerate(value):
                if char.isalnum():
                    # Mask alphanumeric, but show last 4
                    if i >= len(value) - 4:
                        result.append(char)
                    else:
                        result.append(mask_char if char.isalpha() else 'X')
                else:
                    # Preserve formatting characters
                    result.append(char)
            return ''.join(result)

        # Default to standard masking
        return self._mask(value, rule)

    def _label(self, entity_type: str, rule: Dict[str, Any]) -> str:
        """Replace with entity type label."""
        format_template = rule.get('format', '[{entity_type}]')
        return format_template.format(entity_type=entity_type)

    def _hash(self, value: str, rule: Dict[str, Any]) -> str:
        """
        Replace with cryptographic hash.

        Rule options:
        - algorithm: Hash algorithm (sha256, md5, sha1)
        - salt: Optional salt for hashing
        - prefix: Prefix for hash (e.g., "HASH_")
        - truncate: Number of characters to show (0 for full)
        """
        algorithm = rule.get('algorithm', 'sha256')
        salt = rule.get('salt', '')
        prefix = rule.get('prefix', '')
        truncate = rule.get('truncate', 8)

        # Create hash
        to_hash = (value + salt).encode('utf-8')

        if algorithm == 'md5':
            hash_value = hashlib.md5(to_hash).hexdigest()
        elif algorithm == 'sha1':
            hash_value = hashlib.sha1(to_hash).hexdigest()
        else:  # sha256
            hash_value = hashlib.sha256(to_hash).hexdigest()

        # Truncate if specified
        if truncate > 0:
            hash_value = hash_value[:truncate]

        return prefix + hash_value

    def _tokenize(self, value: str, entity_type: str, rule: Dict[str, Any]) -> str:
        """
        Replace with unique token.

        Rule options:
        - prefix: Token prefix (default: TOKEN_)
        - preserve_mapping: Keep mapping for de-tokenization
        """
        prefix = rule.get('prefix', 'TOKEN_')
        preserve_mapping = rule.get('preserve_mapping', True)

        # Check if we've seen this value before
        if preserve_mapping and value in self.token_mapping:
            return self.token_mapping[value]

        # Generate new token
        self.token_counter += 1
        token = f"{prefix}{entity_type}_{self.token_counter:04d}"

        if preserve_mapping:
            self.token_mapping[value] = token

        return token

    def redact_with_metadata(
        self,
        text: str,
        findings: List[Dict[str, Any]],
        policy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Redact text and return metadata about redactions.

        Args:
            text: Original text
            findings: List of detected entities
            policy: Policy name

        Returns:
            Dictionary with redacted text and metadata
        """
        redacted_text = self.redact_text(text, findings, policy)

        # Calculate statistics
        redaction_count = 0
        redaction_by_type = {}
        redaction_by_action = {}

        active_policy = self.policy_manager.get_policy(policy) if policy else None

        for finding in findings:
            entity_type = finding['entity_type']
            confidence = finding.get('confidence', 1.0)

            # Check if redacted
            if active_policy and not active_policy.should_redact(entity_type, confidence):
                continue

            redaction_count += 1

            # Count by type
            redaction_by_type[entity_type] = redaction_by_type.get(entity_type, 0) + 1

            # Count by action
            if active_policy:
                rule = active_policy.get_rule(entity_type)
                action = rule.get('action', 'block')
                redaction_by_action[action] = redaction_by_action.get(action, 0) + 1

        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'redaction_count': redaction_count,
            'findings_count': len(findings),
            'redacted_percentage': (redaction_count / len(findings) * 100) if findings else 0,
            'by_entity_type': redaction_by_type,
            'by_action': redaction_by_action,
            'policy_used': policy or (active_policy.name if active_policy else None)
        }

    def get_token_mapping(self) -> Dict[str, str]:
        """
        Get token mapping for de-tokenization.

        Returns:
            Dictionary mapping original values to tokens
        """
        return self.token_mapping.copy()

    def clear_token_mapping(self):
        """Clear token mapping and reset counter."""
        self.token_mapping.clear()
        self.token_counter = 0
