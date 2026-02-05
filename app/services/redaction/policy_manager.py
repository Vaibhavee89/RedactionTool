"""
Policy Manager for loading and managing redaction policies.
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class RedactionPolicy:
    """
    Represents a redaction policy with rules for different entity types.
    """

    def __init__(self, policy_data: Dict[str, Any]):
        """
        Initialize policy from dictionary.

        Args:
            policy_data: Policy configuration dictionary
        """
        self.name = policy_data.get('name', 'Unnamed Policy')
        self.description = policy_data.get('description', '')
        self.version = policy_data.get('version', '1.0')
        self.rules = policy_data.get('rules', {})
        self.global_config = policy_data.get('global', {})

    def get_rule(self, entity_type: str) -> Dict[str, Any]:
        """
        Get redaction rule for entity type.

        Args:
            entity_type: Entity type (e.g., 'PAN', 'PHONE', 'EMAIL')

        Returns:
            Rule configuration dictionary
        """
        # Check for exact match
        if entity_type in self.rules:
            return self.rules[entity_type]

        # Check for wildcard rules
        for pattern, rule in self.rules.items():
            if pattern == '*':  # Default rule
                return rule

        # Return default rule
        return {
            'action': 'block',
            'min_confidence': 0.0
        }

    def should_redact(self, entity_type: str, confidence: float) -> bool:
        """
        Check if entity should be redacted based on policy.

        Args:
            entity_type: Entity type
            confidence: Detection confidence score

        Returns:
            True if should redact, False otherwise
        """
        rule = self.get_rule(entity_type)
        min_confidence = rule.get('min_confidence', 0.0)

        # Check global minimum confidence
        global_min = self.global_config.get('min_confidence', 0.0)
        min_confidence = max(min_confidence, global_min)

        return confidence >= min_confidence and rule.get('action') != 'allow'


class PolicyManager:
    """
    Manages redaction policies - loading, validation, and application.
    """

    def __init__(self, policy_dir: Optional[str] = None):
        """
        Initialize policy manager.

        Args:
            policy_dir: Directory containing policy YAML files
        """
        if policy_dir is None:
            # Default to policies directory in project
            policy_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'policies'
            )

        self.policy_dir = Path(policy_dir)
        self.policies: Dict[str, RedactionPolicy] = {}
        self.current_policy: Optional[RedactionPolicy] = None

        # Create policy directory if it doesn't exist
        self.policy_dir.mkdir(parents=True, exist_ok=True)

        # Load built-in policies
        self._load_builtin_policies()

    def _load_builtin_policies(self):
        """Load all built-in policies from policy directory."""
        if not self.policy_dir.exists():
            return

        for yaml_file in self.policy_dir.glob('*.yaml'):
            try:
                self.load_policy_from_file(str(yaml_file))
            except Exception as e:
                print(f"Warning: Failed to load policy {yaml_file}: {e}")

    def load_policy_from_file(self, file_path: str) -> RedactionPolicy:
        """
        Load policy from YAML file.

        Args:
            file_path: Path to YAML policy file

        Returns:
            RedactionPolicy object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            policy_data = yaml.safe_load(f)

        policy = RedactionPolicy(policy_data)
        policy_name = policy_data.get('name', Path(file_path).stem)

        self.policies[policy_name] = policy
        return policy

    def load_policy_from_dict(self, policy_data: Dict[str, Any]) -> RedactionPolicy:
        """
        Load policy from dictionary.

        Args:
            policy_data: Policy configuration dictionary

        Returns:
            RedactionPolicy object
        """
        policy = RedactionPolicy(policy_data)
        self.policies[policy.name] = policy
        return policy

    def load_policy_from_string(self, yaml_string: str) -> RedactionPolicy:
        """
        Load policy from YAML string.

        Args:
            yaml_string: YAML policy configuration as string

        Returns:
            RedactionPolicy object
        """
        policy_data = yaml.safe_load(yaml_string)
        return self.load_policy_from_dict(policy_data)

    def set_policy(self, policy_name: str):
        """
        Set active policy by name.

        Args:
            policy_name: Name of policy to activate
        """
        if policy_name not in self.policies:
            raise ValueError(f"Policy '{policy_name}' not found")

        self.current_policy = self.policies[policy_name]

    def get_policy(self, policy_name: Optional[str] = None) -> Optional[RedactionPolicy]:
        """
        Get policy by name or current policy.

        Args:
            policy_name: Name of policy (None for current)

        Returns:
            RedactionPolicy object or None
        """
        if policy_name:
            return self.policies.get(policy_name)
        return self.current_policy

    def list_policies(self) -> List[str]:
        """
        Get list of available policy names.

        Returns:
            List of policy names
        """
        return list(self.policies.keys())

    def validate_policy(self, policy: RedactionPolicy) -> Dict[str, Any]:
        """
        Validate policy configuration.

        Args:
            policy: Policy to validate

        Returns:
            Validation result with 'valid' flag and 'errors' list
        """
        errors = []

        # Check required fields
        if not policy.name:
            errors.append("Policy must have a name")

        # Validate rules
        valid_actions = {'block', 'mask', 'partial_mask', 'label', 'allow', 'hash', 'tokenize'}

        for entity_type, rule in policy.rules.items():
            if 'action' not in rule:
                errors.append(f"Rule for '{entity_type}' missing 'action' field")
            elif rule['action'] not in valid_actions:
                errors.append(
                    f"Invalid action '{rule['action']}' for '{entity_type}'. "
                    f"Must be one of: {valid_actions}"
                )

            # Validate confidence threshold
            if 'min_confidence' in rule:
                conf = rule['min_confidence']
                if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                    errors.append(
                        f"Invalid min_confidence for '{entity_type}': {conf}. "
                        "Must be between 0.0 and 1.0"
                    )

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def get_policy_summary(self, policy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of policy details.
        
        Args:
            policy_name: Name of policy (None for current)
            
        Returns:
            Dictionary with policy summary
        """
        policy = self.get_policy(policy_name)
        if not policy:
            return {}
            
        return {
            "name": policy.name,
            "description": policy.description,
            "version": policy.version,
            "rules_count": len(policy.rules),
            "global_config": policy.global_config
        }

    def get_action_for_entity(self, entity_type: str) -> str:
        """
        Get action for a specific entity type from current policy.
        
        Args:
            entity_type: Entity type to check
            
        Returns:
            Action string (e.g., 'block', 'mask')
        """
        if not self.current_policy:
            return 'block'
            
        rule = self.current_policy.get_rule(entity_type)
        return rule.get('action', 'block')

    def save_policy(self, policy: RedactionPolicy, file_path: Optional[str] = None):
        """
        Save policy to YAML file.

        Args:
            policy: Policy to save
            file_path: Path to save to (None for default location)
        """
        if file_path is None:
            file_path = self.policy_dir / f"{policy.name.lower().replace(' ', '_')}.yaml"

        policy_data = {
            'name': policy.name,
            'description': policy.description,
            'version': policy.version,
            'global': policy.global_config,
            'rules': policy.rules
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(policy_data, f, default_flow_style=False, allow_unicode=True)

    def create_custom_policy(
        self,
        name: str,
        rules: Dict[str, Dict[str, Any]],
        description: str = '',
        global_config: Optional[Dict[str, Any]] = None
    ) -> RedactionPolicy:
        """
        Create a custom policy.

        Args:
            name: Policy name
            rules: Dictionary of entity_type -> rule configuration
            description: Policy description
            global_config: Global configuration options

        Returns:
            RedactionPolicy object
        """
        policy_data = {
            'name': name,
            'description': description,
            'version': '1.0',
            'global': global_config or {},
            'rules': rules
        }

        policy = RedactionPolicy(policy_data)

        # Validate before adding
        validation = self.validate_policy(policy)
        if not validation['valid']:
            raise ValueError(f"Invalid policy: {validation['errors']}")

        self.policies[name] = policy
        return policy
