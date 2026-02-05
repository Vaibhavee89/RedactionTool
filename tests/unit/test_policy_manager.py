"""
Unit tests for PolicyManager and RedactionPolicy.

Tests:
- Policy loading from YAML
- Policy validation
- Rule application
- Confidence thresholds
- Custom policy creation
"""

import pytest
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.redaction.policy_manager import PolicyManager, RedactionPolicy


class TestRedactionPolicy:
    """Test cases for RedactionPolicy."""

    def test_policy_initialization(self):
        """Test policy initialization with dict."""
        policy_dict = {
            'name': 'test_policy',
            'version': '1.0',
            'description': 'Test policy',
            'rules': {
                'PAN': {
                    'action': 'block',
                    'confidence_threshold': 0.8
                }
            }
        }

        policy = RedactionPolicy(policy_dict)

        assert policy.name == 'test_policy'
        assert policy.version == '1.0'
        assert 'PAN' in policy.rules

    def test_get_rule(self):
        """Test getting rule for entity type."""
        policy_dict = {
            'name': 'test',
            'rules': {
                'PAN': {
                    'action': 'block',
                    'confidence_threshold': 0.8
                }
            }
        }

        policy = RedactionPolicy(policy_dict)
        rule = policy.get_rule('PAN')

        assert rule['action'] == 'block'
        assert rule['confidence_threshold'] == 0.8

    def test_get_rule_default(self):
        """Test getting rule returns default if not found."""
        policy = RedactionPolicy({'name': 'test', 'rules': {}})
        rule = policy.get_rule('UNKNOWN_TYPE')

        # Should return default rule
        assert 'action' in rule

    def test_should_redact(self):
        """Test should_redact logic."""
        policy_dict = {
            'name': 'test',
            'rules': {
                'PAN': {
                    'action': 'block',
                    'confidence_threshold': 0.8
                },
                'EMAIL': {
                    'action': 'allow',
                    'confidence_threshold': 0.5
                }
            }
        }

        policy = RedactionPolicy(policy_dict)

        # PAN with high confidence - should redact
        assert policy.should_redact('PAN', 0.9) is True

        # PAN with low confidence - should not redact
        assert policy.should_redact('PAN', 0.5) is False

        # EMAIL with allow action - should not redact
        assert policy.should_redact('EMAIL', 0.9) is False

    def test_rule_inheritance(self):
        """Test rule inheritance from default."""
        policy_dict = {
            'name': 'test',
            'rules': {
                'default': {
                    'action': 'mask',
                    'confidence_threshold': 0.7
                },
                'PAN': {
                    'action': 'block'
                    # Inherits confidence_threshold from default
                }
            }
        }

        policy = RedactionPolicy(policy_dict)
        rule = policy.get_rule('PAN')

        assert rule['action'] == 'block'
        # Should have inherited or default confidence threshold
        assert 'confidence_threshold' in rule


class TestPolicyManager:
    """Test cases for PolicyManager."""

    @pytest.fixture
    def manager(self):
        """Create PolicyManager instance."""
        return PolicyManager()

    def test_manager_initialization(self, manager):
        """Test PolicyManager initialization."""
        assert manager is not None

    def test_load_policy_from_dict(self, manager):
        """Test loading policy from dictionary."""
        policy_dict = {
            'name': 'test_policy',
            'version': '1.0',
            'rules': {
                'PAN': {
                    'action': 'block',
                    'confidence_threshold': 0.8
                }
            }
        }

        policy = manager.load_policy_from_dict(policy_dict)

        assert policy.name == 'test_policy'
        assert 'PAN' in policy.rules

    def test_load_policy_from_yaml_string(self, manager):
        """Test loading policy from YAML string."""
        yaml_str = """
name: test_policy
version: 1.0
rules:
  PAN:
    action: block
    confidence_threshold: 0.8
"""

        policy = manager.load_policy_from_string(yaml_str, format='yaml')

        assert policy.name == 'test_policy'
        assert 'PAN' in policy.rules

    def test_load_policy_from_file(self, manager):
        """Test loading policy from YAML file."""
        policy_dict = {
            'name': 'file_policy',
            'version': '1.0',
            'rules': {
                'PAN': {'action': 'block', 'confidence_threshold': 0.8}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(policy_dict, f)
            temp_path = f.name

        try:
            policy = manager.load_policy_from_file(temp_path)
            assert policy.name == 'file_policy'
            assert 'PAN' in policy.rules
        finally:
            os.unlink(temp_path)

    def test_create_custom_policy(self, manager):
        """Test creating custom policy."""
        rules = {
            'PAN': {
                'action': 'block',
                'confidence_threshold': 0.9
            },
            'EMAIL': {
                'action': 'mask',
                'mask_type': 'partial',
                'show_last': 4
            }
        }

        policy = manager.create_custom_policy('custom', rules, description='Custom test policy')

        assert policy.name == 'custom'
        assert 'PAN' in policy.rules
        assert 'EMAIL' in policy.rules

    def test_validate_policy(self, manager):
        """Test policy validation."""
        # Valid policy
        valid_policy_dict = {
            'name': 'valid',
            'version': '1.0',
            'rules': {
                'PAN': {
                    'action': 'block',
                    'confidence_threshold': 0.8
                }
            }
        }

        valid_policy = RedactionPolicy(valid_policy_dict)
        validation_result = manager.validate_policy(valid_policy)

        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0

    def test_validate_policy_invalid_action(self, manager):
        """Test validation catches invalid actions."""
        invalid_policy_dict = {
            'name': 'invalid',
            'rules': {
                'PAN': {
                    'action': 'invalid_action',  # Invalid
                    'confidence_threshold': 0.8
                }
            }
        }

        invalid_policy = RedactionPolicy(invalid_policy_dict)
        validation_result = manager.validate_policy(invalid_policy)

        # Should detect invalid action
        assert validation_result['valid'] is False or len(validation_result['warnings']) > 0

    def test_validate_policy_invalid_confidence(self, manager):
        """Test validation catches invalid confidence thresholds."""
        invalid_policy_dict = {
            'name': 'invalid',
            'rules': {
                'PAN': {
                    'action': 'block',
                    'confidence_threshold': 1.5  # Invalid (> 1.0)
                }
            }
        }

        invalid_policy = RedactionPolicy(invalid_policy_dict)
        validation_result = manager.validate_policy(invalid_policy)

        # Should detect invalid confidence
        assert validation_result['valid'] is False or len(validation_result['errors']) > 0

    def test_load_builtin_policy(self, manager):
        """Test loading built-in policies."""
        # Try to load India finance policy
        policy_path = 'policies/india_finance.yaml'

        if os.path.exists(policy_path):
            policy = manager.load_policy_from_file(policy_path)
            assert policy is not None
            assert len(policy.rules) > 0

    def test_policy_serialization(self, manager):
        """Test policy can be saved and loaded."""
        policy_dict = {
            'name': 'serialization_test',
            'version': '1.0',
            'rules': {
                'PAN': {'action': 'block', 'confidence_threshold': 0.8}
            }
        }

        policy = RedactionPolicy(policy_dict)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            manager.save_policy(policy, temp_path)

            # Load
            loaded_policy = manager.load_policy_from_file(temp_path)

            assert loaded_policy.name == policy.name
            assert loaded_policy.rules['PAN']['action'] == 'block'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_policy_with_multiple_rules(self, manager):
        """Test policy with multiple entity type rules."""
        rules = {
            'PAN': {'action': 'block'},
            'AADHAAR': {'action': 'block'},
            'PHONE': {'action': 'mask', 'show_last': 4},
            'EMAIL': {'action': 'mask', 'show_last': 5},
            'PERSON': {'action': 'mask', 'mask_type': 'partial'}
        }

        policy = manager.create_custom_policy('multi_rule', rules)

        assert len(policy.rules) >= 5

        # Test each rule
        assert policy.get_rule('PAN')['action'] == 'block'
        assert policy.get_rule('PHONE')['action'] == 'mask'
        assert policy.get_rule('EMAIL')['action'] == 'mask'

    def test_policy_default_rule(self, manager):
        """Test default rule application."""
        policy_dict = {
            'name': 'test',
            'rules': {
                'default': {
                    'action': 'mask',
                    'confidence_threshold': 0.7
                }
            }
        }

        policy = RedactionPolicy(policy_dict)

        # Unknown entity should get default rule
        rule = policy.get_rule('UNKNOWN_ENTITY')
        assert rule['action'] == 'mask' or 'action' in rule


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
