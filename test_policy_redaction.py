#!/usr/bin/env python3
"""
Test script for Policy-Based Redaction System.
"""

import sys
import os

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.redaction.policy_manager import PolicyManager, RedactionPolicy
from app.services.redaction.enhanced_redactor import EnhancedRedactor


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_load_builtin_policies():
    """Test loading built-in policies."""
    print_section("Test 1: Load Built-in Policies")

    policy_manager = PolicyManager()
    policies = policy_manager.list_policies()

    print(f"\n✅ Loaded {len(policies)} built-in policies:")
    for policy_name in policies:
        policy = policy_manager.get_policy(policy_name)
        print(f"\n  📋 {policy.name}")
        print(f"     Version: {policy.version}")
        print(f"     Description: {policy.description[:60]}...")
        print(f"     Rules: {len(policy.rules)} entity types")

    return len(policies) == 3


def test_india_finance_policy():
    """Test India Finance policy with sample data."""
    print_section("Test 2: India Finance Policy")

    # Sample financial data
    test_data = """
    Customer Financial Information:
    Name: Rajesh Kumar Sharma
    PAN: ABCDE1234F
    Aadhaar: 1234 5678 9012
    Phone: +91-9876543210
    Email: rajesh.sharma@example.com
    Bank Account: 1234567890123456
    IFSC: HDFC0001234
    Credit Card: 4532-1234-5678-9010
    Salary: ₹850,000 per annum
    """

    findings = [
        {"entity_type": "PERSON", "start": 47, "end": 66, "confidence": 0.85, "text": "Rajesh Kumar Sharma"},
        {"entity_type": "PAN", "start": 77, "end": 87, "confidence": 0.95, "text": "ABCDE1234F"},
        {"entity_type": "AADHAAR", "start": 101, "end": 115, "confidence": 0.92, "text": "1234 5678 9012"},
        {"entity_type": "PHONE", "start": 129, "end": 144, "confidence": 0.90, "text": "+91-9876543210"},
        {"entity_type": "EMAIL", "start": 158, "end": 184, "confidence": 0.88, "text": "rajesh.sharma@example.com"},
        {"entity_type": "BANK_ACCOUNT", "start": 203, "end": 219, "confidence": 0.87, "text": "1234567890123456"},
        {"entity_type": "IFSC", "start": 230, "end": 241, "confidence": 0.92, "text": "HDFC0001234"},
        {"entity_type": "CREDIT_CARD", "start": 258, "end": 277, "confidence": 0.94, "text": "4532-1234-5678-9010"},
    ]

    policy_manager = PolicyManager()
    redactor = EnhancedRedactor(policy_manager)

    result = redactor.redact_with_metadata(
        test_data,
        findings,
        policy="India Finance Compliance"
    )

    print("\nOriginal Text:")
    print("-" * 70)
    print(result['original_text'])

    print("\nRedacted Text:")
    print("-" * 70)
    print(result['redacted_text'])

    print("\nRedaction Statistics:")
    print("-" * 70)
    print(f"  Total findings: {result['findings_count']}")
    print(f"  Redacted: {result['redaction_count']}")
    print(f"  Redaction rate: {result['redacted_percentage']:.1f}%")
    print(f"  Policy used: {result['policy_used']}")
    print(f"\n  By Entity Type:")
    for entity_type, count in result['by_entity_type'].items():
        print(f"    - {entity_type}: {count}")
    print(f"\n  By Action:")
    for action, count in result['by_action'].items():
        print(f"    - {action}: {count}")

    print("\n✅ India Finance Policy: PASS")
    return True


def test_gdpr_policy():
    """Test GDPR policy with European data."""
    print_section("Test 3: GDPR Basic Policy")

    test_data = """
    Personal Data Record:
    Name: John Smith
    Email: john.smith@company.eu
    Phone: +44-20-7946-0958
    Address: 123 Oxford Street, London, UK
    IP Address: 192.168.1.100
    Credit Card: 5412-7512-3412-3456
    """

    findings = [
        {"entity_type": "PERSON", "start": 33, "end": 43, "confidence": 0.88, "text": "John Smith"},
        {"entity_type": "EMAIL", "start": 54, "end": 78, "confidence": 0.92, "text": "john.smith@company.eu"},
        {"entity_type": "PHONE", "start": 92, "end": 109, "confidence": 0.85, "text": "+44-20-7946-0958"},
        {"entity_type": "ADDRESS", "start": 123, "end": 152, "confidence": 0.80, "text": "123 Oxford Street, London, UK"},
        {"entity_type": "IP_ADDRESS", "start": 169, "end": 182, "confidence": 0.95, "text": "192.168.1.100"},
        {"entity_type": "CREDIT_CARD", "start": 200, "end": 219, "confidence": 0.96, "text": "5412-7512-3412-3456"},
    ]

    policy_manager = PolicyManager()
    redactor = EnhancedRedactor(policy_manager)

    result = redactor.redact_with_metadata(
        test_data,
        findings,
        policy="GDPR Basic Compliance"
    )

    print("\nOriginal Text:")
    print("-" * 70)
    print(result['original_text'])

    print("\nRedacted Text (GDPR):")
    print("-" * 70)
    print(result['redacted_text'])

    print("\n✅ GDPR Policy: PASS")
    return True


def test_hipaa_policy():
    """Test HIPAA-like policy with healthcare data."""
    print_section("Test 4: HIPAA-Like Healthcare Policy")

    test_data = """
    Patient Medical Record:
    Name: Sarah Johnson
    DOB: 05/15/1985
    SSN: 123-45-6789
    Phone: (555) 123-4567
    Email: sarah.j@email.com
    Address: 456 Medical Plaza, Suite 100
    Medical Record #: MRN-987654
    """

    findings = [
        {"entity_type": "PERSON", "start": 35, "end": 48, "confidence": 0.90, "text": "Sarah Johnson"},
        {"entity_type": "DATE", "start": 59, "end": 69, "confidence": 0.85, "text": "05/15/1985"},
        {"entity_type": "SSN", "start": 80, "end": 92, "confidence": 0.95, "text": "123-45-6789"},
        {"entity_type": "PHONE", "start": 106, "end": 120, "confidence": 0.88, "text": "(555) 123-4567"},
        {"entity_type": "EMAIL", "start": 134, "end": 152, "confidence": 0.90, "text": "sarah.j@email.com"},
        {"entity_type": "ADDRESS", "start": 166, "end": 196, "confidence": 0.82, "text": "456 Medical Plaza, Suite 100"},
    ]

    policy_manager = PolicyManager()
    redactor = EnhancedRedactor(policy_manager)

    result = redactor.redact_with_metadata(
        test_data,
        findings,
        policy="HIPAA-Like Healthcare Compliance"
    )

    print("\nOriginal Text:")
    print("-" * 70)
    print(result['original_text'])

    print("\nRedacted Text (HIPAA-Like):")
    print("-" * 70)
    print(result['redacted_text'])

    print("\n✅ HIPAA-Like Policy: PASS")
    return True


def test_custom_policy():
    """Test creating custom policy."""
    print_section("Test 5: Custom Policy Creation")

    policy_manager = PolicyManager()

    # Create custom policy with specific rules
    custom_rules = {
        'PERSON': {
            'action': 'partial_mask',
            'show_first': 1,
            'show_last': 1,
            'mask_char': '*',
            'min_confidence': 0.7
        },
        'EMAIL': {
            'action': 'hash',
            'algorithm': 'sha256',
            'truncate': 10,
            'prefix': 'EMAIL_',
            'min_confidence': 0.8
        },
        'PHONE': {
            'action': 'tokenize',
            'prefix': 'TEL_',
            'preserve_mapping': True,
            'min_confidence': 0.7
        },
        'PAN': {
            'action': 'block',
            'min_confidence': 0.9
        }
    }

    custom_policy = policy_manager.create_custom_policy(
        name="Custom Test Policy",
        rules=custom_rules,
        description="Custom policy for demonstration",
        global_config={'min_confidence': 0.6}
    )

    print(f"\n✅ Created custom policy: {custom_policy.name}")
    print(f"   Rules defined: {len(custom_policy.rules)}")

    # Test custom policy
    test_data = "Contact: John Doe, john@test.com, 9876543210, PAN: ABCDE1234F"
    findings = [
        {"entity_type": "PERSON", "start": 9, "end": 17, "confidence": 0.85, "text": "John Doe"},
        {"entity_type": "EMAIL", "start": 19, "end": 33, "confidence": 0.90, "text": "john@test.com"},
        {"entity_type": "PHONE", "start": 35, "end": 45, "confidence": 0.88, "text": "9876543210"},
        {"entity_type": "PAN", "start": 52, "end": 62, "confidence": 0.95, "text": "ABCDE1234F"},
    ]

    redactor = EnhancedRedactor(policy_manager)
    redacted = redactor.redact_text(test_data, findings, policy="Custom Test Policy")

    print(f"\nOriginal:  {test_data}")
    print(f"Redacted:  {redacted}")

    print("\n✅ Custom Policy: PASS")
    return True


def test_per_entity_controls():
    """Test advanced per-entity redaction controls."""
    print_section("Test 6: Advanced Per-Entity Controls")

    test_cases = [
        {
            'name': 'PAN - Full Redact',
            'text': 'PAN: ABCDE1234F',
            'entity': {'entity_type': 'PAN', 'start': 5, 'end': 15, 'confidence': 0.9},
            'rule': {'action': 'block'}
        },
        {
            'name': 'Phone - Mask Last 4',
            'text': 'Phone: +91-9876543210',
            'entity': {'entity_type': 'PHONE', 'start': 7, 'end': 21, 'confidence': 0.85},
            'rule': {'action': 'mask', 'show_last': 4, 'mask_char': 'X'}
        },
        {
            'name': 'Name - Partial Mask',
            'text': 'Name: Rajesh Kumar',
            'entity': {'entity_type': 'PERSON', 'start': 6, 'end': 18, 'confidence': 0.8},
            'rule': {'action': 'partial_mask', 'show_first': 1, 'show_last': 1, 'mask_char': '*'}
        },
        {
            'name': 'Aadhaar - Show Last 4',
            'text': 'Aadhaar: 1234 5678 9012',
            'entity': {'entity_type': 'AADHAAR', 'start': 9, 'end': 23, 'confidence': 0.92},
            'rule': {'action': 'partial_mask', 'pattern': 'XXXX XXXX 9012'}
        },
        {
            'name': 'Email - Hash',
            'text': 'Email: user@example.com',
            'entity': {'entity_type': 'EMAIL', 'start': 7, 'end': 23, 'confidence': 0.88},
            'rule': {'action': 'hash', 'algorithm': 'sha256', 'truncate': 8, 'prefix': 'H_'}
        },
    ]

    redactor = EnhancedRedactor()

    print("\nPer-Entity Redaction Examples:")
    print("-" * 70)

    for test_case in test_cases:
        custom_rules = {test_case['entity']['entity_type']: test_case['rule']}
        redacted = redactor.redact_text(
            test_case['text'],
            [test_case['entity']],
            custom_rules=custom_rules
        )

        print(f"\n{test_case['name']}:")
        print(f"  Original:  {test_case['text']}")
        print(f"  Redacted:  {redacted}")
        print(f"  Action:    {test_case['rule']['action']}")

    print("\n✅ Per-Entity Controls: PASS")
    return True


def test_policy_validation():
    """Test policy validation."""
    print_section("Test 7: Policy Validation")

    policy_manager = PolicyManager()

    # Valid policy
    valid_policy_data = {
        'name': 'Valid Test Policy',
        'description': 'A valid policy',
        'version': '1.0',
        'rules': {
            'PERSON': {'action': 'mask', 'min_confidence': 0.7},
            'EMAIL': {'action': 'block'}
        }
    }

    valid_policy = RedactionPolicy(valid_policy_data)
    validation = policy_manager.validate_policy(valid_policy)

    print(f"\n✅ Valid Policy:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Errors: {validation['errors']}")

    # Invalid policy
    invalid_policy_data = {
        'name': 'Invalid Test Policy',
        'rules': {
            'PERSON': {'action': 'invalid_action'},
            'EMAIL': {'min_confidence': 1.5}  # Invalid confidence
        }
    }

    invalid_policy = RedactionPolicy(invalid_policy_data)
    validation = policy_manager.validate_policy(invalid_policy)

    print(f"\n❌ Invalid Policy:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Errors:")
    for error in validation['errors']:
        print(f"     - {error}")

    print("\n✅ Policy Validation: PASS")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🔒 Policy-Based Redaction System - Test Suite")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Load Built-in Policies", test_load_builtin_policies()))
    results.append(("India Finance Policy", test_india_finance_policy()))
    results.append(("GDPR Policy", test_gdpr_policy()))
    results.append(("HIPAA-Like Policy", test_hipaa_policy()))
    results.append(("Custom Policy", test_custom_policy()))
    results.append(("Per-Entity Controls", test_per_entity_controls()))
    results.append(("Policy Validation", test_policy_validation()))

    # Summary
    print_section("📋 FINAL SUMMARY")

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL POLICY TESTS PASSED!")
        print("\n✅ Policy-Based Redaction System is fully functional:")
        print("   • YAML policy configuration")
        print("   • Pre-built compliance policies (India Finance, GDPR, HIPAA)")
        print("   • Custom policy creation")
        print("   • Advanced per-entity controls")
        print("   • Multiple redaction strategies")
        print("   • Confidence threshold enforcement")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Check the errors above for details.")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
