#!/usr/bin/env python3
"""
Test script for Streamlit Web UI features
Tests the helper functions without running the full Streamlit app
"""

import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

# Import helper functions
from app.ui.streamlit_app_enhanced import (
    get_available_policies,
    load_policy_file,
    highlight_entities_in_text,
    generate_audit_report
)

def test_get_available_policies():
    """Test policy discovery"""
    print("=" * 60)
    print("TEST 1: Get Available Policies")
    print("=" * 60)

    policies = get_available_policies()
    print(f"✓ Found {len(policies)} policies:")
    for policy in policies:
        print(f"  - {policy}")
    print()
    return len(policies) > 0

def test_load_policy_file():
    """Test policy loading"""
    print("=" * 60)
    print("TEST 2: Load Policy File")
    print("=" * 60)

    policies = get_available_policies()
    if not policies:
        print("✗ No policies found to test")
        return False

    policy_name = policies[0]
    print(f"Loading policy: {policy_name}")

    policy_manager = load_policy_file(policy_name)
    if policy_manager:
        print(f"✓ Successfully loaded policy: {policy_name}")
        summary = policy_manager.get_policy_summary()
        print(f"  Policy has {len(summary.get('entities', []))} entity rules")
        return True
    else:
        print(f"✗ Failed to load policy: {policy_name}")
        return False

def test_highlight_entities():
    """Test entity highlighting"""
    print("=" * 60)
    print("TEST 3: Entity Highlighting")
    print("=" * 60)

    text = "My name is Rajesh Kumar and my email is rajesh@example.com"
    entities = [
        {
            'entity_type': 'PERSON',
            'text': 'Rajesh Kumar',
            'start': 11,
            'end': 23,
            'confidence': 0.95
        },
        {
            'entity_type': 'EMAIL',
            'text': 'rajesh@example.com',
            'start': 40,
            'end': 58,
            'confidence': 0.99
        }
    ]

    highlighted = highlight_entities_in_text(text, entities)

    if '<span' in highlighted and 'background-color' in highlighted:
        print("✓ Entity highlighting generated HTML spans")
        print(f"\nOriginal text:\n{text}")
        print(f"\nHighlighted HTML (first 200 chars):\n{highlighted[:200]}...")
        return True
    else:
        print("✗ Entity highlighting failed")
        return False

def test_generate_audit_report():
    """Test audit report generation"""
    print("=" * 60)
    print("TEST 4: Audit Report Generation")
    print("=" * 60)

    original_text = "My PAN is ABCDE1234F and phone is 9876543210"
    redacted_text = "My PAN is ██████████ and phone is ██████████"

    findings = [
        {
            'entity_type': 'PAN',
            'text': 'ABCDE1234F',
            'start': 10,
            'end': 20,
            'confidence': 0.92,
            'source': 'regex'
        },
        {
            'entity_type': 'PHONE',
            'text': '9876543210',
            'start': 35,
            'end': 45,
            'confidence': 0.88,
            'source': 'presidio'
        }
    ]

    report = generate_audit_report(
        findings,
        original_text,
        redacted_text,
        "test_document.txt"
    )

    if 'metadata' in report and 'statistics' in report and 'detected_entities' in report:
        print("✓ Audit report generated successfully")
        print(f"\n  - Total entities: {report['statistics']['total_entities_found']}")
        print(f"  - Entity types: {list(report['statistics']['entities_by_type'].keys())}")
        print(f"  - Report keys: {list(report.keys())}")
        return True
    else:
        print("✗ Audit report generation failed")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("STREAMLIT WEB UI - FEATURE TESTS")
    print("=" * 60)
    print()

    results = []
    results.append(("Policy Discovery", test_get_available_policies()))
    print()
    results.append(("Policy Loading", test_load_policy_file()))
    print()
    results.append(("Entity Highlighting", test_highlight_entities()))
    print()
    results.append(("Audit Report", test_generate_audit_report()))
    print()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)
    print()

    if passed == total:
        print("🎉 All tests passed! Web UI features are ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
