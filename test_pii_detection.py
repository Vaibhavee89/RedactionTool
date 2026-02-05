#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced PII detection engine.
"""

import sys
import os

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_sample_text():
    """Create sample text with various PII types."""
    return """
    Personal Information:
    Name: Rajesh Kumar
    Email: rajesh.kumar@example.com
    Phone: +91-9876543210 or 9876543210
    Alternate Phone: +1-555-123-4567

    Government IDs:
    PAN Card: ABCDE1234F
    Aadhaar: 1234 5678 9012
    Voter ID: ABC1234567
    Driving License: KA-12-2020-1234567
    Passport: A1234567

    Financial Information:
    Bank Account: 1234567890123456
    IFSC Code: ABCD0123456
    Credit Card: 4532 1234 5678 9012
    SSN: 123-45-6789

    Other Information:
    Date of Birth: 15/08/1990
    Date: 2024-01-15
    Address: 123 MG Road, Bangalore, Karnataka 560001
    Vehicle: KA-01-AB-1234
    Medical Record: MRN-123456789

    Organization: Acme Corporation
    Location: Mumbai, Maharashtra
    """


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports...")
    print("="*60)

    tests = []

    try:
        from app.services.pii.enhanced_ner_provider import EnhancedNERProvider
        print("✅ EnhancedNERProvider: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ EnhancedNERProvider: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.pii.enhanced_regex_provider import EnhancedRegexProvider
        print("✅ EnhancedRegexProvider: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ EnhancedRegexProvider: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.pii.enhanced_presidio_provider import EnhancedPresidioProvider
        print("✅ EnhancedPresidioProvider: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ EnhancedPresidioProvider: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.pii.custom_presidio_recognizers import get_all_custom_recognizers
        print("✅ Custom Presidio Recognizers: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ Custom Presidio Recognizers: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.pii.ensemble_detector import EnsembleDetector
        print("✅ EnsembleDetector: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ EnsembleDetector: FAIL - {e}")
        tests.append(False)

    passed = sum(tests)
    failed = len(tests) - passed
    print(f"\n📊 Results: {passed} passed, {failed} failed")

    return all(tests)


def test_regex_provider():
    """Test enhanced regex provider."""
    print("\n" + "="*60)
    print("Testing Enhanced Regex Provider...")
    print("="*60)

    try:
        from app.services.pii.enhanced_regex_provider import EnhancedRegexProvider

        provider = EnhancedRegexProvider()
        text = test_sample_text()

        entities = provider.detect(text)

        print(f"\n✅ Detected {len(entities)} entities")

        # Group by type
        by_type = {}
        for entity in entities:
            entity_type = entity['entity_type']
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity)

        print("\n📊 Entities by type:")
        for entity_type, ents in sorted(by_type.items()):
            print(f"  {entity_type}: {len(ents)}")
            for ent in ents[:2]:  # Show first 2
                print(f"    - {ent['text']} (confidence: {ent['confidence']:.2f})")

        # Get statistics
        stats = provider.get_statistics(text)
        print(f"\n📈 Statistics:")
        for entity_type, count in sorted(stats.items()):
            print(f"  {entity_type}: {count}")

        return len(entities) > 0

    except Exception as e:
        print(f"\n❌ Regex provider test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ner_provider():
    """Test enhanced NER provider."""
    print("\n" + "="*60)
    print("Testing Enhanced NER Provider...")
    print("="*60)

    try:
        from app.services.pii.enhanced_ner_provider import EnhancedNERProvider

        provider = EnhancedNERProvider(load_hindi=False)  # Skip Hindi for speed
        text = test_sample_text()

        entities = provider.detect(text, language='en')

        print(f"\n✅ Detected {len(entities)} entities")

        # Show detected entities
        print("\n📊 Detected entities:")
        for entity in entities[:10]:  # Show first 10
            print(f"  {entity['entity_type']}: {entity['text']} (conf: {entity['confidence']:.2f})")

        # Get supported entity types
        supported = provider.get_supported_entity_types('en')
        print(f"\n📋 Supported entity types: {len(supported)}")

        return len(entities) > 0

    except Exception as e:
        print(f"\n❌ NER provider test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_presidio_provider():
    """Test enhanced Presidio provider."""
    print("\n" + "="*60)
    print("Testing Enhanced Presidio Provider...")
    print("="*60)

    try:
        from app.services.pii.enhanced_presidio_provider import EnhancedPresidioProvider

        provider = EnhancedPresidioProvider(custom_recognizers=True)
        text = test_sample_text()

        entities = provider.detect(text, language='en')

        print(f"\n✅ Detected {len(entities)} entities")

        # Show detected entities
        print("\n📊 Detected entities:")
        by_type = {}
        for entity in entities:
            entity_type = entity['entity_type']
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity)

        for entity_type, ents in sorted(by_type.items()):
            print(f"  {entity_type}: {len(ents)}")
            for ent in ents[:2]:  # Show first 2
                print(f"    - {ent['text']} (conf: {ent['confidence']:.2f})")

        # Get supported entities
        supported = provider.get_supported_entities()
        print(f"\n📋 Supported entities: {', '.join(supported[:15])}...")

        return len(entities) > 0

    except Exception as e:
        print(f"\n❌ Presidio provider test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_detector():
    """Test ensemble detector with conflict resolution."""
    print("\n" + "="*60)
    print("Testing Ensemble Detector...")
    print("="*60)

    try:
        from app.services.pii.ensemble_detector import EnsembleDetector

        detector = EnsembleDetector(
            use_ner=True,
            use_regex=True,
            use_presidio=True,
            load_hindi=False  # Skip Hindi for speed
        )

        text = test_sample_text()

        # Detect with provenance
        result = detector.detect_with_provenance(text, language='en')

        print(f"\n✅ Merged: {len(result['merged_results'])} entities")

        # Show statistics
        stats = result['statistics']
        print(f"\n📊 Detection Statistics:")
        print(f"  Total entities (merged): {stats['total_entities']}")
        print(f"\n  By Provider:")
        for provider, count in stats['by_provider'].items():
            print(f"    {provider}: {count}")

        print(f"\n  By Type:")
        for entity_type, count in sorted(stats['by_type'].items()):
            print(f"    {entity_type}: {count}")

        print(f"\n  Confidence Distribution:")
        conf_dist = stats['confidence_distribution']
        print(f"    High (>=0.8): {conf_dist['high']}")
        print(f"    Medium (>=0.5): {conf_dist['medium']}")
        print(f"    Low (<0.5): {conf_dist['low']}")

        # Show some merged results
        print(f"\n📋 Sample Merged Results:")
        for entity in result['merged_results'][:15]:
            print(f"  {entity['entity_type']}: {entity['text']}")
            print(f"    Source: {entity['source']}, Confidence: {entity['confidence']:.2f}")

        # Benchmark
        print(f"\n⏱️ Benchmarking providers...")
        benchmark = detector.benchmark_providers(text)
        print(f"  Results:")
        for provider, metrics in benchmark.items():
            print(f"    {provider}: {metrics['time_ms']:.2f}ms ({metrics['count']} entities)")

        return len(result['merged_results']) > 0

    except Exception as e:
        print(f"\n❌ Ensemble detector test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_recognizers():
    """Test custom Presidio recognizers."""
    print("\n" + "="*60)
    print("Testing Custom Presidio Recognizers...")
    print("="*60)

    try:
        from app.services.pii.custom_presidio_recognizers import get_all_custom_recognizers

        recognizers = get_all_custom_recognizers()

        print(f"\n✅ Loaded {len(recognizers)} custom recognizers")

        print("\n📋 Custom Recognizers:")
        for recognizer in recognizers:
            entity_type = getattr(recognizer, 'supported_entity', 'UNKNOWN')
            print(f"  - {entity_type}")

        # Test PAN recognition
        from app.services.pii.custom_presidio_recognizers import PANRecognizer
        pan_recognizer = PANRecognizer()
        print(f"\n✅ PAN Recognizer loaded")
        if hasattr(pan_recognizer, 'supported_entities'):
            print(f"   Supported entities: {pan_recognizer.supported_entities}")
        elif hasattr(pan_recognizer, 'supported_entity'):
            print(f"   Supported entity: {pan_recognizer.supported_entity}")

        return len(recognizers) > 0

    except Exception as e:
        print(f"\n❌ Custom recognizers test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("🧠 PII Detection Engine - Comprehensive Test Suite")
    print("="*60 + "\n")

    results = []

    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Custom Recognizers", test_custom_recognizers()))
    results.append(("Regex Provider", test_regex_provider()))
    results.append(("NER Provider", test_ner_provider()))
    results.append(("Presidio Provider", test_presidio_provider()))
    results.append(("Ensemble Detector", test_ensemble_detector()))

    # Summary
    print("\n" + "="*60)
    print("📋 FINAL SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("PII Detection Engine is working correctly.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Check the errors above for details.")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
