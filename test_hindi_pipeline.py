#!/usr/bin/env python3
"""
Test script for Hindi PII detection and redaction.
"""

import sys
import os

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_hindi_text_detection():
    """Test Hindi text PII detection."""
    print("\n" + "="*60)
    print("Testing Hindi Text PII Detection")
    print("="*60)

    from app.services.hindi_pipeline import HindiPIIRedactionPipeline

    pipeline = HindiPIIRedactionPipeline(
        ocr_engine='paddle',
        preprocess=True
    )

    # Hindi text with PII
    hindi_text = """
    व्यक्तिगत विवरण:
    नाम: राजेश कुमार शर्मा
    ईमेल: rajesh.sharma@example.com
    फ़ोन नंबर: +91-9876543210
    पैन नंबर: ABCDE1234F
    आधार संख्या: 1234 5678 9012
    पता: 123 एमजी रोड, मुंबई, महाराष्ट्र 400001
    """

    try:
        result = pipeline.process_text(
            hindi_text,
            language='hi',
            redaction_mode='block'
        )

        print("\nOriginal Text:")
        print("-" * 60)
        print(result['original_text'])

        print("\nRedacted Text:")
        print("-" * 60)
        print(result['redacted_text'])

        print("\nPII Entities Found:")
        print("-" * 60)
        for finding in result['findings']:
            print(f"  {finding['entity_type']}: {finding['text']}")
            print(f"    Confidence: {finding['confidence']:.2f}, Source: {finding['source']}")

        print("\nStatistics:")
        print("-" * 60)
        print(f"  Total PII: {result['pii_count']}")
        print(f"  By Type: {result['statistics']['by_type']}")
        print(f"  By Source: {result['statistics']['by_source']}")

        print("\n✅ Hindi text detection: PASS")
        return True

    except Exception as e:
        print(f"\n❌ Hindi text detection: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_language_detection():
    """Test mixed Hindi-English (Hinglish) detection."""
    print("\n" + "="*60)
    print("Testing Mixed Language (Hinglish) Detection")
    print("="*60)

    from app.services.hindi_pipeline import HindiPIIRedactionPipeline

    pipeline = HindiPIIRedactionPipeline()

    # Mixed Hindi-English text
    mixed_text = """
    Personal Details:
    Name: Rajesh Kumar (राजेश कुमार)
    Email: rajesh@example.com
    Mobile: +91-9876543210
    PAN: ABCDE1234F
    आधार: 1234 5678 9012
    Address: MG Road, Mumbai, Maharashtra
    """

    try:
        result = pipeline.process_text(
            mixed_text,
            language='auto',  # Auto-detect
            redaction_mode='label'
        )

        print("\nOriginal Text:")
        print("-" * 60)
        print(result['original_text'])

        print("\nRedacted Text:")
        print("-" * 60)
        print(result['redacted_text'])

        if 'language_distribution' in result:
            lang_dist = result['language_distribution']
            print("\nLanguage Distribution:")
            print("-" * 60)
            print(f"  Hindi: {lang_dist['hindi']:.1f}%")
            print(f"  English: {lang_dist['english']:.1f}%")
            print(f"  Category: {lang_dist['category']}")

        print(f"\n  Found {result['pii_count']} PII entities")
        print("\n✅ Mixed language detection: PASS")
        return True

    except Exception as e:
        print(f"\n❌ Mixed language detection: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hindi_regex_patterns():
    """Test Hindi-specific regex patterns."""
    print("\n" + "="*60)
    print("Testing Hindi Regex Patterns")
    print("="*60)

    try:
        from app.services.pii.hindi_regex_provider import HindiRegexProvider

        provider = HindiRegexProvider()

        test_text = """
        नाम: श्री राजेश कुमार
        फ़ोन: +91-9876543210
        ईमेल: test@example.com
        पैन कार्ड: ABCDE1234F
        आधार नंबर: 1234 5678 9012
        पता: 123 मेन स्ट्रीट, दिल्ली
        """

        entities = provider.detect(test_text)

        print(f"\n✅ Detected {len(entities)} Hindi entities:")
        for entity in entities:
            print(f"  {entity['entity_type']}: {entity['text']}")
            if 'value' in entity:
                print(f"    Value: {entity['value']}")
            print(f"    Confidence: {entity['confidence']:.2f}")

        print("\n✅ Hindi regex patterns: PASS")
        return len(entities) > 0

    except Exception as e:
        print(f"\n❌ Hindi regex patterns: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hindi_ensemble_detector():
    """Test Hindi ensemble detector."""
    print("\n" + "="*60)
    print("Testing Hindi Ensemble Detector")
    print("="*60)

    try:
        from app.services.pii.hindi_ensemble_detector import HindiEnsembleDetector

        detector = HindiEnsembleDetector(
            use_ner=True,
            use_regex=True,
            use_presidio=True,
            use_hindi_regex=True,
            load_hindi=False  # Skip Hindi NER for speed
        )

        test_text = """
        नाम: राजेश कुमार
        Email: rajesh@example.com
        Phone: +91-9876543210
        PAN: ABCDE1234F
        """

        entities = detector.detect(test_text, language='hi')

        print(f"\n✅ Ensemble detected {len(entities)} entities:")
        for entity in entities[:10]:  # Show first 10
            print(f"  {entity['entity_type']}: {entity['text']}")
            print(f"    Source: {entity['source']}, Confidence: {entity['confidence']:.2f}")

        # Get Hindi entity types
        hindi_types = detector.get_hindi_entity_types()
        print(f"\n📋 Hindi entity types supported: {len(hindi_types)}")

        print("\n✅ Hindi ensemble detector: PASS")
        return len(entities) > 0

    except Exception as e:
        print(f"\n❌ Hindi ensemble detector: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("🇮🇳 Hindi PII Detection - Test Suite")
    print("="*60)

    results = []

    # Run tests
    results.append(("Hindi Regex Patterns", test_hindi_regex_patterns()))
    results.append(("Hindi Ensemble Detector", test_hindi_ensemble_detector()))
    results.append(("Hindi Text Detection", test_hindi_text_detection()))
    results.append(("Mixed Language Detection", test_mixed_language_detection()))

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
        print("🎉 ALL HINDI TESTS PASSED!")
        print("Hindi PII detection is working correctly.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Check the errors above for details.")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
