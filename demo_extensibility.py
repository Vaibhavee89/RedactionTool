"""
Demo Script for Extensibility Framework

This script demonstrates the plugin system and enhanced ensemble detector.
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_plugin_discovery():
    """Demonstrate plugin auto-discovery."""
    print("\n" + "="*60)
    print("DEMO 1: Plugin Auto-Discovery")
    print("="*60)

    from app.extensions.registry.plugin_registry import get_plugin_registry

    registry = get_plugin_registry()

    # Discover plugins
    logger.info("Discovering plugins from plugins/detectors/...")
    discovered = registry.discover_plugins("plugins/detectors")

    print(f"\n✓ Discovered {len(discovered)} plugins:")
    for plugin_name in discovered:
        metadata = registry.get_metadata(plugin_name)
        print(f"  - {metadata.name} v{metadata.version}")
        print(f"    Description: {metadata.description}")
        print(f"    Entity Types: {', '.join(metadata.supported_entity_types[:3])}...")
        print(f"    Priority: {metadata.priority}")
        print()

    # Get stats
    stats = registry.get_stats()
    print(f"Registry Statistics:")
    print(f"  Total Plugins: {stats['total_plugins']}")
    print(f"  Enabled: {stats['enabled_plugins']}")
    print(f"  Disabled: {stats['disabled_plugins']}")


def demo_crypto_detection():
    """Demonstrate cryptocurrency detection."""
    print("\n" + "="*60)
    print("DEMO 2: Cryptocurrency Detection")
    print("="*60)

    from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

    # Create detector with plugins enabled
    detector = EnhancedEnsembleDetector(
        use_ner=False,
        use_regex=False,
        use_presidio=False,
        enable_plugins=True
    )

    # Test crypto addresses
    test_texts = [
        "Please send payment to Bitcoin address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "My Ethereum wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "Litecoin address for donations: LMXn6vT3cqXqJQQaCQSqQ5QL9qJbZzKqQw"
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        results = detector.detect(text)
        print(f"Found {len(results)} entities:")
        for entity in results:
            print(f"  - Type: {entity['entity_type']}")
            print(f"    Text: {entity['text']}")
            print(f"    Confidence: {entity['confidence']:.2f}")
            print(f"    Source: {entity['source']}")


def demo_medical_codes_detection():
    """Demonstrate medical codes detection."""
    print("\n" + "="*60)
    print("DEMO 3: Medical Codes Detection")
    print("="*60)

    from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

    detector = EnhancedEnsembleDetector(
        use_ner=False,
        use_regex=False,
        use_presidio=False,
        enable_plugins=True
    )

    # Test medical codes
    test_texts = [
        "Patient diagnosed with ICD-10 code J45.909 (asthma)",
        "Procedure performed: CPT code 99213 for office visit",
        "Prescribed medication NDC: 0002-3229-02"
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        results = detector.detect(text)
        print(f"Found {len(results)} entities:")
        for entity in results:
            print(f"  - Type: {entity['entity_type']}")
            print(f"    Text: {entity['text']}")
            print(f"    Confidence: {entity['confidence']:.2f}")
            metadata = entity.get('metadata', {})
            if 'code_name' in metadata:
                print(f"    Code Name: {metadata['code_name']}")


def demo_custom_regex_detection():
    """Demonstrate custom regex patterns."""
    print("\n" + "="*60)
    print("DEMO 4: Custom Regex Patterns")
    print("="*60)

    from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

    detector = EnhancedEnsembleDetector(
        use_ner=False,
        use_regex=False,
        use_presidio=False,
        enable_plugins=True
    )

    # Test custom patterns (from default config)
    test_texts = [
        "Employee EMP-123456 submitted the report",
        "Project PROJ-ABC-1234 is behind schedule",
        "Support ticket TICKET-98765432 has been resolved"
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        results = detector.detect(text)
        print(f"Found {len(results)} entities:")
        for entity in results:
            print(f"  - Type: {entity['entity_type']}")
            print(f"    Text: {entity['text']}")
            print(f"    Confidence: {entity['confidence']:.2f}")
            metadata = entity.get('metadata', {})
            if 'pattern_name' in metadata:
                print(f"    Pattern: {metadata['pattern_name']}")


def demo_enhanced_provenance():
    """Demonstrate enhanced provenance tracking."""
    print("\n" + "="*60)
    print("DEMO 5: Enhanced Provenance Tracking")
    print("="*60)

    from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

    detector = EnhancedEnsembleDetector(
        use_ner=True,
        use_regex=True,
        use_presidio=True,
        enable_plugins=True
    )

    text = """
    Employee EMP-123456 (John Doe) at john@example.com sent Bitcoin payment
    1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa for ICD-10 code J45.909 treatment.
    Contact: +1-555-123-4567
    """

    print(f"Analyzing text with ALL detectors enabled...")
    result = detector.detect_with_provenance_enhanced(text.strip())

    print(f"\n📊 Detection Statistics:")
    stats = result['statistics']
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"\n  By Source:")
    for source, count in stats['by_source'].items():
        print(f"    {source}: {count}")

    if stats['by_plugin']:
        print(f"\n  By Plugin:")
        for plugin, count in stats['by_plugin'].items():
            print(f"    {plugin}: {count}")

    print(f"\n  Confidence Distribution:")
    for level, count in stats['confidence_distribution'].items():
        print(f"    {level}: {count}")

    print(f"\n📋 Merged Results ({len(result['merged_results'])} entities):")
    for entity in result['merged_results']:
        print(f"  - {entity['entity_type']}: {entity['text']}")
        print(f"    Source: {entity['source']}, Confidence: {entity['confidence']:.2f}")


def demo_extension_info():
    """Demonstrate extension information retrieval."""
    print("\n" + "="*60)
    print("DEMO 6: Extension Information")
    print("="*60)

    from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

    detector = EnhancedEnsembleDetector(enable_plugins=True)

    info = detector.get_extension_info()

    print(f"Extensions Status:")
    print(f"  Plugins Enabled: {info['plugins_enabled']}")
    print(f"  LLM Enabled: {info['llm_enabled']}")

    print(f"\n  Registered Plugins ({len(info['plugins'])} total):")
    for plugin in info['plugins']:
        status = "✓ ENABLED" if plugin['enabled'] else "✗ DISABLED"
        print(f"    {status} {plugin['name']} v{plugin['version']}")
        print(f"       Priority: {plugin['priority']}, Types: {len(plugin['entity_types'])}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🚀 RedactionTool Extensibility Framework Demo")
    print("="*60)

    try:
        demo_plugin_discovery()
        demo_crypto_detection()
        demo_medical_codes_detection()
        demo_custom_regex_detection()
        demo_enhanced_provenance()
        demo_extension_info()

        print("\n" + "="*60)
        print("✅ All Demos Completed Successfully!")
        print("="*60)
        print("\nNext Steps:")
        print("  1. Add more custom patterns to custom_patterns.json")
        print("  2. Create your own plugins in plugins/detectors/")
        print("  3. Implement language packs in plugins/languages/")
        print("  4. Configure LLM providers for context-aware detection")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
