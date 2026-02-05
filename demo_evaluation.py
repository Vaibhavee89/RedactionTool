#!/usr/bin/env python3
"""
Quick Demo of Evaluation & Quality Metrics System.

Demonstrates:
1. Dataset generation
2. Metrics calculation
3. Error analysis
4. Regression testing

This is a minimal demo for quick testing.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.evaluation import (
    EvaluationDatasetGenerator,
    MetricsCalculator,
    ErrorAnalyzer,
    RegressionTester
)


def demo_dataset_generation():
    """Demo: Generate evaluation dataset."""
    print("\n" + "="*70)
    print("DEMO 1: Dataset Generation")
    print("="*70)

    generator = EvaluationDatasetGenerator(seed=42)

    # Generate small dataset
    print("\n📊 Generating 20 samples...")
    dataset = generator.generate_dataset(
        num_samples=20,
        language_distribution={'en': 0.7, 'hi': 0.2, 'hinglish': 0.1}
    )

    print(f"✅ Generated {len(dataset)} samples")

    # Show sample
    sample = dataset[0]
    print(f"\n📄 Sample 1:")
    print(f"   Language: {sample['language']}")
    print(f"   Template: {sample['template_type']}")
    print(f"   Entities: {len(sample['entities'])}")
    print(f"\n   Text preview: {sample['text'][:100]}...")

    # Show statistics
    stats = generator.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Languages: {stats['language_distribution']}")
    print(f"   Avg entities: {stats['average_entities_per_sample']:.2f}")

    return dataset


def demo_metrics_calculation(dataset):
    """Demo: Calculate metrics with simulated predictions."""
    print("\n" + "="*70)
    print("DEMO 2: Metrics Calculation")
    print("="*70)

    calculator = MetricsCalculator(iou_threshold=0.5)

    print("\n🔍 Simulating PII detection...")

    # Simulate predictions (use 80% of ground truth for demo)
    for sample in dataset[:10]:
        # Simulate predictions: take 80% of actual entities
        predictions = sample['entities'][:int(len(sample['entities']) * 0.8)]

        calculator.update(
            predictions=predictions,
            ground_truth=sample['entities'],
            sample_id=sample['id']
        )

    # Get metrics
    overall = calculator.get_overall_metrics()

    print(f"\n📊 Overall Metrics:")
    print(f"   Precision: {overall['precision']:.4f}")
    print(f"   Recall:    {overall['recall']:.4f}")
    print(f"   F1 Score:  {overall['f1']:.4f}")
    print(f"   Accuracy:  {overall['accuracy']:.4f}")

    # Per-entity metrics
    per_entity = calculator.get_metrics_per_entity()
    print(f"\n📊 Per-Entity Metrics:")
    for entity_type, metrics in list(per_entity.items())[:3]:
        print(f"   {entity_type:12} - F1: {metrics['f1']:.4f}, "
              f"P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}")

    return calculator


def demo_error_analysis(dataset):
    """Demo: Analyze errors."""
    print("\n" + "="*70)
    print("DEMO 3: Error Analysis")
    print("="*70)

    analyzer = ErrorAnalyzer(iou_threshold=0.5)

    print("\n⚠️ Analyzing errors...")

    # Simulate predictions with errors
    for sample in dataset[:10]:
        # Simulate: miss 20% of entities, add 1 false positive
        predictions = sample['entities'][:int(len(sample['entities']) * 0.8)]

        # Add a false positive
        if len(predictions) > 0:
            fp = predictions[0].copy()
            fp['text'] = 'FAKE_FP'
            fp['start'] = 999
            fp['end'] = 1006
            predictions.append(fp)

        analyzer.analyze_sample(
            predictions=predictions,
            ground_truth=sample['entities'],
            sample_id=sample['id'],
            sample_text=sample['text']
        )

    # Get summary
    summary = analyzer.get_error_summary()
    print(f"\n📈 Error Summary:")
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   Error types: {summary['error_type_counts']}")

    # Get false positives
    fps = analyzer.get_false_positives(limit=3)
    if fps:
        print(f"\n⚠️ Sample False Positives:")
        for fp in fps[:2]:
            print(f"   - {fp['entity_type']}: '{fp['text']}'")

    # Get false negatives
    fns = analyzer.get_false_negatives(limit=3)
    if fns:
        print(f"\n⚠️ Sample False Negatives:")
        for fn in fns[:2]:
            print(f"   - {fn['entity_type']}: '{fn['text']}'")

    # Recommendations
    recommendations = analyzer.generate_recommendations()
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"   {i}. {rec[:70]}...")

    return analyzer


def demo_regression_testing(calculator):
    """Demo: Regression testing."""
    print("\n" + "="*70)
    print("DEMO 4: Regression Testing")
    print("="*70)

    import tempfile
    tmpdir = tempfile.mkdtemp()

    tester = RegressionTester(storage_dir=tmpdir)

    # Set baseline
    print("\n📊 Setting baseline...")
    metrics_report = calculator.generate_report()
    tester.set_baseline(
        metrics=metrics_report,
        version='1.0.0',
        description='Demo baseline'
    )

    # Simulate improved metrics
    print("\n📊 Simulating improved performance...")
    improved_metrics = {
        'overall_metrics': {
            'precision': metrics_report['overall_metrics']['precision'] + 0.02,
            'recall': metrics_report['overall_metrics']['recall'] + 0.02,
            'f1': metrics_report['overall_metrics']['f1'] + 0.02,
            'accuracy': metrics_report['overall_metrics']['accuracy'] + 0.02,
            'true_positives': metrics_report['overall_metrics']['true_positives'],
            'false_positives': metrics_report['overall_metrics']['false_positives'],
            'false_negatives': metrics_report['overall_metrics']['false_negatives'],
            'true_negatives': metrics_report['overall_metrics']['true_negatives']
        },
        'per_entity_metrics': metrics_report['per_entity_metrics']
    }

    tester.record_metrics(improved_metrics, version='1.1.0', description='With improvements')

    # Compare
    comparison = tester.compare_to_baseline(improved_metrics, threshold=0.05)

    print(f"\n📊 Comparison Results:")
    print(f"   Status: {comparison['status']}")
    print(f"   Has regression: {comparison['has_regression']}")

    if comparison['improvements']:
        print(f"\n✅ Improvements detected:")
        for imp in comparison['improvements']:
            print(f"   - {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} "
                  f"({imp['percent']:+.1f}%)")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    return tester


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("EVALUATION & QUALITY METRICS - QUICK DEMO")
    print("="*70)
    print("\nThis demo shows the evaluation system in action with simulated data.")

    # Run demos
    dataset = demo_dataset_generation()
    calculator = demo_metrics_calculation(dataset)
    analyzer = demo_error_analysis(dataset)
    tester = demo_regression_testing(calculator)

    # Final summary
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n✅ All evaluation components demonstrated successfully!")
    print("\n📚 Next Steps:")
    print("   1. Read EVALUATION_GUIDE.md for detailed documentation")
    print("   2. Run test_evaluation.py to see comprehensive tests")
    print("   3. Use run_evaluation.py for full evaluation workflow")
    print("   4. Generate your own dataset and test your detector")
    print("\n💡 Quick Start:")
    print("   python3 run_evaluation.py --samples 100 --version 1.0.0 --set-baseline")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
