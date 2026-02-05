#!/usr/bin/env python3
"""
Sample Evaluation Script for PII Detection System.

This script demonstrates the complete evaluation workflow:
1. Generate/load evaluation dataset
2. Run PII detection on all samples
3. Calculate metrics (Precision, Recall, F1)
4. Analyze errors (FP, FN, patterns)
5. Run regression tests
6. Generate comprehensive reports

Usage:
    python3 run_evaluation.py --samples 100 --version 1.0.0
"""

import os
import sys
import argparse
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.evaluation import (
    EvaluationDatasetGenerator,
    MetricsCalculator,
    ErrorAnalyzer,
    RegressionTester
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run PII detection evaluation')

    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )

    parser.add_argument(
        '--version',
        type=str,
        default='dev',
        help='Version identifier (default: dev)'
    )

    parser.add_argument(
        '--description',
        type=str,
        default='Evaluation run',
        help='Description of this run'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for reports (default: evaluation_results)'
    )

    parser.add_argument(
        '--detector',
        type=str,
        choices=['ensemble', 'presidio', 'regex', 'spacy'],
        default='ensemble',
        help='PII detector to use (default: ensemble)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )

    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching (default: 0.5)'
    )

    parser.add_argument(
        '--regression-threshold',
        type=float,
        default=0.05,
        help='Regression threshold (default: 0.05 = 5%%)'
    )

    parser.add_argument(
        '--set-baseline',
        action='store_true',
        help='Set this run as baseline for regression testing'
    )

    parser.add_argument(
        '--load-dataset',
        type=str,
        default=None,
        help='Load existing dataset from file'
    )

    parser.add_argument(
        '--save-dataset',
        type=str,
        default=None,
        help='Save generated dataset to file'
    )

    parser.add_argument(
        '--language-dist',
        type=str,
        default='en:0.6,hi:0.25,hinglish:0.15',
        help='Language distribution (default: en:0.6,hi:0.25,hinglish:0.15)'
    )

    return parser.parse_args()


def parse_language_distribution(lang_str: str) -> dict:
    """Parse language distribution string."""
    dist = {}
    for part in lang_str.split(','):
        lang, prob = part.split(':')
        dist[lang.strip()] = float(prob.strip())
    return dist


def load_detector(detector_type: str, confidence: float):
    """Load PII detector based on type."""
    print(f"\n📦 Loading {detector_type} detector...")

    if detector_type == 'ensemble':
        from app.services.pii.ensemble_detector import EnsembleDetector
        detector = EnsembleDetector()
        print("   ✅ Ensemble detector loaded (Presidio + Regex + SpaCy)")

    elif detector_type == 'presidio':
        from app.services.pii.presidio_detector import PresidioDetector
        detector = PresidioDetector()
        print("   ✅ Presidio detector loaded")

    elif detector_type == 'regex':
        from app.services.pii.regex_detector import RegexPIIDetector
        detector = RegexPIIDetector()
        print("   ✅ Regex detector loaded")

    elif detector_type == 'spacy':
        from app.services.pii.spacy_detector import SpacyPIIDetector
        detector = SpacyPIIDetector()
        print("   ✅ SpaCy detector loaded")

    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

    return detector


def main():
    """Main evaluation workflow."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print header
    print("\n" + "="*70)
    print("PII DETECTION SYSTEM EVALUATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Version: {args.version}")
    print(f"  Description: {args.description}")
    print(f"  Samples: {args.samples}")
    print(f"  Detector: {args.detector}")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Output directory: {args.output_dir}")

    # Step 1: Generate or load dataset
    print("\n" + "="*70)
    print("STEP 1: Dataset Preparation")
    print("="*70)

    generator = EvaluationDatasetGenerator(seed=42)

    if args.load_dataset:
        print(f"\n📂 Loading dataset from {args.load_dataset}...")
        dataset = generator.load_dataset(args.load_dataset)
        print(f"   ✅ Loaded {len(dataset)} samples")
    else:
        print(f"\n📊 Generating {args.samples} samples...")
        lang_dist = parse_language_distribution(args.language_dist)
        dataset = generator.generate_dataset(
            num_samples=args.samples,
            language_distribution=lang_dist,
            include_negatives=True,
            include_edge_cases=True
        )
        print(f"   ✅ Generated {len(dataset)} samples")

        if args.save_dataset:
            generator.save_dataset(args.save_dataset)
            print(f"   💾 Saved to {args.save_dataset}")

    # Show statistics
    stats = generator.get_statistics()
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Languages: {stats['language_distribution']}")
    print(f"   Negative samples: {stats['negative_samples']}")
    print(f"   Edge cases: {stats['edge_case_samples']}")
    print(f"   Avg entities/sample: {stats['average_entities_per_sample']:.2f}")
    print(f"   Entity types: {len(stats['entity_counts'])}")

    # Step 2: Initialize evaluation tools
    print("\n" + "="*70)
    print("STEP 2: Evaluation Tools Initialization")
    print("="*70)

    calculator = MetricsCalculator(iou_threshold=args.iou_threshold)
    analyzer = ErrorAnalyzer(iou_threshold=args.iou_threshold)
    print("   ✅ MetricsCalculator initialized")
    print("   ✅ ErrorAnalyzer initialized")

    # Step 3: Load detector
    print("\n" + "="*70)
    print("STEP 3: PII Detector Loading")
    print("="*70)

    detector = load_detector(args.detector, args.confidence)

    # Step 4: Process samples
    print("\n" + "="*70)
    print("STEP 4: Processing Samples")
    print("="*70)

    print(f"\n🔍 Processing {len(dataset)} samples...")

    for i, sample in enumerate(dataset, 1):
        if i % 25 == 0 or i == 1:
            print(f"   Progress: {i}/{len(dataset)} samples ({i*100//len(dataset)}%)")

        try:
            # Detect PII
            result = detector.detect(sample['text'], min_confidence=args.confidence)
            predictions = result.get('entities', [])

            # Update metrics
            calculator.update(
                predictions=predictions,
                ground_truth=sample['entities'],
                sample_id=sample['id']
            )

            # Analyze errors
            analyzer.analyze_sample(
                predictions=predictions,
                ground_truth=sample['entities'],
                sample_id=sample['id'],
                sample_text=sample['text']
            )

        except Exception as e:
            print(f"   ⚠️ Error processing sample {sample['id']}: {str(e)}")
            continue

    print(f"   ✅ Processed all {len(dataset)} samples")

    # Step 5: Display results
    print("\n" + "="*70)
    print("STEP 5: Results Summary")
    print("="*70)

    # Metrics summary
    calculator.print_summary()

    # Error analysis summary
    analyzer.print_summary()

    # Step 6: Save reports
    print("\n" + "="*70)
    print("STEP 6: Generating Reports")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metrics report
    metrics_file = os.path.join(args.output_dir, f'metrics_{args.version}_{timestamp}.json')
    calculator.save_report(metrics_file, include_details=True)
    print(f"   ✅ Metrics report: {metrics_file}")

    # Error analysis report
    error_file = os.path.join(args.output_dir, f'errors_{args.version}_{timestamp}.json')
    analyzer.save_report(error_file, include_details=True)
    print(f"   ✅ Error report: {error_file}")

    # Step 7: Regression testing
    print("\n" + "="*70)
    print("STEP 7: Regression Testing")
    print("="*70)

    tester = RegressionTester(storage_dir=args.output_dir)
    metrics_report = calculator.generate_report()

    if args.set_baseline:
        print(f"\n📊 Setting baseline for version {args.version}...")
        tester.set_baseline(
            metrics=metrics_report,
            version=args.version,
            description=args.description
        )
        print("   ✅ Baseline set successfully")

    elif tester.baseline:
        print(f"\n📊 Comparing to baseline ({tester.baseline['version']})...")

        # Record metrics
        tester.record_metrics(
            metrics=metrics_report,
            version=args.version,
            description=args.description
        )

        # Compare to baseline
        comparison = tester.compare_to_baseline(
            current_metrics=metrics_report,
            threshold=args.regression_threshold
        )

        # Print comparison
        tester.print_comparison(comparison)

        # Check for drift
        if len(tester.history) >= 5:
            print("\n📊 Checking for performance drift...")
            drift = tester.detect_performance_drift(window=5, threshold=0.1)

            if drift['detected']:
                print("\n⚠️ Performance drift detected!")
                print(drift['summary'])
            else:
                print("   ✅ No significant performance drift detected")

        # Save regression report
        reg_file = os.path.join(args.output_dir, f'regression_{args.version}_{timestamp}.json')
        reg_report = tester.generate_regression_report(
            current_metrics=metrics_report,
            version=args.version,
            threshold=args.regression_threshold
        )
        tester.save_report(reg_report, reg_file)
        print(f"\n   ✅ Regression report: {reg_file}")

    else:
        print("\n⚠️ No baseline set. Use --set-baseline to establish baseline.")
        print("   Recording metrics for future comparison...")
        tester.record_metrics(
            metrics=metrics_report,
            version=args.version,
            description=args.description
        )

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    overall = metrics_report['overall_metrics']
    print(f"\n📊 Final Metrics:")
    print(f"   Precision: {overall['precision']:.4f}")
    print(f"   Recall:    {overall['recall']:.4f}")
    print(f"   F1 Score:  {overall['f1']:.4f}")
    print(f"   Accuracy:  {overall['accuracy']:.4f}")

    error_summary = analyzer.get_error_summary()
    print(f"\n⚠️ Error Summary:")
    print(f"   Total errors: {error_summary['total_errors']}")
    print(f"   False positives: {error_summary['error_type_counts'].get('false_positive', 0)}")
    print(f"   False negatives: {error_summary['error_type_counts'].get('false_negative', 0)}")

    print(f"\n💡 Top Recommendations:")
    recommendations = analyzer.generate_recommendations()
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. {rec[:70]}...")

    print(f"\n📁 All reports saved to: {args.output_dir}/")
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
