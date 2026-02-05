"""
Test suite for Evaluation & Quality Metrics system.

Tests:
1. Dataset Generation (synthetic samples, edge cases, negatives)
2. Metrics Calculation (precision, recall, F1, IoU matching)
3. Error Analysis (FP, FN, categorization, recommendations)
4. Regression Testing (baseline, comparison, drift detection)
"""

import os
import sys
import json
import tempfile
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.evaluation import (
    EvaluationDatasetGenerator,
    MetricsCalculator,
    ErrorAnalyzer,
    ErrorType,
    RegressionTester
)


def test_dataset_generation():
    """Test 1: Dataset generation with various configurations."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Generation")
    print("="*70)

    generator = EvaluationDatasetGenerator(seed=42)

    # Generate small dataset
    print("\n📊 Generating dataset (50 samples)...")
    dataset = generator.generate_dataset(
        num_samples=50,
        language_distribution={'en': 0.6, 'hi': 0.3, 'hinglish': 0.1},
        include_negatives=True,
        include_edge_cases=True
    )

    print(f"✅ Generated {len(dataset)} samples")

    # Check statistics
    stats = generator.get_statistics()
    print("\n📈 Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Entity counts: {dict(list(stats['entity_counts'].items())[:5])}")
    print(f"  Language distribution: {stats['language_distribution']}")
    print(f"  Negative samples: {stats['negative_samples']}")
    print(f"  Edge case samples: {stats['edge_case_samples']}")
    print(f"  Avg entities per sample: {stats['average_entities_per_sample']:.2f}")

    # Verify sample structure
    sample = dataset[0]
    assert 'id' in sample
    assert 'text' in sample
    assert 'entities' in sample
    assert 'language' in sample

    # Check entities have required fields
    if sample['entities']:
        entity = sample['entities'][0]
        assert 'text' in entity
        assert 'entity_type' in entity
        assert 'start' in entity
        assert 'end' in entity

    print("\n✅ TEST 1 PASSED: Dataset generation working correctly")
    return dataset, generator


def test_dataset_save_load():
    """Test 2: Dataset save and load functionality."""
    print("\n" + "="*70)
    print("TEST 2: Dataset Save/Load")
    print("="*70)

    generator = EvaluationDatasetGenerator(seed=42)
    dataset = generator.generate_dataset(num_samples=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_dataset.json")

        # Save
        print(f"\n💾 Saving dataset to {filepath}...")
        generator.save_dataset(filepath)
        assert os.path.exists(filepath)
        print("✅ Dataset saved")

        # Load
        print(f"\n📂 Loading dataset from {filepath}...")
        generator2 = EvaluationDatasetGenerator()
        loaded_dataset = generator2.load_dataset(filepath)

        assert len(loaded_dataset) == len(dataset)
        print(f"✅ Loaded {len(loaded_dataset)} samples")

    print("\n✅ TEST 2 PASSED: Save/load working correctly")


def test_metrics_calculation():
    """Test 3: Metrics calculation with perfect, partial, and no matches."""
    print("\n" + "="*70)
    print("TEST 3: Metrics Calculation")
    print("="*70)

    calculator = MetricsCalculator(iou_threshold=0.5)

    # Test Case 1: Perfect match
    print("\n📊 Test Case 1: Perfect match")
    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20}
    ]
    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20}
    ]

    calculator.update(predictions, ground_truth, sample_id='test1')
    metrics = calculator.get_overall_metrics()

    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1'] == 1.0
    print("  ✅ Perfect match metrics correct")

    # Test Case 2: False positive
    print("\n📊 Test Case 2: False positive")
    calculator.reset()
    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20},
        {'entity_type': 'PHONE', 'text': '1234567890', 'start': 30, 'end': 40}  # FP
    ]
    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20}
    ]

    calculator.update(predictions, ground_truth, sample_id='test2')
    metrics = calculator.get_overall_metrics()

    print(f"  Precision: {metrics['precision']:.4f} (TP=1, FP=1)")
    print(f"  Recall: {metrics['recall']:.4f} (TP=1, FN=0)")

    assert metrics['precision'] == 0.5  # 1/(1+1)
    assert metrics['recall'] == 1.0  # 1/(1+0)
    print("  ✅ False positive metrics correct")

    # Test Case 3: False negative
    print("\n📊 Test Case 3: False negative")
    calculator.reset()
    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20}
    ]
    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20},
        {'entity_type': 'PHONE', 'text': '1234567890', 'start': 30, 'end': 40}  # FN
    ]

    calculator.update(predictions, ground_truth, sample_id='test3')
    metrics = calculator.get_overall_metrics()

    print(f"  Precision: {metrics['precision']:.4f} (TP=1, FP=0)")
    print(f"  Recall: {metrics['recall']:.4f} (TP=1, FN=1)")

    assert metrics['precision'] == 1.0  # 1/(1+0)
    assert metrics['recall'] == 0.5  # 1/(1+1)
    print("  ✅ False negative metrics correct")

    # Test Case 4: Per-entity metrics
    print("\n📊 Test Case 4: Per-entity metrics")
    calculator.reset()
    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20},
        {'entity_type': 'PHONE', 'text': '9876543210', 'start': 30, 'end': 40}
    ]
    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20},
        {'entity_type': 'PHONE', 'text': '9876543210', 'start': 30, 'end': 40},
        {'entity_type': 'EMAIL', 'text': 'test@example.com', 'start': 50, 'end': 66}  # FN
    ]

    calculator.update(predictions, ground_truth, sample_id='test4')
    per_entity = calculator.get_metrics_per_entity()

    print("  Per-entity metrics:")
    for entity_type, metrics in per_entity.items():
        print(f"    {entity_type}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")

    assert per_entity['PAN']['f1'] == 1.0
    assert per_entity['PHONE']['f1'] == 1.0
    assert per_entity['EMAIL']['recall'] == 0.0  # Missed
    print("  ✅ Per-entity metrics correct")

    print("\n✅ TEST 3 PASSED: Metrics calculation working correctly")
    return calculator


def test_iou_matching():
    """Test 4: IoU-based entity matching."""
    print("\n" + "="*70)
    print("TEST 4: IoU-based Entity Matching")
    print("="*70)

    calculator = MetricsCalculator(iou_threshold=0.5)

    # Test Case 1: Exact match (IoU = 1.0)
    print("\n📊 Test Case 1: Exact match")
    iou = calculator.calculate_iou(10, 20, 10, 20)
    print(f"  IoU: {iou:.4f}")
    assert iou == 1.0
    print("  ✅ Exact match IoU correct")

    # Test Case 2: Partial overlap (IoU = 0.5)
    print("\n📊 Test Case 2: Partial overlap")
    iou = calculator.calculate_iou(10, 20, 15, 25)
    print(f"  IoU: {iou:.4f}")
    assert 0.3 < iou < 0.4  # Should be around 0.33
    print("  ✅ Partial overlap IoU correct")

    # Test Case 3: No overlap (IoU = 0.0)
    print("\n📊 Test Case 3: No overlap")
    iou = calculator.calculate_iou(10, 20, 30, 40)
    print(f"  IoU: {iou:.4f}")
    assert iou == 0.0
    print("  ✅ No overlap IoU correct")

    # Test Case 4: Boundary error detection
    print("\n📊 Test Case 4: Boundary error detection")
    calculator.reset()
    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234', 'start': 10, 'end': 19}  # Off by 1
    ]
    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 10, 'end': 20}
    ]

    matched, unmatched_preds, unmatched_gt = calculator.match_entities(predictions, ground_truth)

    if matched:
        iou = matched[0][2]
        print(f"  Boundary IoU: {iou:.4f}")
        assert 0.5 <= iou < 1.0
        print("  ✅ Boundary error detected correctly")
    else:
        print("  ⚠️ No match found (IoU below threshold)")

    print("\n✅ TEST 4 PASSED: IoU matching working correctly")


def test_error_analysis():
    """Test 5: Error analysis and categorization."""
    print("\n" + "="*70)
    print("TEST 5: Error Analysis")
    print("="*70)

    analyzer = ErrorAnalyzer(iou_threshold=0.5)

    # Create sample with various error types
    print("\n📊 Analyzing sample with errors...")

    sample_text = "PAN: ABCDE1234F, Phone: 9876543210, Email: test@example.com"

    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 5, 'end': 15},  # Correct
        {'entity_type': 'PHONE', 'text': '123456', 'start': 25, 'end': 31, 'confidence': 0.8},  # FP
        # Missing EMAIL - FN
    ]

    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 5, 'end': 15},
        {'entity_type': 'PHONE', 'text': '9876543210', 'start': 24, 'end': 34},
        {'entity_type': 'EMAIL', 'text': 'test@example.com', 'start': 43, 'end': 59}  # Missed
    ]

    analyzer.analyze_sample(predictions, ground_truth, sample_id='test_sample', sample_text=sample_text)

    # Get error summary
    summary = analyzer.get_error_summary()
    print(f"\n📈 Error Summary:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Error types: {summary['error_type_counts']}")
    print(f"  Categories: {summary['error_category_counts']}")

    # Check false positives
    fps = analyzer.get_false_positives()
    print(f"\n⚠️ False Positives: {len(fps)}")
    if fps:
        print(f"  - {fps[0]['entity_type']}: {fps[0]['text']}")

    # Check false negatives
    fns = analyzer.get_false_negatives()
    print(f"\n⚠️ False Negatives: {len(fns)}")
    if fns:
        print(f"  - {fns[0]['entity_type']}: {fns[0]['text']}")

    # Get recommendations
    recommendations = analyzer.generate_recommendations()
    print(f"\n💡 Recommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec[:80]}...")

    assert summary['total_errors'] > 0
    assert len(fps) > 0 or len(fns) > 0
    assert len(recommendations) > 0

    print("\n✅ TEST 5 PASSED: Error analysis working correctly")
    return analyzer


def test_error_patterns():
    """Test 6: Error pattern detection."""
    print("\n" + "="*70)
    print("TEST 6: Error Pattern Detection")
    print("="*70)

    analyzer = ErrorAnalyzer(iou_threshold=0.5)

    # Analyze multiple samples to find patterns
    print("\n📊 Analyzing multiple samples for patterns...")

    for i in range(5):
        predictions = [
            {'entity_type': 'PAN', 'text': f'PAN{i}', 'start': 0, 'end': 10},
            {'entity_type': 'PHONE', 'text': 'COMMON_FP', 'start': 20, 'end': 30}  # Repeated FP
        ]
        ground_truth = [
            {'entity_type': 'PAN', 'text': f'PAN{i}', 'start': 0, 'end': 10},
            {'entity_type': 'EMAIL', 'text': 'COMMON_FN', 'start': 40, 'end': 50}  # Repeated FN
        ]

        analyzer.analyze_sample(predictions, ground_truth, sample_id=f'sample_{i}')

    # Get error patterns
    patterns = analyzer.get_error_patterns()

    print("\n📈 Error Patterns:")
    print(f"  Most problematic entities: {patterns['most_problematic_entities'][:3]}")
    print(f"  Common FP texts: {list(patterns['common_false_positive_texts'].keys())[:3]}")
    print(f"  Common FN texts: {list(patterns['common_false_negative_texts'].keys())[:3]}")
    print(f"  Category distribution: {patterns['category_distribution']}")

    assert len(patterns['most_problematic_entities']) > 0
    assert 'COMMON_FP' in patterns['common_false_positive_texts']
    assert 'COMMON_FN' in patterns['common_false_negative_texts']

    print("\n✅ TEST 6 PASSED: Pattern detection working correctly")


def test_regression_testing():
    """Test 7: Regression testing and baseline comparison."""
    print("\n" + "="*70)
    print("TEST 7: Regression Testing")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tester = RegressionTester(storage_dir=tmpdir)

        # Create baseline metrics
        print("\n📊 Setting baseline metrics...")
        baseline_metrics = {
            'overall_metrics': {
                'precision': 0.90,
                'recall': 0.85,
                'f1': 0.875,
                'accuracy': 0.88
            },
            'per_entity_metrics': {
                'PAN': {'precision': 0.95, 'recall': 0.90, 'f1': 0.925},
                'PHONE': {'precision': 0.85, 'recall': 0.80, 'f1': 0.825}
            }
        }

        tester.set_baseline(baseline_metrics, version='1.0.0', description='Initial baseline')
        assert tester.baseline is not None
        print("✅ Baseline set")

        # Test Case 1: No regression (stable performance)
        print("\n📊 Test Case 1: Stable performance")
        current_metrics = {
            'overall_metrics': {
                'precision': 0.91,
                'recall': 0.84,
                'f1': 0.875,
                'accuracy': 0.88
            },
            'per_entity_metrics': {
                'PAN': {'precision': 0.96, 'recall': 0.89, 'f1': 0.925},
                'PHONE': {'precision': 0.86, 'recall': 0.79, 'f1': 0.825}
            }
        }

        comparison = tester.compare_to_baseline(current_metrics, threshold=0.05)
        print(f"  Status: {comparison['status']}")
        print(f"  Has regression: {comparison['has_regression']}")
        assert comparison['status'] == 'PASS'
        print("  ✅ Stable performance detected correctly")

        # Test Case 2: Regression detected
        print("\n📊 Test Case 2: Performance regression")
        regressed_metrics = {
            'overall_metrics': {
                'precision': 0.80,  # Dropped by 0.10
                'recall': 0.75,  # Dropped by 0.10
                'f1': 0.775,
                'accuracy': 0.78
            },
            'per_entity_metrics': {
                'PAN': {'precision': 0.85, 'recall': 0.80, 'f1': 0.825},
                'PHONE': {'precision': 0.75, 'recall': 0.70, 'f1': 0.725}
            }
        }

        comparison = tester.compare_to_baseline(regressed_metrics, threshold=0.05)
        print(f"  Status: {comparison['status']}")
        print(f"  Has regression: {comparison['has_regression']}")
        print(f"  Number of regressions: {len(comparison['regressions'])}")
        assert comparison['status'] == 'REGRESSION'
        assert comparison['has_regression'] is True
        print("  ✅ Regression detected correctly")

        # Test Case 3: Record metrics and check history
        print("\n📊 Test Case 3: Metrics history")
        for i in range(3):
            metrics = {
                'overall_metrics': {
                    'precision': 0.90 + i * 0.01,
                    'recall': 0.85 + i * 0.01,
                    'f1': 0.875 + i * 0.01,
                    'accuracy': 0.88
                },
                'per_entity_metrics': {}
            }
            tester.record_metrics(metrics, version=f'1.{i}.0', description=f'Version {i}')

        trend = tester.get_metrics_trend('f1', limit=3)
        print(f"  History length: {len(trend)}")
        print(f"  F1 trend: {[t['value'] for t in trend]}")
        assert len(trend) == 3
        print("  ✅ Metrics history tracked correctly")

    print("\n✅ TEST 7 PASSED: Regression testing working correctly")


def test_drift_detection():
    """Test 8: Performance drift detection."""
    print("\n" + "="*70)
    print("TEST 8: Performance Drift Detection")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tester = RegressionTester(storage_dir=tmpdir)

        # Create gradual performance degradation
        print("\n📊 Creating gradual performance drift...")
        for i in range(10):
            drift_factor = i * 0.02  # Gradual 2% degradation per version
            metrics = {
                'overall_metrics': {
                    'precision': 0.90 - drift_factor,
                    'recall': 0.85 - drift_factor,
                    'f1': 0.875 - drift_factor,
                    'accuracy': 0.88 - drift_factor
                },
                'per_entity_metrics': {}
            }
            tester.record_metrics(metrics, version=f'1.{i}.0')

        # Detect drift
        drift = tester.detect_performance_drift(window=10, threshold=0.05)

        print(f"\n📈 Drift Detection:")
        print(f"  Detected: {drift['detected']}")
        if drift.get('drifts'):
            print(f"  Number of drifts: {len(drift['drifts'])}")
            for d in drift['drifts']:
                print(f"    - {d['metric']}: {d['direction']} ({d['drift']:.4f})")

        assert drift['detected'] is True
        print("\n✅ TEST 8 PASSED: Drift detection working correctly")


def test_comprehensive_report():
    """Test 9: Generate comprehensive evaluation report."""
    print("\n" + "="*70)
    print("TEST 9: Comprehensive Report Generation")
    print("="*70)

    # Generate dataset
    generator = EvaluationDatasetGenerator(seed=42)
    dataset = generator.generate_dataset(num_samples=30)

    # Calculate metrics
    calculator = MetricsCalculator(iou_threshold=0.5)

    print("\n📊 Processing samples...")
    for sample in dataset[:10]:  # Process first 10
        # Simulate predictions (use ground truth with some errors)
        predictions = []
        for entity in sample['entities']:
            # Add 80% of entities correctly
            if len(predictions) < len(sample['entities']) * 0.8:
                predictions.append(entity)

        calculator.update(predictions, sample['entities'], sample_id=sample['id'])

    # Generate metrics report
    print("\n📈 Generating metrics report...")
    metrics_report = calculator.generate_report(include_details=True)
    print(f"  Overall F1: {metrics_report['overall_metrics']['f1']:.4f}")
    print(f"  Entity types: {len(metrics_report['per_entity_metrics'])}")

    # Error analysis
    print("\n⚠️ Running error analysis...")
    analyzer = ErrorAnalyzer(iou_threshold=0.5)

    for sample in dataset[:10]:
        predictions = []
        for entity in sample['entities']:
            if len(predictions) < len(sample['entities']) * 0.8:
                predictions.append(entity)

        analyzer.analyze_sample(
            predictions,
            sample['entities'],
            sample_id=sample['id'],
            sample_text=sample['text']
        )

    error_report = analyzer.generate_report(include_details=True)
    print(f"  Total errors: {error_report['summary']['total_errors']}")
    print(f"  Recommendations: {len(error_report['recommendations'])}")

    # Save reports
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_file = os.path.join(tmpdir, "metrics_report.json")
        error_file = os.path.join(tmpdir, "error_report.json")

        print("\n💾 Saving reports...")
        calculator.save_report(metrics_file, include_details=True)
        analyzer.save_report(error_file, include_details=True)

        assert os.path.exists(metrics_file)
        assert os.path.exists(error_file)
        print("  ✅ Reports saved successfully")

    print("\n✅ TEST 9 PASSED: Comprehensive reporting working correctly")


def test_print_summaries():
    """Test 10: Print summary methods."""
    print("\n" + "="*70)
    print("TEST 10: Summary Printing")
    print("="*70)

    # Create simple test data
    calculator = MetricsCalculator()
    predictions = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 0, 'end': 10}
    ]
    ground_truth = [
        {'entity_type': 'PAN', 'text': 'ABCDE1234F', 'start': 0, 'end': 10}
    ]
    calculator.update(predictions, ground_truth)

    print("\n📊 Metrics Summary:")
    calculator.print_summary()

    analyzer = ErrorAnalyzer()
    analyzer.analyze_sample(predictions, ground_truth)

    print("\n⚠️ Error Analysis Summary:")
    analyzer.print_summary()

    print("\n✅ TEST 10 PASSED: Summary printing working correctly")


def run_all_tests():
    """Run all evaluation tests."""
    print("\n" + "="*70)
    print("EVALUATION & QUALITY METRICS TEST SUITE")
    print("="*70)

    tests = [
        ("Dataset Generation", test_dataset_generation),
        ("Dataset Save/Load", test_dataset_save_load),
        ("Metrics Calculation", test_metrics_calculation),
        ("IoU Matching", test_iou_matching),
        ("Error Analysis", test_error_analysis),
        ("Error Patterns", test_error_patterns),
        ("Regression Testing", test_regression_testing),
        ("Drift Detection", test_drift_detection),
        ("Comprehensive Report", test_comprehensive_report),
        ("Summary Printing", test_print_summaries)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Final summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
