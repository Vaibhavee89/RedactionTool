# Evaluation & Quality Metrics System

## 🎯 Overview

The Evaluation & Quality Metrics system provides comprehensive tools to assess and improve PII detection performance. All requested features have been fully implemented and tested.

## ✅ What's Included

### 1. Labeled Evaluation Dataset (200-500 Samples) ✅
- Synthetic PII generation (PAN, Aadhaar, Phone, Email, Address)
- Indian-specific data (names, cities, Hindi names)
- Multiple template types (basic, form, letter, mixed)
- Negative samples and edge cases
- Multilingual support (English, Hindi, Hinglish)

### 2. Metrics Calculation ✅
- Precision, Recall, F1 per entity type
- Overall accuracy
- IoU-based entity matching
- Confusion matrices
- Detailed reports

### 3. Error Analysis ✅
- False positives identification
- False negatives tracking
- Error categorization (6 types, 7 categories)
- Pattern detection
- Actionable recommendations

### 4. Regression Testing ✅
- Baseline tracking
- Version comparison
- Performance drift detection
- Historical metrics
- Degradation alerts

## 🚀 Quick Start

### Option 1: Run Demo (Fastest)

```bash
python3 demo_evaluation.py
```

Shows all features with simulated data in 30 seconds.

### Option 2: Run Tests

```bash
python3 test_evaluation.py
```

Runs comprehensive test suite (10 tests, all passing).

### Option 3: Run Full Evaluation

```bash
# Set baseline
python3 run_evaluation.py --samples 200 --version 1.0.0 --set-baseline

# Test new version
python3 run_evaluation.py --samples 200 --version 1.1.0

# Use specific detector
python3 run_evaluation.py --samples 200 --detector presidio --confidence 0.7
```

### Option 4: Python API

```python
from app.evaluation import (
    EvaluationDatasetGenerator,
    MetricsCalculator,
    ErrorAnalyzer,
    RegressionTester
)

# 1. Generate dataset
generator = EvaluationDatasetGenerator(seed=42)
dataset = generator.generate_dataset(num_samples=200)

# 2. Calculate metrics
calculator = MetricsCalculator()
for sample in dataset:
    predictions = your_detector.detect(sample['text'])
    calculator.update(predictions, sample['entities'])

calculator.print_summary()

# 3. Analyze errors
analyzer = ErrorAnalyzer()
for sample in dataset:
    predictions = your_detector.detect(sample['text'])
    analyzer.analyze_sample(predictions, sample['entities'], sample['text'])

analyzer.print_summary()

# 4. Regression testing
tester = RegressionTester()
metrics = calculator.generate_report()
tester.set_baseline(metrics, version='1.0.0')
```

## 📁 File Structure

```
RedactionTool/
├── app/evaluation/
│   ├── __init__.py                  # Module exports
│   ├── dataset_generator.py         # Dataset generation (448 lines)
│   ├── metrics_calculator.py        # Metrics calculation (416 lines)
│   ├── error_analyzer.py            # Error analysis (630 lines)
│   └── regression_tester.py         # Regression testing (650 lines)
│
├── test_evaluation.py               # Test suite (700 lines)
├── run_evaluation.py                # CLI tool (400 lines)
├── demo_evaluation.py               # Quick demo (200 lines)
│
├── EVALUATION_GUIDE.md              # Comprehensive guide (900+ lines)
├── EVALUATION_IMPLEMENTATION_SUMMARY.md  # Implementation details
└── EVALUATION_README.md             # This file
```

## 📊 Example Output

### Metrics Summary
```
Overall Metrics:
  Precision: 0.9200
  Recall:    0.8800
  F1 Score:  0.8995
  Accuracy:  0.9100

Per-Entity Metrics:
Entity Type           Precision     Recall         F1    Support
----------------------------------------------------------------------
PAN                      0.9500     0.9200     0.9348         50
AADHAAR                  0.9200     0.8800     0.8995         45
PHONE                    0.9000     0.8500     0.8745         40
```

### Error Analysis
```
Total Errors: 100

Error Type Distribution:
  false_positive             40 ( 40.0%)
  false_negative             60 ( 60.0%)

Recommendations:
  1. High false negative rate (60.0%). Consider lowering confidence thresholds...
  2. Common missed entity: 'test@example.com'. Add regex patterns...
```

### Regression Test
```
Baseline: 1.0.0 (2026-01-15)
Overall Status: PASS

Overall Metrics Comparison:
Metric          Baseline    Current       Diff    Change Status
---------------------------------------------------------------------------------
precision         0.9200     0.9250   +0.0050     +0.5% ➡️ STABLE
recall            0.8800     0.8850   +0.0050     +0.6% ➡️ STABLE
f1                0.8995     0.9045   +0.0050     +0.6% ➡️ STABLE
```

## 🧪 Test Results

**Test Suite:** `test_evaluation.py`

```
✅ Passed: 10/10
❌ Failed: 0/10

🎉 ALL TESTS PASSED!
```

### Test Cases
1. ✅ Dataset Generation
2. ✅ Dataset Save/Load
3. ✅ Metrics Calculation
4. ✅ IoU Matching
5. ✅ Error Analysis
6. ✅ Error Patterns
7. ✅ Regression Testing
8. ✅ Drift Detection
9. ✅ Comprehensive Report
10. ✅ Summary Printing

## 📚 Documentation

1. **EVALUATION_README.md** (this file)
   - Quick start guide
   - File structure
   - Example usage

2. **EVALUATION_GUIDE.md** (900+ lines)
   - Complete API reference
   - Detailed examples
   - Best practices
   - Troubleshooting

3. **EVALUATION_IMPLEMENTATION_SUMMARY.md**
   - Implementation details
   - Requirements mapping
   - Technical specifications

## 🎯 Key Features

### Dataset Generation
- ✅ Configurable sample size (200-500+)
- ✅ Indian-specific data (names, cities, PAN, Aadhaar)
- ✅ Multiple languages (English, Hindi, Hinglish)
- ✅ Template types (basic, form, letter, mixed)
- ✅ Negative samples (no PII)
- ✅ Edge cases (unusual formats)
- ✅ Save/Load functionality

### Metrics Calculation
- ✅ Precision = TP / (TP + FP)
- ✅ Recall = TP / (TP + FN)
- ✅ F1 = 2 * (P * R) / (P + R)
- ✅ Accuracy = (TP + TN) / Total
- ✅ IoU-based matching (configurable threshold)
- ✅ Per-entity and overall metrics
- ✅ Confusion matrices
- ✅ Detailed reports

### Error Analysis
- ✅ 6 error types (FP, FN, Boundary, Type Mismatch, Duplicate, Confidence)
- ✅ 7 error categories (Pattern, Context, Language, Format, Overlap, Boundary, Unknown)
- ✅ False positive/negative lists
- ✅ Error pattern detection
- ✅ Most problematic entity identification
- ✅ Actionable recommendations

### Regression Testing
- ✅ Baseline storage and comparison
- ✅ Version tracking
- ✅ Configurable threshold (default: 5%)
- ✅ Performance drift detection
- ✅ Historical metrics
- ✅ Improvement/degradation alerts
- ✅ Comprehensive reports

## 💡 Usage Examples

### Example 1: Quick Evaluation

```python
from app.evaluation import EvaluationDatasetGenerator, MetricsCalculator

# Generate dataset
generator = EvaluationDatasetGenerator()
dataset = generator.generate_dataset(num_samples=100)

# Run evaluation
calculator = MetricsCalculator()
for sample in dataset:
    predictions = detector.detect(sample['text'])
    calculator.update(predictions, sample['entities'])

# Print results
calculator.print_summary()
```

### Example 2: Error Analysis

```python
from app.evaluation import ErrorAnalyzer

analyzer = ErrorAnalyzer()

for sample in dataset:
    predictions = detector.detect(sample['text'])
    analyzer.analyze_sample(predictions, sample['entities'], sample['text'])

# Get insights
fps = analyzer.get_false_positives(limit=10)
fns = analyzer.get_false_negatives(limit=10)
recommendations = analyzer.generate_recommendations()

analyzer.print_summary()
```

### Example 3: Regression Testing

```python
from app.evaluation import RegressionTester

tester = RegressionTester()

# First time: set baseline
tester.set_baseline(metrics, version='1.0.0', description='Initial release')

# Later: compare new version
tester.record_metrics(new_metrics, version='1.1.0')
comparison = tester.compare_to_baseline(new_metrics, threshold=0.05)

if comparison['has_regression']:
    print("⚠️ Regression detected!")
else:
    print("✅ No regressions")

tester.print_comparison(comparison)
```

### Example 4: Command Line

```bash
# Basic evaluation
python3 run_evaluation.py --samples 200 --version 1.0.0

# Set baseline
python3 run_evaluation.py --samples 200 --version 1.0.0 --set-baseline

# Test with specific detector
python3 run_evaluation.py --samples 200 --detector presidio --confidence 0.7

# Load existing dataset
python3 run_evaluation.py --load-dataset eval_data.json --version 1.1.0

# Configure language distribution
python3 run_evaluation.py --samples 200 --language-dist "en:0.5,hi:0.3,hinglish:0.2"
```

## 🔍 Understanding Metrics

### Precision
**What:** Of all entities detected, how many were correct?
**When high:** System is conservative, few false positives
**When low:** System over-detects, many false positives

### Recall
**What:** Of all actual entities, how many were detected?
**When high:** System is thorough, few missed entities
**When low:** System misses entities, many false negatives

### F1 Score
**What:** Harmonic mean of precision and recall
**Goal:** Balance between precision and recall
**Target:** 0.85+ for production systems

### Accuracy
**What:** Overall correctness including true negatives
**Note:** Less useful for imbalanced datasets

## 🛠️ Troubleshooting

### Low Precision (High False Positives)
- Increase confidence threshold
- Add exclusion patterns
- Improve context detection
- Refine regex patterns

### Low Recall (High False Negatives)
- Lower confidence threshold
- Add more patterns
- Improve model coverage
- Support unusual formats

### Boundary Errors
- Review tokenization
- Adjust boundary detection
- Use word boundaries
- Improve span merging

### Performance Regression
- Review recent changes
- Check pattern conflicts
- Verify model updates
- Consider rollback

## 📈 Best Practices

1. **Dataset Generation**
   - Generate 200-500 samples minimum
   - Match production language distribution
   - Include edge cases (10-20%)
   - Include negatives (10-20%)
   - Update quarterly

2. **Metrics Calculation**
   - Use IoU threshold 0.5 for general use
   - Focus on per-entity metrics
   - Always compare to baseline
   - Target F1 > 0.85 for production

3. **Error Analysis**
   - Review top errors first
   - Look for patterns
   - Prioritize by impact
   - Track over time

4. **Regression Testing**
   - Set baseline early
   - Use 5% threshold for critical metrics
   - Monitor drift over 5-10 versions
   - Test before each release

## 🎉 Summary

The Evaluation & Quality Metrics system is **fully implemented, tested, and production-ready**.

**Delivered:**
- ✅ 5 core modules (2,170 lines)
- ✅ 10 passing tests (700 lines)
- ✅ 3 documentation files (900+ lines)
- ✅ 3 tools (demo, CLI, tests)
- ✅ All features implemented
- ✅ Production ready

**Status:** Ready for research and production use! 🚀

## 📞 Getting Help

1. **Quick Demo:** `python3 demo_evaluation.py`
2. **Run Tests:** `python3 test_evaluation.py`
3. **Read Guide:** Open `EVALUATION_GUIDE.md`
4. **Check Examples:** See code examples above
5. **Run Evaluation:** `python3 run_evaluation.py --help`

---

**System Status:** ✅ All features implemented and tested

**Test Results:** 10/10 PASSED ✅

**Production Ready:** YES 🚀
