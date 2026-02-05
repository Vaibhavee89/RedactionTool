"""
Evaluation & Quality Metrics Module.

Provides:
- Labeled evaluation datasets (synthetic + semi-real)
- Metrics calculation (Precision, Recall, F1)
- Error analysis reports (FP, FN, error categorization)
- Regression testing framework
- Performance tracking over time
"""

from .dataset_generator import EvaluationDatasetGenerator, load_evaluation_dataset
from .metrics_calculator import MetricsCalculator, calculate_metrics
from .error_analyzer import ErrorAnalyzer, ErrorType
from .regression_tester import RegressionTester

__all__ = [
    'EvaluationDatasetGenerator',
    'load_evaluation_dataset',
    'MetricsCalculator',
    'calculate_metrics',
    'ErrorAnalyzer',
    'ErrorType',
    'RegressionTester'
]
