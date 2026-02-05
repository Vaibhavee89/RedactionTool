"""
Error Analyzer for PII Detection Evaluation.

Analyzes and categorizes detection errors:
- False Positives (incorrect detections)
- False Negatives (missed entities)
- Boundary errors (partial matches)
- Type mismatches (wrong entity type)
- Error patterns and recommendations
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import json


class ErrorType(Enum):
    """Types of detection errors."""
    FALSE_POSITIVE = "false_positive"  # Detected but not PII
    FALSE_NEGATIVE = "false_negative"  # Missed PII
    BOUNDARY_ERROR = "boundary_error"  # Partial overlap, wrong boundaries
    TYPE_MISMATCH = "type_mismatch"  # Detected but wrong type
    DUPLICATE_DETECTION = "duplicate_detection"  # Same entity detected multiple times
    CONFIDENCE_ERROR = "confidence_error"  # Low confidence on true PII


class ErrorCategory(Enum):
    """High-level error categories."""
    PATTERN_FAILURE = "pattern_failure"  # Regex/pattern matching failed
    CONTEXT_FAILURE = "context_failure"  # Context-based detection failed
    LANGUAGE_FAILURE = "language_failure"  # Language-specific issues
    FORMAT_FAILURE = "format_failure"  # Unusual formatting
    OVERLAP_FAILURE = "overlap_failure"  # Overlapping entities
    BOUNDARY_FAILURE = "boundary_failure"  # Incorrect boundaries
    UNKNOWN = "unknown"


class ErrorAnalyzer:
    """
    Analyze detection errors and provide insights.

    Features:
    - Categorize errors by type
    - Identify error patterns
    - Generate detailed reports
    - Provide improvement recommendations
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize error analyzer.

        Args:
            iou_threshold: IoU threshold for boundary error detection
        """
        self.iou_threshold = iou_threshold
        self.errors = []
        self.error_counts = defaultdict(lambda: defaultdict(int))
        self.entity_error_counts = defaultdict(lambda: defaultdict(int))

    def calculate_iou(
        self,
        pred_start: int,
        pred_end: int,
        gt_start: int,
        gt_end: int
    ) -> float:
        """Calculate Intersection over Union for entity spans."""
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        intersection = max(0, intersection_end - intersection_start)

        union_start = min(pred_start, gt_start)
        union_end = max(pred_end, gt_end)
        union = union_end - union_start

        if union == 0:
            return 0.0

        return intersection / union

    def categorize_error(
        self,
        error_type: ErrorType,
        prediction: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> ErrorCategory:
        """
        Categorize an error into a high-level category.

        Args:
            error_type: Type of error
            prediction: Predicted entity (if any)
            ground_truth: Ground truth entity (if any)
            context: Surrounding context text

        Returns:
            ErrorCategory
        """
        if error_type == ErrorType.BOUNDARY_ERROR:
            return ErrorCategory.BOUNDARY_FAILURE

        if error_type == ErrorType.TYPE_MISMATCH:
            # Check if it's a language issue
            if prediction and ground_truth:
                pred_type = prediction.get('entity_type', '')
                gt_type = ground_truth.get('entity_type', '')

                if 'HINDI' in pred_type or 'HINDI' in gt_type:
                    return ErrorCategory.LANGUAGE_FAILURE

            return ErrorCategory.CONTEXT_FAILURE

        if error_type == ErrorType.FALSE_POSITIVE:
            # Check if it's a pattern matching issue
            if prediction:
                text = prediction.get('text', '')
                # Check for unusual formats
                if any(char in text for char in ['-', '_', '/', '\\', '|']):
                    return ErrorCategory.FORMAT_FAILURE

                # Check for very short detections
                if len(text) < 3:
                    return ErrorCategory.PATTERN_FAILURE

            return ErrorCategory.PATTERN_FAILURE

        if error_type == ErrorType.FALSE_NEGATIVE:
            # Check if it's a format issue
            if ground_truth:
                text = ground_truth.get('text', '')
                if any(char in text for char in ['-', '_', '/', '\\', '|']):
                    return ErrorCategory.FORMAT_FAILURE

            return ErrorCategory.PATTERN_FAILURE

        if error_type == ErrorType.DUPLICATE_DETECTION:
            return ErrorCategory.OVERLAP_FAILURE

        return ErrorCategory.UNKNOWN

    def analyze_sample(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        sample_id: Optional[str] = None,
        sample_text: Optional[str] = None
    ):
        """
        Analyze errors in a single sample.

        Args:
            predictions: Predicted entities
            ground_truth: Ground truth entities
            sample_id: Sample identifier
            sample_text: Original text (for context)
        """
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()

        # Find matches and boundary errors
        for i, gt in enumerate(ground_truth):
            best_match = None
            best_iou = 0
            best_pred_idx = -1

            for j, pred in enumerate(predictions):
                if j in matched_pred:
                    continue

                iou = self.calculate_iou(
                    pred['start'], pred['end'],
                    gt['start'], gt['end']
                )

                if iou > best_iou:
                    best_iou = iou
                    best_match = pred
                    best_pred_idx = j

            if best_match:
                if gt['entity_type'] == best_match['entity_type']:
                    # Check if boundary error (IoU between threshold and 1.0)
                    if best_iou < 1.0 and best_iou >= self.iou_threshold:
                        error = {
                            'error_type': ErrorType.BOUNDARY_ERROR,
                            'sample_id': sample_id,
                            'prediction': best_match,
                            'ground_truth': gt,
                            'iou': best_iou,
                            'context': self._extract_context(sample_text, gt['start'], gt['end'])
                        }
                        error['category'] = self.categorize_error(
                            ErrorType.BOUNDARY_ERROR,
                            best_match,
                            gt,
                            error['context']
                        )
                        self.errors.append(error)
                        self.error_counts[ErrorType.BOUNDARY_ERROR][gt['entity_type']] += 1
                        self.entity_error_counts[gt['entity_type']][ErrorType.BOUNDARY_ERROR] += 1

                    matched_gt.add(i)
                    matched_pred.add(best_pred_idx)
                else:
                    # Type mismatch
                    error = {
                        'error_type': ErrorType.TYPE_MISMATCH,
                        'sample_id': sample_id,
                        'prediction': best_match,
                        'ground_truth': gt,
                        'predicted_type': best_match['entity_type'],
                        'actual_type': gt['entity_type'],
                        'iou': best_iou,
                        'context': self._extract_context(sample_text, gt['start'], gt['end'])
                    }
                    error['category'] = self.categorize_error(
                        ErrorType.TYPE_MISMATCH,
                        best_match,
                        gt,
                        error['context']
                    )
                    self.errors.append(error)
                    self.error_counts[ErrorType.TYPE_MISMATCH][gt['entity_type']] += 1
                    self.entity_error_counts[gt['entity_type']][ErrorType.TYPE_MISMATCH] += 1

                    matched_gt.add(i)
                    matched_pred.add(best_pred_idx)

        # Find false negatives (missed entities)
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt:
                error = {
                    'error_type': ErrorType.FALSE_NEGATIVE,
                    'sample_id': sample_id,
                    'ground_truth': gt,
                    'entity_type': gt['entity_type'],
                    'text': gt['text'],
                    'context': self._extract_context(sample_text, gt['start'], gt['end'])
                }
                error['category'] = self.categorize_error(
                    ErrorType.FALSE_NEGATIVE,
                    None,
                    gt,
                    error['context']
                )
                self.errors.append(error)
                self.error_counts[ErrorType.FALSE_NEGATIVE][gt['entity_type']] += 1
                self.entity_error_counts[gt['entity_type']][ErrorType.FALSE_NEGATIVE] += 1

        # Find false positives (incorrect detections)
        for j, pred in enumerate(predictions):
            if j not in matched_pred:
                error = {
                    'error_type': ErrorType.FALSE_POSITIVE,
                    'sample_id': sample_id,
                    'prediction': pred,
                    'entity_type': pred['entity_type'],
                    'text': pred['text'],
                    'confidence': pred.get('confidence', 0.0),
                    'context': self._extract_context(sample_text, pred['start'], pred['end'])
                }
                error['category'] = self.categorize_error(
                    ErrorType.FALSE_POSITIVE,
                    pred,
                    None,
                    error['context']
                )
                self.errors.append(error)
                self.error_counts[ErrorType.FALSE_POSITIVE][pred['entity_type']] += 1
                self.entity_error_counts[pred['entity_type']][ErrorType.FALSE_POSITIVE] += 1

        # Check for duplicate detections
        self._check_duplicates(predictions, sample_id, sample_text)

    def _check_duplicates(
        self,
        predictions: List[Dict[str, Any]],
        sample_id: Optional[str],
        sample_text: Optional[str]
    ):
        """Check for duplicate detections of the same entity."""
        seen = {}

        for pred in predictions:
            key = (pred['entity_type'], pred['text'], pred['start'], pred['end'])

            if key in seen:
                error = {
                    'error_type': ErrorType.DUPLICATE_DETECTION,
                    'sample_id': sample_id,
                    'prediction': pred,
                    'original': seen[key],
                    'entity_type': pred['entity_type'],
                    'context': self._extract_context(sample_text, pred['start'], pred['end'])
                }
                error['category'] = self.categorize_error(
                    ErrorType.DUPLICATE_DETECTION,
                    pred,
                    None,
                    error['context']
                )
                self.errors.append(error)
                self.error_counts[ErrorType.DUPLICATE_DETECTION][pred['entity_type']] += 1
            else:
                seen[key] = pred

    def _extract_context(
        self,
        text: Optional[str],
        start: int,
        end: int,
        window: int = 50
    ) -> str:
        """Extract surrounding context for an entity."""
        if not text:
            return ""

        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        return text[context_start:context_end]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        total_errors = len(self.errors)

        error_type_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for error in self.errors:
            error_type_counts[error['error_type'].value] += 1
            category_counts[error['category'].value] += 1

        # Convert entity error counts with enum keys to string keys
        errors_by_entity = {}
        for entity_type, errors in self.entity_error_counts.items():
            errors_by_entity[entity_type] = {
                error_type.value if isinstance(error_type, Enum) else error_type: count
                for error_type, count in errors.items()
            }

        return {
            'total_errors': total_errors,
            'error_type_counts': dict(error_type_counts),
            'error_category_counts': dict(category_counts),
            'errors_by_entity': errors_by_entity
        }

    def get_false_positives(
        self,
        entity_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get false positive errors.

        Args:
            entity_type: Filter by entity type
            limit: Maximum number to return

        Returns:
            List of false positive errors
        """
        fps = [
            error for error in self.errors
            if error['error_type'] == ErrorType.FALSE_POSITIVE
        ]

        if entity_type:
            fps = [fp for fp in fps if fp['entity_type'] == entity_type]

        return fps[:limit]

    def get_false_negatives(
        self,
        entity_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get false negative errors.

        Args:
            entity_type: Filter by entity type
            limit: Maximum number to return

        Returns:
            List of false negative errors
        """
        fns = [
            error for error in self.errors
            if error['error_type'] == ErrorType.FALSE_NEGATIVE
        ]

        if entity_type:
            fns = [fn for fn in fns if fn['entity_type'] == entity_type]

        return fns[:limit]

    def get_error_patterns(self) -> Dict[str, Any]:
        """
        Identify common error patterns.

        Returns:
            Dictionary with error pattern analysis
        """
        patterns = {
            'most_problematic_entities': [],
            'common_false_positive_texts': defaultdict(int),
            'common_false_negative_texts': defaultdict(int),
            'category_distribution': defaultdict(int),
            'boundary_error_entities': defaultdict(int)
        }

        # Count errors by entity type
        entity_total_errors = defaultdict(int)
        for entity_type, errors in self.entity_error_counts.items():
            entity_total_errors[entity_type] = sum(errors.values())

        # Sort by error count
        sorted_entities = sorted(
            entity_total_errors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        patterns['most_problematic_entities'] = sorted_entities[:10]

        # Analyze error texts and categories
        for error in self.errors:
            patterns['category_distribution'][error['category'].value] += 1

            if error['error_type'] == ErrorType.FALSE_POSITIVE:
                text = error.get('text', '')
                patterns['common_false_positive_texts'][text] += 1

            elif error['error_type'] == ErrorType.FALSE_NEGATIVE:
                text = error['ground_truth'].get('text', '')
                patterns['common_false_negative_texts'][text] += 1

            elif error['error_type'] == ErrorType.BOUNDARY_ERROR:
                entity_type = error['ground_truth']['entity_type']
                patterns['boundary_error_entities'][entity_type] += 1

        # Convert to regular dicts and limit
        patterns['common_false_positive_texts'] = dict(
            sorted(
                patterns['common_false_positive_texts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        )

        patterns['common_false_negative_texts'] = dict(
            sorted(
                patterns['common_false_negative_texts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        )

        patterns['category_distribution'] = dict(patterns['category_distribution'])
        patterns['boundary_error_entities'] = dict(patterns['boundary_error_entities'])

        return patterns

    def generate_recommendations(self) -> List[str]:
        """
        Generate improvement recommendations based on error analysis.

        Returns:
            List of recommendations
        """
        recommendations = []
        patterns = self.get_error_patterns()
        summary = self.get_error_summary()

        # Analyze error types
        error_types = summary['error_type_counts']

        # False positive recommendations
        fp_count = error_types.get('false_positive', 0)
        if fp_count > 0:
            fp_rate = fp_count / max(summary['total_errors'], 1)
            if fp_rate > 0.3:
                recommendations.append(
                    f"High false positive rate ({fp_rate:.1%}). "
                    "Consider increasing confidence thresholds or refining regex patterns."
                )

            # Check common FP texts
            common_fps = patterns['common_false_positive_texts']
            if common_fps:
                top_fp = list(common_fps.keys())[0]
                recommendations.append(
                    f"Common false positive: '{top_fp}'. "
                    "Add to exclusion list or improve context detection."
                )

        # False negative recommendations
        fn_count = error_types.get('false_negative', 0)
        if fn_count > 0:
            fn_rate = fn_count / max(summary['total_errors'], 1)
            if fn_rate > 0.3:
                recommendations.append(
                    f"High false negative rate ({fn_rate:.1%}). "
                    "Consider lowering confidence thresholds or adding more patterns."
                )

            # Check common FN texts
            common_fns = patterns['common_false_negative_texts']
            if common_fns:
                top_fn = list(common_fns.keys())[0]
                recommendations.append(
                    f"Common missed entity: '{top_fn}'. "
                    "Add regex patterns or improve model training."
                )

        # Boundary error recommendations
        be_count = error_types.get('boundary_error', 0)
        if be_count > 0:
            be_rate = be_count / max(summary['total_errors'], 1)
            if be_rate > 0.2:
                recommendations.append(
                    f"Significant boundary errors ({be_rate:.1%}). "
                    "Review entity boundary detection logic and tokenization."
                )

        # Type mismatch recommendations
        tm_count = error_types.get('type_mismatch', 0)
        if tm_count > 0:
            recommendations.append(
                f"Type mismatches detected ({tm_count}). "
                "Review entity classification logic and context features."
            )

        # Category-specific recommendations
        categories = patterns['category_distribution']

        if categories.get('language_failure', 0) > 0:
            recommendations.append(
                "Language detection failures detected. "
                "Ensure proper language-specific models and preprocessing."
            )

        if categories.get('format_failure', 0) > 0:
            recommendations.append(
                "Format-related failures detected. "
                "Add support for unusual formatting and delimiters."
            )

        if categories.get('pattern_failure', 0) > 0:
            recommendations.append(
                "Pattern matching failures detected. "
                "Review and expand regex patterns for better coverage."
            )

        # Entity-specific recommendations
        problematic = patterns['most_problematic_entities']
        if problematic:
            entity_type, count = problematic[0]
            recommendations.append(
                f"Most problematic entity: {entity_type} ({count} errors). "
                "Focus improvement efforts on this entity type."
            )

        if not recommendations:
            recommendations.append("No significant issues detected. System performing well!")

        return recommendations

    def generate_report(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive error analysis report.

        Args:
            include_details: Include detailed error lists

        Returns:
            Error analysis report
        """
        report = {
            'summary': self.get_error_summary(),
            'patterns': self.get_error_patterns(),
            'recommendations': self.generate_recommendations()
        }

        if include_details:
            report['details'] = {
                'false_positives': self.get_false_positives(limit=30),
                'false_negatives': self.get_false_negatives(limit=30),
                'all_errors': self.errors[:100]  # Limit to first 100
            }

        return report

    def save_report(self, filepath: str, include_details: bool = True):
        """Save error analysis report to JSON file."""
        report = self.generate_report(include_details=include_details)

        # Convert enums to strings for JSON serialization
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        report = convert_enums(report)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print error analysis summary to console."""
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 70)

        summary = self.get_error_summary()
        patterns = self.get_error_patterns()

        print(f"\nTotal Errors: {summary['total_errors']}")

        print(f"\nError Type Distribution:")
        for error_type, count in summary['error_type_counts'].items():
            rate = count / max(summary['total_errors'], 1) * 100
            print(f"  {error_type:<25} {count:>5} ({rate:>5.1f}%)")

        print(f"\nError Category Distribution:")
        for category, count in patterns['category_distribution'].items():
            rate = count / max(summary['total_errors'], 1) * 100
            print(f"  {category:<25} {count:>5} ({rate:>5.1f}%)")

        print(f"\nMost Problematic Entities:")
        for entity_type, count in patterns['most_problematic_entities'][:5]:
            print(f"  {entity_type:<25} {count:>5} errors")

        print(f"\nRecommendations:")
        for i, rec in enumerate(self.generate_recommendations(), 1):
            print(f"  {i}. {rec}")

        print("=" * 70 + "\n")
