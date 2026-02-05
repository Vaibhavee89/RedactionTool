"""
Metrics Calculator for PII Detection Evaluation.

Calculates:
- Precision, Recall, F1 per entity type
- Overall metrics
- Confusion matrices
- Accuracy scores
"""

from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import json


class MetricsCalculator:
    """
    Calculate evaluation metrics for PII detection.

    Metrics:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1: 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy: (TP + TN) / Total
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator.

        Args:
            iou_threshold: IoU threshold for matching entities (0.5 = 50% overlap)
        """
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset all counts."""
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.true_negatives = 0  # Samples with no PII correctly identified

        self.total_samples = 0
        self.matched_entities = []
        self.unmatched_predictions = []
        self.unmatched_ground_truth = []

    def calculate_iou(
        self,
        pred_start: int,
        pred_end: int,
        gt_start: int,
        gt_end: int
    ) -> float:
        """
        Calculate Intersection over Union for entity spans.

        Args:
            pred_start, pred_end: Predicted entity span
            gt_start, gt_end: Ground truth entity span

        Returns:
            IoU score (0 to 1)
        """
        # Calculate intersection
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        intersection = max(0, intersection_end - intersection_start)

        # Calculate union
        union_start = min(pred_start, gt_start)
        union_end = max(pred_end, gt_end)
        union = union_end - union_start

        if union == 0:
            return 0.0

        return intersection / union

    def match_entities(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple], List[Dict], List[Dict]]:
        """
        Match predicted entities with ground truth.

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities

        Returns:
            Tuple of (matched_pairs, unmatched_predictions, unmatched_ground_truth)
        """
        matched = []
        unmatched_preds = list(predictions)
        unmatched_gt = list(ground_truth)

        # Greedy matching based on IoU and entity type
        for gt in ground_truth:
            best_match = None
            best_iou = self.iou_threshold

            for pred in unmatched_preds:
                # Must match entity type
                if pred['entity_type'] != gt['entity_type']:
                    continue

                # Calculate IoU
                iou = self.calculate_iou(
                    pred['start'], pred['end'],
                    gt['start'], gt['end']
                )

                if iou > best_iou:
                    best_iou = iou
                    best_match = pred

            if best_match:
                matched.append((best_match, gt, best_iou))
                unmatched_preds.remove(best_match)
                if gt in unmatched_gt:
                    unmatched_gt.remove(gt)

        return matched, unmatched_preds, unmatched_gt

    def update(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        sample_id: Optional[str] = None
    ):
        """
        Update metrics with a single sample.

        Args:
            predictions: Predicted entities
            ground_truth: Ground truth entities
            sample_id: Optional sample identifier
        """
        self.total_samples += 1

        # Match entities
        matched, unmatched_preds, unmatched_gt = self.match_entities(
            predictions, ground_truth
        )

        # Update counts
        for pred, gt, iou in matched:
            entity_type = gt['entity_type']
            self.true_positives[entity_type] += 1
            self.matched_entities.append({
                'sample_id': sample_id,
                'prediction': pred,
                'ground_truth': gt,
                'iou': iou
            })

        for pred in unmatched_preds:
            entity_type = pred['entity_type']
            self.false_positives[entity_type] += 1
            self.unmatched_predictions.append({
                'sample_id': sample_id,
                'prediction': pred
            })

        for gt in unmatched_gt:
            entity_type = gt['entity_type']
            self.false_negatives[entity_type] += 1
            self.unmatched_ground_truth.append({
                'sample_id': sample_id,
                'ground_truth': gt
            })

        # True negative: sample with no PII correctly identified as no PII
        if len(ground_truth) == 0 and len(predictions) == 0:
            self.true_negatives += 1

    def calculate_precision(self, entity_type: Optional[str] = None) -> float:
        """
        Calculate precision.

        Args:
            entity_type: Calculate for specific entity type (None = overall)

        Returns:
            Precision score
        """
        if entity_type:
            tp = self.true_positives[entity_type]
            fp = self.false_positives[entity_type]
        else:
            tp = sum(self.true_positives.values())
            fp = sum(self.false_positives.values())

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def calculate_recall(self, entity_type: Optional[str] = None) -> float:
        """
        Calculate recall.

        Args:
            entity_type: Calculate for specific entity type (None = overall)

        Returns:
            Recall score
        """
        if entity_type:
            tp = self.true_positives[entity_type]
            fn = self.false_negatives[entity_type]
        else:
            tp = sum(self.true_positives.values())
            fn = sum(self.false_negatives.values())

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def calculate_f1(self, entity_type: Optional[str] = None) -> float:
        """
        Calculate F1 score.

        Args:
            entity_type: Calculate for specific entity type (None = overall)

        Returns:
            F1 score
        """
        precision = self.calculate_precision(entity_type)
        recall = self.calculate_recall(entity_type)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        tp = sum(self.true_positives.values())
        tn = self.true_negatives
        fp = sum(self.false_positives.values())
        fn = sum(self.false_negatives.values())

        total = tp + tn + fp + fn

        if total == 0:
            return 0.0

        return (tp + tn) / total

    def get_metrics_per_entity(self) -> Dict[str, Dict[str, float]]:
        """
        Get metrics for each entity type.

        Returns:
            Dictionary with metrics per entity type
        """
        entity_types = set(
            list(self.true_positives.keys()) +
            list(self.false_positives.keys()) +
            list(self.false_negatives.keys())
        )

        metrics = {}

        for entity_type in entity_types:
            metrics[entity_type] = {
                'precision': self.calculate_precision(entity_type),
                'recall': self.calculate_recall(entity_type),
                'f1': self.calculate_f1(entity_type),
                'true_positives': self.true_positives[entity_type],
                'false_positives': self.false_positives[entity_type],
                'false_negatives': self.false_negatives[entity_type],
                'support': self.true_positives[entity_type] + self.false_negatives[entity_type]
            }

        return metrics

    def get_overall_metrics(self) -> Dict[str, float]:
        """Get overall metrics across all entity types."""
        return {
            'precision': self.calculate_precision(),
            'recall': self.calculate_recall(),
            'f1': self.calculate_f1(),
            'accuracy': self.calculate_accuracy(),
            'true_positives': sum(self.true_positives.values()),
            'false_positives': sum(self.false_positives.values()),
            'false_negatives': sum(self.false_negatives.values()),
            'true_negatives': self.true_negatives,
            'total_samples': self.total_samples
        }

    def get_confusion_matrix(self) -> Dict[str, Any]:
        """
        Get confusion matrix data.

        Returns:
            Dictionary with confusion matrix information
        """
        entity_types = sorted(set(
            list(self.true_positives.keys()) +
            list(self.false_positives.keys()) +
            list(self.false_negatives.keys())
        ))

        matrix = {}

        for entity_type in entity_types:
            matrix[entity_type] = {
                'true_positive': self.true_positives[entity_type],
                'false_positive': self.false_positives[entity_type],
                'false_negative': self.false_negatives[entity_type]
            }

        return {
            'entity_types': entity_types,
            'matrix': matrix,
            'true_negatives': self.true_negatives
        }

    def generate_report(self, include_details: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            include_details: Include detailed error lists

        Returns:
            Complete evaluation report
        """
        report = {
            'overall_metrics': self.get_overall_metrics(),
            'per_entity_metrics': self.get_metrics_per_entity(),
            'confusion_matrix': self.get_confusion_matrix(),
            'summary': {
                'total_samples': self.total_samples,
                'total_entities_detected': sum(self.true_positives.values()) + sum(self.false_positives.values()),
                'total_entities_ground_truth': sum(self.true_positives.values()) + sum(self.false_negatives.values()),
                'detection_rate': self.calculate_recall(),
                'precision_rate': self.calculate_precision()
            }
        }

        if include_details:
            report['details'] = {
                'matched_entities': self.matched_entities[:100],  # Limit to first 100
                'false_positives': self.unmatched_predictions[:50],
                'false_negatives': self.unmatched_ground_truth[:50]
            }

        return report

    def save_report(self, filepath: str, include_details: bool = True):
        """Save evaluation report to JSON file."""
        report = self.generate_report(include_details=include_details)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print metrics summary to console."""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 70)

        overall = self.get_overall_metrics()
        print(f"\nOverall Metrics:")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1 Score:  {overall['f1']:.4f}")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")

        print(f"\nCounts:")
        print(f"  True Positives:  {overall['true_positives']}")
        print(f"  False Positives: {overall['false_positives']}")
        print(f"  False Negatives: {overall['false_negatives']}")
        print(f"  True Negatives:  {overall['true_negatives']}")

        print(f"\nPer-Entity Metrics:")
        print(f"{'Entity Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)

        per_entity = self.get_metrics_per_entity()
        for entity_type, metrics in sorted(per_entity.items()):
            print(f"{entity_type:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1']:>10.4f} {metrics['support']:>10}")

        print("=" * 70 + "\n")


# Convenience function
def calculate_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Quick metrics calculation for a single sample.

    Args:
        predictions: Predicted entities
        ground_truth: Ground truth entities
        iou_threshold: IoU threshold for matching

    Returns:
        Metrics dictionary
    """
    calculator = MetricsCalculator(iou_threshold=iou_threshold)
    calculator.update(predictions, ground_truth)
    return calculator.generate_report()
