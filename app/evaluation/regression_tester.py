"""
Regression Tester for PII Detection.

Tracks performance over time and detects regressions:
- Store baseline metrics
- Compare current vs baseline
- Alert on performance degradation
- Track metrics history
- Version comparison
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import os
from pathlib import Path


class RegressionTester:
    """
    Track PII detection performance and detect regressions.

    Features:
    - Store baseline metrics
    - Compare versions
    - Track metrics over time
    - Alert on degradation
    - Generate comparison reports
    """

    def __init__(self, storage_dir: str = "evaluation_results"):
        """
        Initialize regression tester.

        Args:
            storage_dir: Directory to store metrics history
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.baseline_file = os.path.join(storage_dir, "baseline.json")
        self.history_file = os.path.join(storage_dir, "metrics_history.json")

        self.baseline = self._load_baseline()
        self.history = self._load_history()

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline metrics."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None

    def _save_baseline(self):
        """Save baseline metrics."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baseline, f, indent=2)

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load metrics history."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save metrics history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def set_baseline(
        self,
        metrics: Dict[str, Any],
        version: str = "1.0.0",
        description: str = ""
    ):
        """
        Set baseline metrics for comparison.

        Args:
            metrics: Metrics dictionary (from MetricsCalculator)
            version: Version identifier
            description: Description of baseline
        """
        self.baseline = {
            'version': version,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': metrics['overall_metrics'],
            'per_entity_metrics': metrics['per_entity_metrics']
        }

        self._save_baseline()
        print(f"✅ Baseline set: {version} - {description}")

    def record_metrics(
        self,
        metrics: Dict[str, Any],
        version: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record metrics for a specific version.

        Args:
            metrics: Metrics dictionary
            version: Version identifier
            description: Description of this run
            metadata: Additional metadata
        """
        record = {
            'version': version,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': metrics['overall_metrics'],
            'per_entity_metrics': metrics['per_entity_metrics'],
            'metadata': metadata or {}
        }

        self.history.append(record)
        self._save_history()

        print(f"✅ Metrics recorded: {version}")

    def compare_to_baseline(
        self,
        current_metrics: Dict[str, Any],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare current metrics to baseline.

        Args:
            current_metrics: Current metrics to compare
            threshold: Degradation threshold (0.05 = 5%)

        Returns:
            Comparison report
        """
        if not self.baseline:
            return {
                'has_baseline': False,
                'message': 'No baseline set. Use set_baseline() first.'
            }

        baseline_overall = self.baseline['overall_metrics']
        current_overall = current_metrics['overall_metrics']

        # Compare overall metrics
        comparisons = {}
        regressions = []
        improvements = []

        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            baseline_val = baseline_overall.get(metric, 0)
            current_val = current_overall.get(metric, 0)

            diff = current_val - baseline_val
            percent_change = (diff / baseline_val * 100) if baseline_val > 0 else 0

            comparisons[metric] = {
                'baseline': baseline_val,
                'current': current_val,
                'difference': diff,
                'percent_change': percent_change,
                'status': self._get_status(diff, threshold)
            }

            if diff < -threshold:
                regressions.append({
                    'metric': metric,
                    'baseline': baseline_val,
                    'current': current_val,
                    'degradation': abs(diff),
                    'percent': percent_change
                })
            elif diff > threshold:
                improvements.append({
                    'metric': metric,
                    'baseline': baseline_val,
                    'current': current_val,
                    'improvement': diff,
                    'percent': percent_change
                })

        # Compare per-entity metrics
        entity_comparisons = {}
        entity_regressions = []

        baseline_entities = self.baseline.get('per_entity_metrics', {})
        current_entities = current_metrics.get('per_entity_metrics', {})

        for entity_type in set(baseline_entities.keys()) | set(current_entities.keys()):
            baseline_f1 = baseline_entities.get(entity_type, {}).get('f1', 0)
            current_f1 = current_entities.get(entity_type, {}).get('f1', 0)

            diff = current_f1 - baseline_f1
            percent_change = (diff / baseline_f1 * 100) if baseline_f1 > 0 else 0

            entity_comparisons[entity_type] = {
                'baseline_f1': baseline_f1,
                'current_f1': current_f1,
                'difference': diff,
                'percent_change': percent_change,
                'status': self._get_status(diff, threshold)
            }

            if diff < -threshold:
                entity_regressions.append({
                    'entity_type': entity_type,
                    'baseline_f1': baseline_f1,
                    'current_f1': current_f1,
                    'degradation': abs(diff),
                    'percent': percent_change
                })

        # Overall status
        has_regression = len(regressions) > 0 or len(entity_regressions) > 0

        return {
            'has_baseline': True,
            'baseline_version': self.baseline['version'],
            'baseline_date': self.baseline['timestamp'],
            'comparison_date': datetime.now().isoformat(),
            'threshold': threshold,
            'overall_comparisons': comparisons,
            'entity_comparisons': entity_comparisons,
            'regressions': regressions,
            'entity_regressions': entity_regressions,
            'improvements': improvements,
            'has_regression': has_regression,
            'status': 'REGRESSION' if has_regression else 'PASS',
            'summary': self._generate_comparison_summary(
                regressions,
                entity_regressions,
                improvements
            )
        }

    def _get_status(self, diff: float, threshold: float) -> str:
        """Get status string based on difference."""
        if diff < -threshold:
            return '⚠️ REGRESSION'
        elif diff > threshold:
            return '✅ IMPROVED'
        else:
            return '➡️ STABLE'

    def _generate_comparison_summary(
        self,
        regressions: List[Dict[str, Any]],
        entity_regressions: List[Dict[str, Any]],
        improvements: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable comparison summary."""
        lines = []

        if regressions:
            lines.append(f"⚠️ {len(regressions)} overall metric regression(s) detected:")
            for reg in regressions:
                lines.append(
                    f"  - {reg['metric']}: {reg['baseline']:.4f} → {reg['current']:.4f} "
                    f"({reg['percent']:+.1f}%)"
                )

        if entity_regressions:
            lines.append(f"⚠️ {len(entity_regressions)} entity-level regression(s):")
            for reg in entity_regressions[:5]:  # Show top 5
                lines.append(
                    f"  - {reg['entity_type']}: {reg['baseline_f1']:.4f} → {reg['current_f1']:.4f} "
                    f"({reg['percent']:+.1f}%)"
                )

        if improvements:
            lines.append(f"✅ {len(improvements)} improvement(s):")
            for imp in improvements:
                lines.append(
                    f"  - {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} "
                    f"({imp['percent']:+.1f}%)"
                )

        if not regressions and not entity_regressions and not improvements:
            lines.append("➡️ Performance stable (within threshold)")

        return "\n".join(lines)

    def get_metrics_trend(
        self,
        metric: str = 'f1',
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trend for a specific metric over time.

        Args:
            metric: Metric name ('precision', 'recall', 'f1', 'accuracy')
            limit: Number of recent records to return

        Returns:
            List of metric values over time
        """
        trend = []

        for record in self.history[-limit:]:
            trend.append({
                'version': record['version'],
                'timestamp': record['timestamp'],
                'value': record['overall_metrics'].get(metric, 0),
                'description': record.get('description', '')
            })

        return trend

    def get_entity_trend(
        self,
        entity_type: str,
        metric: str = 'f1',
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trend for a specific entity type over time.

        Args:
            entity_type: Entity type
            metric: Metric name
            limit: Number of recent records

        Returns:
            List of entity metric values over time
        """
        trend = []

        for record in self.history[-limit:]:
            entity_metrics = record['per_entity_metrics'].get(entity_type, {})
            trend.append({
                'version': record['version'],
                'timestamp': record['timestamp'],
                'value': entity_metrics.get(metric, 0),
                'description': record.get('description', '')
            })

        return trend

    def detect_performance_drift(
        self,
        window: int = 5,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect gradual performance drift over recent runs.

        Args:
            window: Number of recent runs to analyze
            threshold: Drift threshold

        Returns:
            Drift detection report
        """
        if len(self.history) < window:
            return {
                'detected': False,
                'message': f'Insufficient history (need {window}, have {len(self.history)})'
            }

        recent = self.history[-window:]
        metrics = ['precision', 'recall', 'f1', 'accuracy']

        drifts = []

        for metric in metrics:
            values = [r['overall_metrics'].get(metric, 0) for r in recent]

            if len(values) < 2:
                continue

            # Calculate trend (simple linear)
            first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
            second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

            drift = second_half_avg - first_half_avg

            if abs(drift) > threshold:
                drifts.append({
                    'metric': metric,
                    'drift': drift,
                    'first_half_avg': first_half_avg,
                    'second_half_avg': second_half_avg,
                    'direction': 'degradation' if drift < 0 else 'improvement'
                })

        return {
            'detected': len(drifts) > 0,
            'window': window,
            'threshold': threshold,
            'drifts': drifts,
            'summary': self._generate_drift_summary(drifts)
        }

    def _generate_drift_summary(self, drifts: List[Dict[str, Any]]) -> str:
        """Generate drift summary."""
        if not drifts:
            return "No significant performance drift detected"

        lines = []
        degradations = [d for d in drifts if d['direction'] == 'degradation']
        improvements = [d for d in drifts if d['direction'] == 'improvement']

        if degradations:
            lines.append(f"⚠️ {len(degradations)} metric(s) showing degradation:")
            for d in degradations:
                lines.append(
                    f"  - {d['metric']}: {d['first_half_avg']:.4f} → {d['second_half_avg']:.4f} "
                    f"({d['drift']:+.4f})"
                )

        if improvements:
            lines.append(f"✅ {len(improvements)} metric(s) showing improvement:")
            for d in improvements:
                lines.append(
                    f"  - {d['metric']}: {d['first_half_avg']:.4f} → {d['second_half_avg']:.4f} "
                    f"({d['drift']:+.4f})"
                )

        return "\n".join(lines)

    def generate_regression_report(
        self,
        current_metrics: Dict[str, Any],
        version: str,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generate comprehensive regression test report.

        Args:
            current_metrics: Current metrics
            version: Current version
            threshold: Regression threshold

        Returns:
            Regression test report
        """
        comparison = self.compare_to_baseline(current_metrics, threshold)
        drift = self.detect_performance_drift()

        report = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'baseline_comparison': comparison,
            'drift_detection': drift,
            'metrics_history': {
                'precision': self.get_metrics_trend('precision'),
                'recall': self.get_metrics_trend('recall'),
                'f1': self.get_metrics_trend('f1')
            },
            'status': comparison.get('status', 'UNKNOWN'),
            'recommendations': self._generate_recommendations(comparison, drift)
        }

        return report

    def _generate_recommendations(
        self,
        comparison: Dict[str, Any],
        drift: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on regression analysis."""
        recommendations = []

        if comparison.get('has_regression'):
            regressions = comparison.get('regressions', [])
            if regressions:
                recommendations.append(
                    "⚠️ Significant regressions detected. Review recent code changes and "
                    "consider rolling back or investigating root cause."
                )

            entity_regressions = comparison.get('entity_regressions', [])
            if entity_regressions:
                worst = max(entity_regressions, key=lambda x: abs(x['degradation']))
                recommendations.append(
                    f"⚠️ Worst performing entity: {worst['entity_type']} "
                    f"(F1: {worst['baseline_f1']:.4f} → {worst['current_f1']:.4f}). "
                    "Focus investigation on this entity type."
                )

        if drift.get('detected'):
            degradation_drifts = [
                d for d in drift.get('drifts', [])
                if d['direction'] == 'degradation'
            ]
            if degradation_drifts:
                recommendations.append(
                    "⚠️ Gradual performance degradation detected over recent runs. "
                    "Consider reviewing data quality or model drift."
                )

        if not comparison.get('has_regression') and not drift.get('detected'):
            recommendations.append(
                "✅ No regressions detected. Performance is stable or improving."
            )

        return recommendations

    def save_report(
        self,
        report: Dict[str, Any],
        filepath: str
    ):
        """Save regression report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def print_comparison(self, comparison: Dict[str, Any]):
        """Print comparison results to console."""
        print("\n" + "=" * 70)
        print("REGRESSION TEST RESULTS")
        print("=" * 70)

        if not comparison.get('has_baseline'):
            print("\n⚠️ No baseline set. Use set_baseline() to establish baseline.")
            print("=" * 70 + "\n")
            return

        print(f"\nBaseline: {comparison['baseline_version']} ({comparison['baseline_date'][:10]})")
        print(f"Threshold: ±{comparison['threshold']*100:.1f}%")
        print(f"\nOverall Status: {comparison['status']}")

        print("\nOverall Metrics Comparison:")
        print(f"{'Metric':<15} {'Baseline':>10} {'Current':>10} {'Diff':>10} {'Change':>10} {'Status':<20}")
        print("-" * 85)

        for metric, data in comparison['overall_comparisons'].items():
            print(
                f"{metric:<15} {data['baseline']:>10.4f} {data['current']:>10.4f} "
                f"{data['difference']:>+10.4f} {data['percent_change']:>+9.1f}% {data['status']:<20}"
            )

        if comparison['regressions']:
            print(f"\n⚠️ REGRESSIONS DETECTED ({len(comparison['regressions'])}):")
            for reg in comparison['regressions']:
                print(
                    f"  - {reg['metric']}: {reg['baseline']:.4f} → {reg['current']:.4f} "
                    f"({reg['percent']:+.1f}%)"
                )

        if comparison['entity_regressions']:
            print(f"\n⚠️ ENTITY REGRESSIONS ({len(comparison['entity_regressions'])}):")
            for reg in comparison['entity_regressions'][:5]:
                print(
                    f"  - {reg['entity_type']}: {reg['baseline_f1']:.4f} → {reg['current_f1']:.4f} "
                    f"({reg['percent']:+.1f}%)"
                )

        if comparison['improvements']:
            print(f"\n✅ IMPROVEMENTS ({len(comparison['improvements'])}):")
            for imp in comparison['improvements']:
                print(
                    f"  - {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} "
                    f"({imp['percent']:+.1f}%)"
                )

        print("\n" + comparison['summary'])
        print("=" * 70 + "\n")
