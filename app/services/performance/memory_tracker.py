"""
Memory Usage Tracker - Real-time memory monitoring and profiling.

Features:
- Real-time memory usage tracking
- Peak memory detection
- Memory leak detection
- Per-operation memory profiling
- Automatic garbage collection
"""

import os
import gc
import psutil
import tracemalloc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
import logging


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float
    available_mb: float


class MemoryTracker:
    """
    Real-time memory usage tracker with profiling capabilities.

    Tracks memory usage throughout processing and detects potential issues.
    """

    def __init__(self, enable_detailed_tracking: bool = False):
        """
        Initialize MemoryTracker.

        Args:
            enable_detailed_tracking: Enable tracemalloc for detailed tracking
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.logger = logging.getLogger(__name__)

        # Get process handle
        self.process = psutil.Process(os.getpid())

        # Snapshots
        self._snapshots: List[MemorySnapshot] = []
        self._operation_snapshots: Dict[str, List[MemorySnapshot]] = {}

        # Peak memory
        self._peak_memory_mb = 0.0

        # Enable tracemalloc if requested
        if self.enable_detailed_tracking:
            tracemalloc.start()
            self.logger.info("Detailed memory tracking enabled")

    def take_snapshot(self, label: Optional[str] = None) -> MemorySnapshot:
        """
        Take a snapshot of current memory usage.

        Args:
            label: Optional label for the snapshot

        Returns:
            MemorySnapshot
        """
        # Get memory info
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()

        # Create snapshot
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=self.process.memory_percent(),
            available_mb=system_mem.available / (1024 * 1024)
        )

        # Track peak
        if snapshot.rss_mb > self._peak_memory_mb:
            self._peak_memory_mb = snapshot.rss_mb

        # Store snapshot
        self._snapshots.append(snapshot)

        # Store in operation snapshots if label provided
        if label:
            if label not in self._operation_snapshots:
                self._operation_snapshots[label] = []
            self._operation_snapshots[label].append(snapshot)

        return snapshot

    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager to track memory usage of an operation.

        Args:
            operation_name: Name of the operation

        Example:
            with tracker.track_operation("file_processing"):
                # ... do work ...
                pass
        """
        # Take snapshot before
        self.take_snapshot(f"{operation_name}_start")

        try:
            yield
        finally:
            # Take snapshot after
            self.take_snapshot(f"{operation_name}_end")

    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.

        Returns:
            Dictionary with current memory stats
        """
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()

        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "percent": self.process.memory_percent(),
            "system_total_mb": system_mem.total / (1024 * 1024),
            "system_available_mb": system_mem.available / (1024 * 1024),
            "system_used_mb": system_mem.used / (1024 * 1024),
            "system_percent": system_mem.percent
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory stats
        """
        if not self._snapshots:
            return self.get_current_usage()

        # Calculate statistics
        rss_values = [s.rss_mb for s in self._snapshots]

        stats = {
            "current": self.get_current_usage(),
            "peak_mb": self._peak_memory_mb,
            "min_mb": min(rss_values),
            "max_mb": max(rss_values),
            "avg_mb": sum(rss_values) / len(rss_values),
            "snapshots_taken": len(self._snapshots),
            "memory_growth_mb": rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0
        }

        # Add detailed stats if available
        if self.enable_detailed_tracking:
            stats["detailed"] = self._get_detailed_stats()

        return stats

    def _get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed memory statistics using tracemalloc.

        Returns:
            Dictionary with detailed stats
        """
        if not tracemalloc.is_tracing():
            return {}

        current, peak = tracemalloc.get_traced_memory()

        # Get top memory allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        top_allocations = []
        for stat in top_stats[:10]:
            top_allocations.append({
                "file": stat.traceback.format()[0] if stat.traceback.format() else "unknown",
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count
            })

        return {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "top_allocations": top_allocations
        }

    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get memory statistics for a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary with operation memory stats or None
        """
        start_key = f"{operation_name}_start"
        end_key = f"{operation_name}_end"

        if start_key not in self._operation_snapshots or end_key not in self._operation_snapshots:
            return None

        start_snapshot = self._operation_snapshots[start_key][-1]
        end_snapshot = self._operation_snapshots[end_key][-1]

        return {
            "operation": operation_name,
            "start_mb": start_snapshot.rss_mb,
            "end_mb": end_snapshot.rss_mb,
            "delta_mb": end_snapshot.rss_mb - start_snapshot.rss_mb,
            "duration_seconds": (end_snapshot.timestamp - start_snapshot.timestamp).total_seconds()
        }

    def check_memory_pressure(self, threshold_percent: float = 80.0) -> bool:
        """
        Check if system is under memory pressure.

        Args:
            threshold_percent: Memory usage threshold

        Returns:
            True if memory usage exceeds threshold
        """
        system_mem = psutil.virtual_memory()
        return system_mem.percent > threshold_percent

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and report results.

        Returns:
            Dictionary with GC stats
        """
        before = self.get_current_usage()

        # Force collection
        collected = gc.collect()

        after = self.get_current_usage()

        freed_mb = before["rss_mb"] - after["rss_mb"]

        self.logger.info(f"Garbage collection: freed {freed_mb:.2f} MB, collected {collected} objects")

        return {
            "objects_collected": collected,
            "memory_before_mb": before["rss_mb"],
            "memory_after_mb": after["rss_mb"],
            "memory_freed_mb": freed_mb
        }

    def detect_memory_leak(self, window_size: int = 10) -> Optional[Dict[str, Any]]:
        """
        Detect potential memory leaks by analyzing trends.

        Args:
            window_size: Number of snapshots to analyze

        Returns:
            Dictionary with leak detection results or None
        """
        if len(self._snapshots) < window_size:
            return None

        # Get recent snapshots
        recent = self._snapshots[-window_size:]
        rss_values = [s.rss_mb for s in recent]

        # Calculate trend
        avg_growth = (rss_values[-1] - rss_values[0]) / window_size

        # Check if memory is consistently growing
        if avg_growth > 1.0:  # Growing by > 1 MB per snapshot
            return {
                "leak_detected": True,
                "avg_growth_per_snapshot_mb": avg_growth,
                "total_growth_mb": rss_values[-1] - rss_values[0],
                "current_usage_mb": rss_values[-1],
                "recommendation": "Consider forcing garbage collection or reducing batch size"
            }

        return {
            "leak_detected": False,
            "avg_growth_per_snapshot_mb": avg_growth
        }

    def get_summary(self) -> str:
        """
        Get human-readable memory summary.

        Returns:
            Summary string
        """
        stats = self.get_memory_stats()
        current = stats["current"]

        lines = [
            "Memory Usage Summary:",
            f"  Current: {current['rss_mb']:.2f} MB ({current['percent']:.1f}%)",
            f"  Peak: {stats['peak_mb']:.2f} MB",
            f"  Average: {stats['avg_mb']:.2f} MB",
            f"  Growth: {stats['memory_growth_mb']:.2f} MB",
            f"  System Available: {current['system_available_mb']:.2f} MB"
        ]

        # Check for memory pressure
        if self.check_memory_pressure():
            lines.append("  ⚠️  HIGH MEMORY USAGE DETECTED")

        # Check for leaks
        leak_info = self.detect_memory_leak()
        if leak_info and leak_info.get("leak_detected"):
            lines.append(f"  ⚠️  POTENTIAL MEMORY LEAK: {leak_info['avg_growth_per_snapshot_mb']:.2f} MB/snapshot")

        return "\n".join(lines)

    def reset(self):
        """Reset all tracking data."""
        self._snapshots.clear()
        self._operation_snapshots.clear()
        self._peak_memory_mb = 0.0

        if self.enable_detailed_tracking and tracemalloc.is_tracing():
            tracemalloc.clear_traces()

    def stop(self):
        """Stop memory tracking and cleanup."""
        if self.enable_detailed_tracking and tracemalloc.is_tracing():
            tracemalloc.stop()


# Global singleton
_global_tracker: Optional[MemoryTracker] = None


def get_tracker(enable_detailed: bool = False) -> MemoryTracker:
    """
    Get global MemoryTracker singleton.

    Args:
        enable_detailed: Enable detailed tracking

    Returns:
        MemoryTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker(enable_detailed_tracking=enable_detailed)
    return _global_tracker
