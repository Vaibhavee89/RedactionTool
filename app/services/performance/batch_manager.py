"""
Batch Manager - Configurable batch processing for optimal memory usage.

Features:
- Dynamic batch size adjustment
- Memory-aware batching
- Adaptive batch sizing based on file size
- Progress tracking
- Throughput optimization
"""

import os
import psutil
from pathlib import Path
from typing import List, Iterator, Dict, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 100
    max_batch_size: int = 1000
    min_batch_size: int = 10
    memory_threshold_percent: float = 80.0
    adaptive: bool = True


class BatchManager:
    """
    Intelligent batch manager with adaptive sizing.

    Automatically adjusts batch sizes based on:
    - Available memory
    - File sizes
    - Processing performance
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize BatchManager.

        Args:
            config: Batch configuration
        """
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self._processing_times: List[float] = []
        self._batch_sizes: List[int] = []

    def create_batches(
        self,
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> Iterator[List[Any]]:
        """
        Create batches from list of items.

        Args:
            items: List of items to batch
            batch_size: Override default batch size

        Yields:
            Batches of items
        """
        batch_size = batch_size or self.config.batch_size

        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def create_adaptive_batches(
        self,
        file_paths: List[str]
    ) -> Iterator[List[str]]:
        """
        Create batches with adaptive sizing based on file sizes.

        Args:
            file_paths: List of file paths

        Yields:
            Batches of file paths
        """
        if not self.config.adaptive:
            # Use fixed batch size
            yield from self.create_batches(file_paths, self.config.batch_size)
            return

        # Sort files by size (process smaller files in larger batches)
        file_sizes = [(f, self._get_file_size(f)) for f in file_paths]
        file_sizes.sort(key=lambda x: x[1])

        current_batch = []
        current_batch_size_bytes = 0
        max_batch_size_bytes = 100 * 1024 * 1024  # 100 MB per batch

        for file_path, file_size in file_sizes:
            # Check if adding this file would exceed memory threshold
            if self._check_memory_usage():
                # Memory pressure - yield current batch and wait
                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_batch_size_bytes = 0

            # Add file to batch
            current_batch.append(file_path)
            current_batch_size_bytes += file_size

            # Yield batch if limits reached
            if (len(current_batch) >= self.config.max_batch_size or
                current_batch_size_bytes >= max_batch_size_bytes):
                yield current_batch
                current_batch = []
                current_batch_size_bytes = 0

        # Yield remaining files
        if current_batch:
            yield current_batch

    def _get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes
        """
        try:
            return Path(file_path).stat().st_size
        except Exception:
            return 0

    def _check_memory_usage(self) -> bool:
        """
        Check if memory usage exceeds threshold.

        Returns:
            True if memory pressure detected
        """
        try:
            memory = psutil.virtual_memory()
            return memory.percent > self.config.memory_threshold_percent
        except Exception:
            return False

    def get_optimal_batch_size(
        self,
        file_paths: List[str],
        available_memory_mb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal batch size based on file sizes and available memory.

        Args:
            file_paths: List of file paths
            available_memory_mb: Available memory in MB (auto-detected if None)

        Returns:
            Optimal batch size
        """
        if not file_paths:
            return self.config.batch_size

        # Get available memory
        if available_memory_mb is None:
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)

        # Calculate average file size
        total_size = sum(self._get_file_size(f) for f in file_paths[:100])  # Sample first 100
        avg_file_size_mb = (total_size / min(len(file_paths), 100)) / (1024 * 1024)

        if avg_file_size_mb == 0:
            return self.config.batch_size

        # Estimate how many files can fit in memory
        # Use 50% of available memory for safety
        usable_memory_mb = available_memory_mb * 0.5
        estimated_batch_size = int(usable_memory_mb / avg_file_size_mb)

        # Clamp to configured limits
        batch_size = max(
            self.config.min_batch_size,
            min(estimated_batch_size, self.config.max_batch_size)
        )

        self.logger.info(
            f"Optimal batch size: {batch_size} "
            f"(avg file: {avg_file_size_mb:.2f} MB, available mem: {available_memory_mb:.2f} MB)"
        )

        return batch_size

    def record_batch_performance(self, batch_size: int, processing_time_ms: float):
        """
        Record batch processing performance for future optimization.

        Args:
            batch_size: Size of the batch
            processing_time_ms: Time taken to process batch
        """
        self._batch_sizes.append(batch_size)
        self._processing_times.append(processing_time_ms)

        # Keep only last 100 records
        if len(self._batch_sizes) > 100:
            self._batch_sizes = self._batch_sizes[-100:]
            self._processing_times = self._processing_times[-100:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get batch processing performance statistics.

        Returns:
            Dictionary with performance stats
        """
        if not self._processing_times:
            return {}

        return {
            "batches_processed": len(self._processing_times),
            "avg_batch_size": sum(self._batch_sizes) / len(self._batch_sizes),
            "avg_processing_time_ms": sum(self._processing_times) / len(self._processing_times),
            "min_processing_time_ms": min(self._processing_times),
            "max_processing_time_ms": max(self._processing_times),
            "total_items_processed": sum(self._batch_sizes)
        }

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory information.

        Returns:
            Dictionary with memory stats
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent_used": memory.percent,
                "threshold_percent": self.config.memory_threshold_percent,
                "memory_pressure": memory.percent > self.config.memory_threshold_percent
            }
        except Exception as e:
            self.logger.error(f"Error getting memory info: {e}")
            return {}

    def get_recommended_config(self, file_count: int, total_size_mb: float) -> Dict[str, Any]:
        """
        Get recommended batch configuration for given workload.

        Args:
            file_count: Total number of files
            total_size_mb: Total size of files in MB

        Returns:
            Recommended configuration
        """
        avg_file_size_mb = total_size_mb / file_count if file_count > 0 else 0

        # Get system memory
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)

        # Calculate recommendations
        if avg_file_size_mb < 1:  # Small files (< 1 MB)
            batch_size = min(500, self.config.max_batch_size)
            mode = "thread"  # I/O bound
        elif avg_file_size_mb < 10:  # Medium files (1-10 MB)
            batch_size = min(100, self.config.max_batch_size)
            mode = "process"  # CPU bound
        else:  # Large files (> 10 MB)
            batch_size = min(20, self.config.max_batch_size)
            mode = "process"

        # Adjust for available memory
        if available_mb < 1000:  # < 1 GB available
            batch_size = max(10, batch_size // 2)

        return {
            "batch_size": batch_size,
            "processing_mode": mode,
            "adaptive_batching": True,
            "parallel_workers": os.cpu_count() or 4,
            "estimated_memory_usage_mb": batch_size * avg_file_size_mb * 2,  # 2x for safety
            "rationale": {
                "avg_file_size_mb": avg_file_size_mb,
                "available_memory_mb": available_mb,
                "total_files": file_count
            }
        }


def create_batch_manager(
    batch_size: int = 100,
    adaptive: bool = True
) -> BatchManager:
    """
    Factory function to create BatchManager.

    Args:
        batch_size: Default batch size
        adaptive: Enable adaptive batching

    Returns:
        BatchManager instance
    """
    config = BatchConfig(batch_size=batch_size, adaptive=adaptive)
    return BatchManager(config)
