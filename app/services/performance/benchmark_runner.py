"""
Benchmark Runner - Comprehensive performance benchmarking and latency measurement.

Features:
- End-to-end latency benchmarks
- Component-level profiling
- Throughput measurement
- Scalability testing
- Comparison reports
"""

import time
import statistics
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    throughput: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for performance testing.

    Measures latency, throughput, and system performance.
    """

    def __init__(self):
        """Initialize BenchmarkRunner."""
        self.logger = logging.getLogger(__name__)
        self._results: List[BenchmarkResult] = []

    def benchmark_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 100,
        warmup: int = 10,
        name: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Benchmark a function with multiple iterations.

        Args:
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            iterations: Number of iterations
            warmup: Number of warmup iterations
            name: Name for the benchmark

        Returns:
            BenchmarkResult
        """
        kwargs = kwargs or {}
        name = name or func.__name__

        self.logger.info(f"Benchmarking: {name} ({iterations} iterations)")

        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        # Actual benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            median_time_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0
        )

        self._results.append(result)
        return result

    def benchmark_file_processing(
        self,
        file_path: str,
        process_func: Callable[[str], Any],
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark file processing performance.

        Args:
            file_path: Path to test file
            process_func: Function to process file
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        # Get file size for throughput calculation
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        result = self.benchmark_function(
            process_func,
            args=(file_path,),
            iterations=iterations,
            warmup=2,
            name=f"file_processing_{Path(file_path).name}"
        )

        # Calculate throughput (MB/s)
        result.throughput = file_size_mb / (result.avg_time_ms / 1000)
        result.metadata["file_size_mb"] = file_size_mb

        return result

    def benchmark_batch_processing(
        self,
        file_paths: List[str],
        process_func: Callable[[List[str]], Any],
        batch_sizes: List[int] = None
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark batch processing with different batch sizes.

        Args:
            file_paths: List of file paths
            process_func: Function to process batch
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary mapping batch size to results
        """
        batch_sizes = batch_sizes or [10, 50, 100, 200]
        results = {}

        for batch_size in batch_sizes:
            # Create batches
            batches = [
                file_paths[i:i + batch_size]
                for i in range(0, len(file_paths), batch_size)
            ]

            if not batches:
                continue

            # Benchmark first batch
            result = self.benchmark_function(
                process_func,
                args=(batches[0],),
                iterations=5,
                warmup=1,
                name=f"batch_size_{batch_size}"
            )

            result.throughput = len(batches[0]) / (result.avg_time_ms / 1000)
            results[batch_size] = result

            self.logger.info(f"Batch size {batch_size}: {result.avg_time_ms:.2f} ms")

        return results

    def benchmark_parallel_scaling(
        self,
        process_func: Callable[[int], Any],
        worker_counts: List[int] = None
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark parallel processing with different worker counts.

        Args:
            process_func: Function that takes worker count
            worker_counts: List of worker counts to test

        Returns:
            Dictionary mapping worker count to results
        """
        worker_counts = worker_counts or [1, 2, 4, 8]
        results = {}

        for workers in worker_counts:
            result = self.benchmark_function(
                process_func,
                args=(workers,),
                iterations=5,
                warmup=1,
                name=f"workers_{workers}"
            )

            results[workers] = result
            self.logger.info(f"Workers {workers}: {result.avg_time_ms:.2f} ms")

        return results

    def benchmark_entity_detection(
        self,
        text_samples: List[str],
        detector_func: Callable[[str], List],
        detector_name: str = "detector"
    ) -> BenchmarkResult:
        """
        Benchmark entity detection performance.

        Args:
            text_samples: List of text samples
            detector_func: Entity detection function
            detector_name: Name of the detector

        Returns:
            BenchmarkResult
        """
        total_chars = sum(len(text) for text in text_samples)

        def process_all():
            for text in text_samples:
                detector_func(text)

        result = self.benchmark_function(
            process_all,
            iterations=10,
            warmup=2,
            name=f"{detector_name}_detection"
        )

        # Calculate throughput (chars/second)
        result.throughput = total_chars / (result.avg_time_ms / 1000)
        result.metadata["total_chars"] = total_chars
        result.metadata["samples"] = len(text_samples)

        return result

    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        test_args: tuple = (),
        iterations: int = 50
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple implementation performance.

        Args:
            implementations: Dictionary of name -> function
            test_args: Arguments for all functions
            iterations: Number of iterations

        Returns:
            Dictionary of results
        """
        results = {}

        for name, func in implementations.items():
            result = self.benchmark_function(
                func,
                args=test_args,
                iterations=iterations,
                name=name
            )
            results[name] = result

        # Print comparison
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: Dict[str, BenchmarkResult]):
        """Print comparison table."""
        if not results:
            return

        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)

        # Sort by avg time
        sorted_results = sorted(results.items(), key=lambda x: x[1].avg_time_ms)

        # Find fastest for percentage calculation
        fastest_time = sorted_results[0][1].avg_time_ms

        for name, result in sorted_results:
            speedup = fastest_time / result.avg_time_ms
            slower_percent = ((result.avg_time_ms / fastest_time) - 1) * 100

            print(f"\n{name}:")
            print(f"  Average: {result.avg_time_ms:.2f} ms")
            print(f"  Median:  {result.median_time_ms:.2f} ms")
            print(f"  Std Dev: {result.std_dev_ms:.2f} ms")

            if name == sorted_results[0][0]:
                print(f"  ⚡ FASTEST (baseline)")
            else:
                print(f"  {slower_percent:.1f}% slower ({speedup:.2f}x)")

        print("=" * 70)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all benchmarks.

        Returns:
            Dictionary with summary
        """
        if not self._results:
            return {}

        return {
            "total_benchmarks": len(self._results),
            "results": [
                {
                    "name": r.name,
                    "avg_time_ms": r.avg_time_ms,
                    "throughput": r.throughput,
                    "iterations": r.iterations
                }
                for r in self._results
            ]
        }

    def save_results(self, output_path: str):
        """
        Save benchmark results to JSON file.

        Args:
            output_path: Path to output file
        """
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "avg_time_ms": r.avg_time_ms,
                    "min_time_ms": r.min_time_ms,
                    "max_time_ms": r.max_time_ms,
                    "median_time_ms": r.median_time_ms,
                    "std_dev_ms": r.std_dev_ms,
                    "throughput": r.throughput,
                    "metadata": r.metadata
                }
                for r in self._results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"Benchmark results saved to: {output_path}")

    def generate_report(self) -> str:
        """
        Generate human-readable benchmark report.

        Returns:
            Report string
        """
        if not self._results:
            return "No benchmark results available."

        lines = [
            "=" * 70,
            "BENCHMARK REPORT",
            "=" * 70,
            f"Total Benchmarks: {len(self._results)}",
            f"Generated: {datetime.now().isoformat()}",
            ""
        ]

        for result in self._results:
            lines.extend([
                f"Benchmark: {result.name}",
                f"  Iterations: {result.iterations}",
                f"  Average: {result.avg_time_ms:.2f} ms",
                f"  Median: {result.median_time_ms:.2f} ms",
                f"  Min: {result.min_time_ms:.2f} ms",
                f"  Max: {result.max_time_ms:.2f} ms",
                f"  Std Dev: {result.std_dev_ms:.2f} ms"
            ])

            if result.throughput:
                lines.append(f"  Throughput: {result.throughput:.2f} items/sec")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def quick_benchmark(func: Callable, iterations: int = 100) -> float:
    """
    Quick benchmark of a function.

    Args:
        func: Function to benchmark
        iterations: Number of iterations

    Returns:
        Average time in milliseconds
    """
    runner = BenchmarkRunner()
    result = runner.benchmark_function(func, iterations=iterations)
    return result.avg_time_ms
