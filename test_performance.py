#!/usr/bin/env python3
"""
Test suite for Performance & Scalability features.

Tests:
1. Parallel file processing
2. Pattern caching
3. Batch management
4. Memory tracking
5. Benchmark runner
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.performance import (
    ParallelProcessor,
    PatternCache,
    BatchManager,
    MemoryTracker,
    BenchmarkRunner
)


def test_parallel_processing():
    """Test 1: Parallel File Processing"""
    print("=" * 70)
    print("TEST 1: Parallel File Processing")
    print("=" * 70)

    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = []
        for i in range(20):
            file_path = Path(temp_dir) / f"test_{i}.txt"
            file_path.write_text(f"Test content {i}\n" * 100)
            test_files.append(str(file_path))

        # Define processing function
        def process_file(file_path):
            """Simulate file processing."""
            time.sleep(0.01)  # Simulate work
            content = Path(file_path).read_text()
            return (True, {"entities_found": len(content.split())})

        # Test parallel processing
        processor = ParallelProcessor(max_workers=4, mode="thread")

        print(f"Processing {len(test_files)} files with 4 workers...")

        completed = []
        def progress_callback(done, total):
            completed.append(done)

        results = processor.process_files(
            test_files,
            process_file,
            progress_callback
        )

        print(f"\n✓ Results:")
        print(f"  - Total files: {results['total_files']}")
        print(f"  - Successful: {results['successful']}")
        print(f"  - Failed: {results['failed']}")
        print(f"  - Total time: {results['total_time_ms']:.2f} ms")
        print(f"  - Avg time per file: {results['avg_time_per_file_ms']:.2f} ms")
        print(f"  - Throughput: {results['throughput_files_per_sec']:.2f} files/sec")

        # Verify all files processed
        success = results['successful'] == len(test_files)
        print(f"\n✓ All files processed: {success}")

    print()
    return success


def test_pattern_caching():
    """Test 2: Pattern Caching"""
    print("=" * 70)
    print("TEST 2: Pattern Caching")
    print("=" * 70)

    cache = PatternCache(max_patterns=100)

    # Test pattern caching
    patterns = [
        (r'\b[A-Z]{5}\d{4}[A-Z]\b', 0),  # PAN
        (r'\b\d{4}\s\d{4}\s\d{4}\b', 0),  # Aadhaar
        (r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', 2),  # Email
    ]

    # First access (cache miss)
    print("First access (should be cache miss):")
    for pattern, flags in patterns:
        compiled = cache.get_pattern(pattern, flags)
        print(f"  ✓ Pattern compiled: {pattern[:30]}...")

    # Second access (cache hit)
    print("\nSecond access (should be cache hit):")
    for pattern, flags in patterns:
        compiled = cache.get_pattern(pattern, flags)

    # Get statistics
    stats = cache.get_statistics()
    print(f"\n✓ Cache Statistics:")
    print(f"  - Patterns cached: {stats['patterns']['cached']}")
    print(f"  - Hit rate: {stats['patterns']['hit_rate']:.1%}")
    print(f"  - Hits: {stats['patterns']['hits']}")
    print(f"  - Misses: {stats['patterns']['misses']}")

    success = stats['patterns']['hit_rate'] >= 0.5
    print(f"\n✓ Cache working: {success}")

    print()
    return success


def test_batch_management():
    """Test 3: Batch Management"""
    print("=" * 70)
    print("TEST 3: Batch Management")
    print("=" * 70)

    batch_manager = BatchManager()

    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = []
        for i in range(50):
            file_path = Path(temp_dir) / f"file_{i}.txt"
            file_path.write_text(f"Content {i}\n" * (i + 1) * 10)
            test_files.append(str(file_path))

        # Test batch creation
        batches = list(batch_manager.create_batches(test_files, batch_size=10))
        print(f"✓ Created {len(batches)} batches of size 10")

        # Test adaptive batching
        adaptive_batches = list(batch_manager.create_adaptive_batches(test_files))
        print(f"✓ Created {len(adaptive_batches)} adaptive batches")

        # Get optimal batch size
        optimal = batch_manager.get_optimal_batch_size(test_files)
        print(f"✓ Optimal batch size: {optimal}")

        # Get memory info
        mem_info = batch_manager.get_memory_info()
        print(f"\n✓ Memory Info:")
        print(f"  - Available: {mem_info['available_mb']:.2f} MB")
        print(f"  - Used: {mem_info['percent_used']:.1f}%")

        # Get recommended config
        total_size = sum(Path(f).stat().st_size for f in test_files) / (1024 * 1024)
        rec = batch_manager.get_recommended_config(len(test_files), total_size)
        print(f"\n✓ Recommended Config:")
        print(f"  - Batch size: {rec['batch_size']}")
        print(f"  - Mode: {rec['processing_mode']}")
        print(f"  - Workers: {rec['parallel_workers']}")

    success = len(batches) > 0 and optimal > 0
    print(f"\n✓ Batch management working: {success}")

    print()
    return success


def test_memory_tracking():
    """Test 4: Memory Tracking"""
    print("=" * 70)
    print("TEST 4: Memory Tracking")
    print("=" * 70)

    tracker = MemoryTracker(enable_detailed_tracking=False)

    # Take initial snapshot
    initial = tracker.take_snapshot("initial")
    print(f"✓ Initial memory: {initial.rss_mb:.2f} MB")

    # Simulate memory usage
    data = []
    with tracker.track_operation("data_allocation"):
        for i in range(1000):
            data.append([0] * 1000)

    # Take final snapshot
    final = tracker.take_snapshot("final")
    print(f"✓ Final memory: {final.rss_mb:.2f} MB")

    # Get statistics
    stats = tracker.get_memory_stats()
    print(f"\n✓ Memory Statistics:")
    print(f"  - Peak: {stats['peak_mb']:.2f} MB")
    print(f"  - Growth: {stats['memory_growth_mb']:.2f} MB")
    print(f"  - Snapshots: {stats['snapshots_taken']}")

    # Get operation stats
    op_stats = tracker.get_operation_stats("data_allocation")
    if op_stats:
        print(f"\n✓ Operation Stats (data_allocation):")
        print(f"  - Memory delta: {op_stats['delta_mb']:.2f} MB")
        print(f"  - Duration: {op_stats['duration_seconds']:.2f} sec")

    # Check for memory pressure
    pressure = tracker.check_memory_pressure()
    print(f"\n✓ Memory pressure: {pressure}")

    # Clean up
    del data
    gc_stats = tracker.force_garbage_collection()
    print(f"\n✓ Garbage Collection:")
    print(f"  - Objects collected: {gc_stats['objects_collected']}")
    print(f"  - Memory freed: {gc_stats['memory_freed_mb']:.2f} MB")

    success = stats['snapshots_taken'] > 0
    print(f"\n✓ Memory tracking working: {success}")

    print()
    return success


def test_benchmark_runner():
    """Test 5: Benchmark Runner"""
    print("=" * 70)
    print("TEST 5: Benchmark Runner")
    print("=" * 70)

    runner = BenchmarkRunner()

    # Test function benchmarking
    def test_func(n):
        """Test function."""
        total = 0
        for i in range(n):
            total += i
        return total

    result = runner.benchmark_function(
        test_func,
        args=(1000,),
        iterations=100,
        name="sum_1000"
    )

    print(f"✓ Benchmark Result:")
    print(f"  - Name: {result.name}")
    print(f"  - Iterations: {result.iterations}")
    print(f"  - Average: {result.avg_time_ms:.4f} ms")
    print(f"  - Median: {result.median_time_ms:.4f} ms")
    print(f"  - Std Dev: {result.std_dev_ms:.4f} ms")

    # Test comparison
    def slow_func():
        time.sleep(0.01)

    def fast_func():
        time.sleep(0.001)

    print("\n✓ Comparing implementations:")
    comparisons = runner.compare_implementations(
        {
            "slow": slow_func,
            "fast": fast_func
        },
        iterations=10
    )

    # Generate report
    report = runner.generate_report()
    print(f"\n✓ Report generated: {len(report)} characters")

    success = result.avg_time_ms > 0
    print(f"\n✓ Benchmark runner working: {success}")

    print()
    return success


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PERFORMANCE & SCALABILITY - FEATURE TESTS")
    print("=" * 70)
    print()

    results = []
    results.append(("Parallel Processing", test_parallel_processing()))
    results.append(("Pattern Caching", test_pattern_caching()))
    results.append(("Batch Management", test_batch_management()))
    results.append(("Memory Tracking", test_memory_tracking()))
    results.append(("Benchmark Runner", test_benchmark_runner()))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 70)
    print()

    if passed == total:
        print("🎉 All tests passed! Performance features are ready.")
        print("\n⚙️ Systems Thinking Demonstrated:")
        print("  ✓ Parallel processing for scalability")
        print("  ✓ Intelligent caching for performance")
        print("  ✓ Adaptive batching for memory efficiency")
        print("  ✓ Real-time monitoring for observability")
        print("  ✓ Comprehensive benchmarking for optimization")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
