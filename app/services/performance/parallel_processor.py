"""
Parallel File Processing - Multi-core document processing for scalability.

Features:
- Multi-process parallel processing
- Thread pool for I/O-bound tasks
- Automatic worker scaling
- Progress tracking
- Error isolation per file
"""

import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    file_path: str
    success: bool
    processing_time_ms: float
    entities_found: int
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None


class ParallelProcessor:
    """
    Parallel file processor for high-throughput redaction.

    Uses multiprocessing for CPU-bound tasks and threading for I/O-bound tasks.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        mode: str = "process",  # "process" or "thread"
        chunk_size: int = 10
    ):
        """
        Initialize ParallelProcessor.

        Args:
            max_workers: Maximum number of workers (default: CPU count)
            mode: Processing mode - "process" for CPU-bound, "thread" for I/O-bound
            chunk_size: Number of files to process in each chunk
        """
        self.max_workers = max_workers or self._get_optimal_workers()
        self.mode = mode
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    def _get_optimal_workers(self) -> int:
        """
        Determine optimal number of workers based on system resources.

        Returns:
            Number of workers
        """
        cpu_count = os.cpu_count() or 4

        # Leave one core free for system operations
        optimal = max(1, cpu_count - 1)

        return optimal

    def process_files(
        self,
        file_paths: List[str],
        process_func: Callable[[str], Tuple[bool, Dict]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            process_func: Function to process each file
                         Should return (success, result_dict)
            progress_callback: Optional callback for progress updates
                              Receives (completed, total)

        Returns:
            Dictionary with processing results and statistics
        """
        if not file_paths:
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "total_time_ms": 0
            }

        start_time = datetime.now()
        results = []
        successful = 0
        failed = 0

        # Choose executor based on mode
        executor_class = ProcessPoolExecutor if self.mode == "process" else ThreadPoolExecutor

        self.logger.info(f"Processing {len(file_paths)} files with {self.max_workers} workers ({self.mode} mode)")

        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file_path, process_func): file_path
                    for file_path in file_paths
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed += 1

                    try:
                        result = future.result()
                        results.append(result)

                        if result.success:
                            successful += 1
                        else:
                            failed += 1

                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        results.append(ProcessingResult(
                            file_path=file_path,
                            success=False,
                            processing_time_ms=0,
                            entities_found=0,
                            error_message=str(e)
                        ))
                        failed += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(completed, len(file_paths))

        except Exception as e:
            self.logger.error(f"Parallel processing error: {e}")
            raise

        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "total_files": len(file_paths),
            "successful": successful,
            "failed": failed,
            "results": results,
            "total_time_ms": total_time_ms,
            "avg_time_per_file_ms": total_time_ms / len(file_paths) if file_paths else 0,
            "throughput_files_per_sec": len(file_paths) / (total_time_ms / 1000) if total_time_ms > 0 else 0
        }

    def _process_single_file(
        self,
        file_path: str,
        process_func: Callable[[str], Tuple[bool, Dict]]
    ) -> ProcessingResult:
        """
        Process a single file (called by workers).

        Args:
            file_path: Path to file
            process_func: Processing function

        Returns:
            ProcessingResult
        """
        start_time = datetime.now()

        try:
            success, result_dict = process_func(file_path)

            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return ProcessingResult(
                file_path=file_path,
                success=success,
                processing_time_ms=processing_time_ms,
                entities_found=result_dict.get("entities_found", 0),
                error_message=result_dict.get("error") if not success else None,
                metadata=result_dict
            )

        except Exception as e:
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return ProcessingResult(
                file_path=file_path,
                success=False,
                processing_time_ms=processing_time_ms,
                entities_found=0,
                error_message=str(e)
            )

    def process_in_batches(
        self,
        file_paths: List[str],
        process_func: Callable[[str], Tuple[bool, Dict]],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Process files in batches for better memory management.

        Args:
            file_paths: List of file paths
            process_func: Processing function
            batch_size: Size of each batch (default: chunk_size)
            progress_callback: Progress callback

        Returns:
            Combined results from all batches
        """
        batch_size = batch_size or self.chunk_size

        all_results = []
        total_successful = 0
        total_failed = 0
        total_time_ms = 0

        # Split into batches
        batches = [
            file_paths[i:i + batch_size]
            for i in range(0, len(file_paths), batch_size)
        ]

        self.logger.info(f"Processing {len(file_paths)} files in {len(batches)} batches of {batch_size}")

        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")

            batch_result = self.process_files(
                batch,
                process_func,
                progress_callback
            )

            all_results.extend(batch_result["results"])
            total_successful += batch_result["successful"]
            total_failed += batch_result["failed"]
            total_time_ms += batch_result["total_time_ms"]

        return {
            "total_files": len(file_paths),
            "successful": total_successful,
            "failed": total_failed,
            "results": all_results,
            "total_time_ms": total_time_ms,
            "batches_processed": len(batches),
            "avg_time_per_file_ms": total_time_ms / len(file_paths) if file_paths else 0,
            "throughput_files_per_sec": len(file_paths) / (total_time_ms / 1000) if total_time_ms > 0 else 0
        }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for performance tuning.

        Returns:
            Dictionary with system info
        """
        return {
            "cpu_count": os.cpu_count(),
            "max_workers": self.max_workers,
            "processing_mode": self.mode,
            "chunk_size": self.chunk_size,
            "optimal_workers": self._get_optimal_workers()
        }


def create_processor(
    max_workers: Optional[int] = None,
    mode: str = "process"
) -> ParallelProcessor:
    """
    Factory function to create a ParallelProcessor.

    Args:
        max_workers: Maximum number of workers
        mode: Processing mode ("process" or "thread")

    Returns:
        ParallelProcessor instance
    """
    return ParallelProcessor(max_workers=max_workers, mode=mode)
