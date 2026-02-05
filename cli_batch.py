#!/usr/bin/env python3
"""
CLI tool for batch processing and streaming mode.
Provides command-line access to advanced RedactionTool features.
"""

import argparse
import sys
import os
from pathlib import Path

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ingestion.batch_processor import BatchProcessor
from app.services.ingestion.streaming_processor import StreamingProcessor
from app.core.config import Config


def batch_mode(args):
    """Run batch processing mode."""
    print("🚀 Starting Batch Processing...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Recursive: {args.recursive}")

    processor = BatchProcessor()

    def progress_callback(filename, progress):
        print(f"Processing: {filename} - {int(progress*100)}%")

    if os.path.isdir(args.input):
        # Process directory
        results = processor.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            recursive=args.recursive,
            file_types=args.types.split(',') if args.types else None,
            progress_callback=progress_callback
        )
    else:
        print("❌ Error: Input must be a directory for batch mode")
        return 1

    # Display results
    print("\n" + "="*50)
    print("📊 BATCH PROCESSING RESULTS")
    print("="*50)
    print(f"Total Files: {results['total_files']}")
    print(f"✅ Processed: {len(results['processed'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"\n📈 Statistics:")
    print(f"  - Text Documents: {results['stats']['text_documents']}")
    print(f"  - Images: {results['stats']['images']}")
    print(f"  - Videos: {results['stats']['videos']}")
    print(f"  - Total PII Found: {results['stats']['total_pii_found']}")

    if results['failed']:
        print(f"\n⚠️ Failed Files:")
        for failed in results['failed']:
            print(f"  - {failed['file']}: {failed['error']}")

    print(f"\n✅ Output saved to: {args.output}")
    return 0


def streaming_mode(args):
    """Run streaming processing mode for large files."""
    print("🌊 Starting Streaming Mode...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Chunk Size: {args.chunk_size} chars")

    processor = StreamingProcessor(chunk_size=args.chunk_size)

    # Estimate processing time
    estimate = processor.estimate_processing_time(args.input)
    print(f"\n⏱️ Estimated Processing Time:")
    print(f"  - File Size: {estimate['file_size_bytes'] / (1024*1024):.2f} MB")
    print(f"  - Estimated: {estimate['estimated_minutes']:.1f} minutes")
    print()

    def progress_callback(progress):
        print(f"Progress: {int(progress*100)}%", end='\r')

    try:
        result = processor.process_large_text_file(
            file_path=args.input,
            output_path=args.output,
            overlap=args.overlap,
            progress_callback=progress_callback
        )

        print("\n" + "="*50)
        print("📊 STREAMING PROCESSING RESULTS")
        print("="*50)
        print(f"✅ Processing Complete!")
        print(f"Total PII Found: {result['total_pii_found']}")
        if 'total_pages' in result:
            print(f"Pages Processed: {result['total_pages']}")
        print(f"Output: {result['output']}")

        return 0

    except Exception as e:
        print(f"\n❌ Error during streaming processing: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="RedactionTool Enterprise - Batch & Streaming CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch process a directory (recursive)
  python cli_batch.py batch -i /path/to/files -o /path/to/output -r

  # Batch process specific file types
  python cli_batch.py batch -i /path/to/files -o /path/to/output --types .pdf,.docx

  # Stream process large file
  python cli_batch.py stream -i large_file.txt -o output.txt --chunk-size 20000

  # Stream process with custom overlap
  python cli_batch.py stream -i large_file.pdf -o output.txt --overlap 1000
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')

    # Batch mode parser
    batch_parser = subparsers.add_parser('batch', help='Batch processing mode')
    batch_parser.add_argument('-i', '--input', required=True, help='Input directory')
    batch_parser.add_argument('-o', '--output', required=True, help='Output directory')
    batch_parser.add_argument('-r', '--recursive', action='store_true', help='Process subdirectories recursively')
    batch_parser.add_argument('--types', help='Comma-separated file types (e.g., .pdf,.docx)')

    # Streaming mode parser
    stream_parser = subparsers.add_parser('stream', help='Streaming mode for large files')
    stream_parser.add_argument('-i', '--input', required=True, help='Input file path')
    stream_parser.add_argument('-o', '--output', required=True, help='Output file path')
    stream_parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size in characters (default: 10000)')
    stream_parser.add_argument('--overlap', type=int, default=500, help='Overlap between chunks (default: 500)')

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return 1

    # Setup paths
    Config.setup_paths()

    # Route to appropriate mode
    if args.mode == 'batch':
        return batch_mode(args)
    elif args.mode == 'stream':
        return streaming_mode(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
