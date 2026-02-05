#!/usr/bin/env python3
"""
PII Redaction Tool - Command Line Interface

Usage:
    redact input_dir/ output_dir/ --policy india_finance.yaml --mode mask --log audit.json

Features:
- Batch processing of multiple files
- Configurable output formats
- JSON summary output
- Exit codes for pipeline integration
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.pii.ensemble_detector import EnsembleDetector
from app.services.redaction.policy_manager import PolicyManager
from app.services.redaction.enhanced_redactor import EnhancedRedactor


# Exit codes for pipeline integration
EXIT_SUCCESS = 0
EXIT_ERROR_GENERAL = 1
EXIT_ERROR_INPUT = 2
EXIT_ERROR_OUTPUT = 3
EXIT_ERROR_POLICY = 4
EXIT_ERROR_PROCESSING = 5


class CLIRedactionProcessor:
    """
    CLI processor for batch PII redaction.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        policy_file: Optional[str] = None,
        mode: str = 'block',
        confidence: float = 0.5,
        formats: Optional[List[str]] = None,
        log_file: Optional[str] = None,
        verbose: bool = False,
        dry_run: bool = False
    ):
        """
        Initialize CLI processor.

        Args:
            input_path: Input directory or file
            output_path: Output directory
            policy_file: Path to policy YAML file
            mode: Redaction mode (block, mask, label, etc.)
            confidence: Confidence threshold (0.0-1.0)
            formats: Output formats (json, text, html)
            log_file: Path to audit log file
            verbose: Enable verbose logging
            dry_run: Dry run mode (no files written)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.policy_file = policy_file
        self.mode = mode
        self.confidence = confidence
        self.formats = formats or ['text']
        self.log_file = log_file
        self.verbose = verbose
        self.dry_run = dry_run

        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_entities': 0,
            'redacted_entities': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.detector = None
        self.redactor = None
        self.policy_manager = None
        self.policy = None

        self._initialize_components()

    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.verbose else logging.INFO

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        """Initialize detection and redaction components."""
        try:
            self.logger.info("Initializing PII detection components...")
            self.detector = EnsembleDetector()

            self.logger.info("Initializing redaction components...")
            self.redactor = EnhancedRedactor()

            # Load policy if provided
            if self.policy_file:
                self.logger.info(f"Loading policy from {self.policy_file}...")
                self.policy_manager = PolicyManager()
                self.policy = self.policy_manager.load_policy_from_file(self.policy_file)
                self.logger.info(f"Policy loaded: {self.policy.name}")

            self.logger.info("✓ Components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def validate_inputs(self) -> bool:
        """
        Validate input parameters.

        Returns:
            True if valid, False otherwise
        """
        # Check input exists
        if not self.input_path.exists():
            self.logger.error(f"Input path does not exist: {self.input_path}")
            return False

        # Check output directory can be created
        if not self.dry_run:
            try:
                self.output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Cannot create output directory: {str(e)}")
                return False

        # Check policy file if provided
        if self.policy_file and not Path(self.policy_file).exists():
            self.logger.error(f"Policy file does not exist: {self.policy_file}")
            return False

        # Validate confidence threshold
        if not 0.0 <= self.confidence <= 1.0:
            self.logger.error(f"Confidence must be between 0.0 and 1.0: {self.confidence}")
            return False

        # Validate redaction mode
        valid_modes = ['block', 'mask', 'partial_mask', 'label', 'hash', 'tokenize', 'allow']
        if self.mode not in valid_modes:
            self.logger.error(f"Invalid mode: {self.mode}. Valid modes: {valid_modes}")
            return False

        # Validate output formats
        valid_formats = ['text', 'json', 'html', 'markdown']
        for fmt in self.formats:
            if fmt not in valid_formats:
                self.logger.error(f"Invalid format: {fmt}. Valid formats: {valid_formats}")
                return False

        return True

    def get_input_files(self) -> List[Path]:
        """
        Get list of input files to process.

        Returns:
            List of file paths
        """
        files = []

        if self.input_path.is_file():
            files.append(self.input_path)
        elif self.input_path.is_dir():
            # Supported extensions
            supported_exts = ['.txt', '.text', '.md', '.json', '.csv', '.log']

            for ext in supported_exts:
                files.extend(self.input_path.glob(f'**/*{ext}'))

        self.logger.info(f"Found {len(files)} file(s) to process")
        return sorted(files)

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file.

        Args:
            file_path: Path to input file

        Returns:
            Processing result dictionary
        """
        result = {
            'file': str(file_path),
            'status': 'pending',
            'entities_detected': 0,
            'entities_redacted': 0,
            'errors': []
        }

        try:
            self.logger.info(f"Processing: {file_path.name}")

            # Read input file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()

            if not text.strip():
                self.logger.warning(f"Empty file: {file_path.name}")
                result['status'] = 'skipped'
                result['errors'].append('Empty file')
                self.stats['skipped_files'] += 1
                return result

            # Detect PII
            entities = self.detector.detect(text)
            result['entities_detected'] = len(entities)
            self.stats['total_entities'] += len(entities)

            if len(entities) == 0:
                self.logger.info(f"  No PII detected in {file_path.name}")
                result['status'] = 'no_pii'
                self.stats['processed_files'] += 1

                # Still write output with original text
                if not self.dry_run:
                    self._write_output(file_path, text, entities, result)

                return result

            # Filter by confidence
            entities = [e for e in entities if e.get('confidence', 1.0) >= self.confidence]
            result['entities_redacted'] = len(entities)
            self.stats['redacted_entities'] += len(entities)

            self.logger.info(f"  Detected {len(entities)} PII entities")

            # Apply policy if available
            if self.policy:
                entities = [
                    e for e in entities
                    if self.policy.should_redact(e['entity_type'], e.get('confidence', 1.0))
                ]
                self.logger.info(f"  Policy filtered to {len(entities)} entities")

            # Redact
            redacted_text = self.redactor.redact_text(
                text,
                entities,
                policy=self.policy.name if self.policy else None
            )

            # Write output
            if not self.dry_run:
                self._write_output(file_path, redacted_text, entities, result)

            result['status'] = 'success'
            self.stats['processed_files'] += 1

            self.logger.info(f"  ✓ Successfully processed {file_path.name}")

        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            self.logger.error(f"  ✗ {error_msg}")
            result['status'] = 'error'
            result['errors'].append(error_msg)
            self.stats['failed_files'] += 1
            self.stats['errors'].append(error_msg)

        return result

    def _write_output(
        self,
        input_file: Path,
        redacted_text: str,
        entities: List[Dict[str, Any]],
        result: Dict[str, Any]
    ):
        """
        Write output files in requested formats.

        Args:
            input_file: Original input file path
            redacted_text: Redacted text
            entities: Detected entities
            result: Processing result
        """
        # Calculate relative path to maintain directory structure
        if self.input_path.is_dir():
            rel_path = input_file.relative_to(self.input_path)
        else:
            rel_path = input_file.name

        # Create output subdirectories
        output_file_base = self.output_path / rel_path.parent / rel_path.stem

        output_file_base.parent.mkdir(parents=True, exist_ok=True)

        # Write in requested formats
        for fmt in self.formats:
            if fmt == 'text':
                output_file = output_file_base.with_suffix('.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(redacted_text)
                self.logger.debug(f"  Written: {output_file}")

            elif fmt == 'json':
                output_file = output_file_base.with_suffix('.json')
                output_data = {
                    'original_file': str(input_file),
                    'redacted_text': redacted_text,
                    'entities': entities,
                    'metadata': {
                        'entities_detected': result['entities_detected'],
                        'entities_redacted': result['entities_redacted'],
                        'redaction_mode': self.mode,
                        'confidence_threshold': self.confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"  Written: {output_file}")

            elif fmt == 'html':
                output_file = output_file_base.with_suffix('.html')
                html_content = self._generate_html(redacted_text, entities, input_file)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.debug(f"  Written: {output_file}")

            elif fmt == 'markdown':
                output_file = output_file_base.with_suffix('.md')
                md_content = self._generate_markdown(redacted_text, entities, input_file)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                self.logger.debug(f"  Written: {output_file}")

    def _generate_html(
        self,
        redacted_text: str,
        entities: List[Dict[str, Any]],
        original_file: Path
    ) -> str:
        """Generate HTML output."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Redacted: {original_file.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .content {{ margin-top: 20px; white-space: pre-wrap; }}
        .stats {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
        .entity {{ background: #fff3cd; padding: 2px 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Redacted Document</h1>
        <p><strong>Original:</strong> {original_file}</p>
        <p><strong>Redacted:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats">
        <h2>Redaction Statistics</h2>
        <ul>
            <li>Entities detected: {len(entities)}</li>
            <li>Redaction mode: {self.mode}</li>
            <li>Confidence threshold: {self.confidence}</li>
        </ul>
    </div>

    <div class="content">
        <h2>Redacted Content</h2>
        <pre>{redacted_text}</pre>
    </div>
</body>
</html>"""
        return html

    def _generate_markdown(
        self,
        redacted_text: str,
        entities: List[Dict[str, Any]],
        original_file: Path
    ) -> str:
        """Generate Markdown output."""
        md = f"""# Redacted Document

**Original:** {original_file}
**Redacted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Redaction Statistics

- Entities detected: {len(entities)}
- Redaction mode: {self.mode}
- Confidence threshold: {self.confidence}

## Redacted Content

```
{redacted_text}
```

## Detected Entities

"""
        for i, entity in enumerate(entities, 1):
            md += f"{i}. **{entity['entity_type']}** - Confidence: {entity.get('confidence', 1.0):.2f}\n"

        return md

    def process_batch(self) -> Dict[str, Any]:
        """
        Process batch of files.

        Returns:
            Summary of batch processing
        """
        self.stats['start_time'] = datetime.now().isoformat()

        # Get input files
        files = self.get_input_files()
        self.stats['total_files'] = len(files)

        if len(files) == 0:
            self.logger.warning("No files found to process")
            return self.get_summary()

        # Process each file
        results = []
        for i, file_path in enumerate(files, 1):
            self.logger.info(f"[{i}/{len(files)}] Processing: {file_path.name}")

            result = self.process_file(file_path)
            results.append(result)

        self.stats['end_time'] = datetime.now().isoformat()

        # Generate summary
        summary = self.get_summary()
        summary['results'] = results

        # Write audit log if requested
        if self.log_file and not self.dry_run:
            self._write_audit_log(summary)

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """
        Get processing summary.

        Returns:
            Summary dictionary
        """
        summary = {
            'summary': {
                'total_files': self.stats['total_files'],
                'processed_files': self.stats['processed_files'],
                'failed_files': self.stats['failed_files'],
                'skipped_files': self.stats['skipped_files'],
                'total_entities': self.stats['total_entities'],
                'redacted_entities': self.stats['redacted_entities'],
                'success_rate': (
                    self.stats['processed_files'] / self.stats['total_files'] * 100
                    if self.stats['total_files'] > 0 else 0
                )
            },
            'configuration': {
                'input_path': str(self.input_path),
                'output_path': str(self.output_path),
                'policy_file': self.policy_file,
                'redaction_mode': self.mode,
                'confidence_threshold': self.confidence,
                'output_formats': self.formats,
                'dry_run': self.dry_run
            },
            'timing': {
                'start_time': self.stats['start_time'],
                'end_time': self.stats['end_time']
            },
            'errors': self.stats['errors']
        }

        return summary

    def _write_audit_log(self, summary: Dict[str, Any]):
        """
        Write audit log to JSON file.

        Args:
            summary: Processing summary
        """
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✓ Audit log written to: {log_path}")

        except Exception as e:
            self.logger.error(f"Failed to write audit log: {str(e)}")

    def print_summary(self, summary: Dict[str, Any]):
        """
        Print summary to console.

        Args:
            summary: Processing summary
        """
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)

        stats = summary['summary']
        print(f"\nFiles Processed:")
        print(f"  Total:     {stats['total_files']}")
        print(f"  Success:   {stats['processed_files']}")
        print(f"  Failed:    {stats['failed_files']}")
        print(f"  Skipped:   {stats['skipped_files']}")

        print(f"\nPII Entities:")
        print(f"  Detected:  {stats['total_entities']}")
        print(f"  Redacted:  {stats['redacted_entities']}")

        print(f"\nSuccess Rate: {stats['success_rate']:.1f}%")

        if summary['errors']:
            print(f"\n⚠ Errors ({len(summary['errors'])}):")
            for error in summary['errors'][:5]:
                print(f"  - {error}")
            if len(summary['errors']) > 5:
                print(f"  ... and {len(summary['errors']) - 5} more")

        print("=" * 70 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PII Redaction Tool - Batch processing for automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  redact input_dir/ output_dir/

  # With policy
  redact input_dir/ output_dir/ --policy india_finance.yaml

  # With custom mode and confidence
  redact input_dir/ output_dir/ --mode mask --confidence 0.8

  # Multiple output formats with audit log
  redact input_dir/ output_dir/ --formats text json html --log audit.json

  # Dry run
  redact input_dir/ output_dir/ --dry-run --verbose

Exit Codes:
  0 - Success
  1 - General error
  2 - Input error
  3 - Output error
  4 - Policy error
  5 - Processing error
"""
    )

    # Positional arguments
    parser.add_argument(
        'input',
        help='Input directory or file'
    )

    parser.add_argument(
        'output',
        help='Output directory'
    )

    # Optional arguments
    parser.add_argument(
        '-p', '--policy',
        help='Policy YAML file (e.g., india_finance.yaml)'
    )

    parser.add_argument(
        '-m', '--mode',
        choices=['block', 'mask', 'partial_mask', 'label', 'hash', 'tokenize', 'allow'],
        default='block',
        help='Redaction mode (default: block)'
    )

    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold 0.0-1.0 (default: 0.5)'
    )

    parser.add_argument(
        '-f', '--formats',
        nargs='+',
        choices=['text', 'json', 'html', 'markdown'],
        default=['text'],
        help='Output formats (default: text)'
    )

    parser.add_argument(
        '-l', '--log',
        help='Audit log file (JSON)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Dry run (no files written)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='PII Redaction Tool v1.0.0'
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_arguments()

    print("PII Redaction Tool - CLI")
    print("=" * 70)

    try:
        # Initialize processor
        processor = CLIRedactionProcessor(
            input_path=args.input,
            output_path=args.output,
            policy_file=args.policy,
            mode=args.mode,
            confidence=args.confidence,
            formats=args.formats,
            log_file=args.log,
            verbose=args.verbose,
            dry_run=args.dry_run
        )

        # Validate inputs
        if not processor.validate_inputs():
            print("\n✗ Validation failed")
            return EXIT_ERROR_INPUT

        # Process batch
        print("\nStarting batch processing...")
        summary = processor.process_batch()

        # Print summary
        processor.print_summary(summary)

        # Determine exit code
        if summary['summary']['failed_files'] > 0:
            print("⚠ Some files failed to process")
            return EXIT_ERROR_PROCESSING
        elif summary['summary']['processed_files'] == 0:
            print("⚠ No files were processed")
            return EXIT_ERROR_GENERAL
        else:
            print("✓ Batch processing completed successfully")
            return EXIT_SUCCESS

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return EXIT_ERROR_GENERAL

    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_ERROR_GENERAL


if __name__ == "__main__":
    sys.exit(main())
