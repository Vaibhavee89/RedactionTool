import os
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .image_loader import ImageLoader
from .text_loader import TextLoader
from app.services.pii.detector_engine import DetectorEngine
from app.services.redaction.redactor import Redactor
from app.services.redaction.image_redactor import ImageRedactor
from app.services.redaction.video_redactor import VideoRedactor

class BatchProcessor:
    """
    Batch processor for folder-level ingestion and mixed file types.
    Supports processing multiple files of different types in a single run.
    """

    def __init__(self):
        self.loaders = {
            '.txt': TextLoader(),
            '.pdf': PDFLoader(),
            '.docx': DocxLoader(),
            '.png': ImageLoader(),
            '.jpg': ImageLoader(),
            '.jpeg': ImageLoader(),
        }
        self.detector = DetectorEngine()
        self.text_redactor = Redactor()
        self.image_redactor = ImageRedactor()
        self.video_redactor = VideoRedactor()

        # Supported file extensions
        self.text_formats = {'.txt', '.pdf', '.docx'}
        self.image_formats = {'.png', '.jpg', '.jpeg'}
        self.video_formats = {'.mp4', '.avi', '.mov'}

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False,
        file_types: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process all supported files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            recursive: If True, process subdirectories recursively
            file_types: List of file extensions to process (e.g., ['.pdf', '.docx'])
                       If None, processes all supported types
            progress_callback: Optional callback(filename, progress) for tracking

        Returns:
            Dictionary with processing results and statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Collect files to process
        files_to_process = self._collect_files(input_dir, recursive, file_types)

        results = {
            'total_files': len(files_to_process),
            'processed': [],
            'failed': [],
            'stats': {
                'text_documents': 0,
                'images': 0,
                'videos': 0,
                'total_pii_found': 0
            }
        }

        # Process each file
        for i, file_path in enumerate(files_to_process):
            try:
                if progress_callback:
                    progress_callback(os.path.basename(file_path), i / len(files_to_process))

                result = self._process_single_file(file_path, output_dir)
                results['processed'].append(result)

                # Update stats
                ext = Path(file_path).suffix.lower()
                if ext in self.text_formats:
                    results['stats']['text_documents'] += 1
                elif ext in self.image_formats:
                    results['stats']['images'] += 1
                elif ext in self.video_formats:
                    results['stats']['videos'] += 1

                results['stats']['total_pii_found'] += result.get('pii_count', 0)

            except Exception as e:
                results['failed'].append({
                    'file': file_path,
                    'error': str(e)
                })

        if progress_callback:
            progress_callback("Complete", 1.0)

        return results

    def process_file_list(
        self,
        file_paths: List[str],
        output_dir: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a list of files (mixed types).

        Args:
            file_paths: List of file paths to process
            output_dir: Output directory path
            progress_callback: Optional callback(filename, progress) for tracking

        Returns:
            Dictionary with processing results and statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'total_files': len(file_paths),
            'processed': [],
            'failed': [],
            'stats': {
                'text_documents': 0,
                'images': 0,
                'videos': 0,
                'total_pii_found': 0
            }
        }

        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(os.path.basename(file_path), i / len(file_paths))

                result = self._process_single_file(file_path, output_dir)
                results['processed'].append(result)

                # Update stats
                ext = Path(file_path).suffix.lower()
                if ext in self.text_formats:
                    results['stats']['text_documents'] += 1
                elif ext in self.image_formats:
                    results['stats']['images'] += 1
                elif ext in self.video_formats:
                    results['stats']['videos'] += 1

                results['stats']['total_pii_found'] += result.get('pii_count', 0)

            except Exception as e:
                results['failed'].append({
                    'file': file_path,
                    'error': str(e)
                })

        if progress_callback:
            progress_callback("Complete", 1.0)

        return results

    def _collect_files(
        self,
        directory: str,
        recursive: bool,
        file_types: Optional[List[str]]
    ) -> List[str]:
        """Collect all files to process from directory."""
        files = []
        all_supported = self.text_formats | self.image_formats | self.video_formats

        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    ext = Path(file_path).suffix.lower()

                    if file_types:
                        if ext in file_types:
                            files.append(file_path)
                    elif ext in all_supported:
                        files.append(file_path)
        else:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    ext = Path(file_path).suffix.lower()

                    if file_types:
                        if ext in file_types:
                            files.append(file_path)
                    elif ext in all_supported:
                        files.append(file_path)

        return sorted(files)

    def _process_single_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single file and return results."""
        filename = os.path.basename(file_path)
        ext = Path(file_path).suffix.lower()

        result = {
            'filename': filename,
            'type': ext,
            'pii_count': 0,
            'output_path': None
        }

        # Text documents
        if ext in self.text_formats:
            loader = self.loaders.get(ext)
            text = loader.load(file_path)

            # Detect PII
            findings = self.detector.detect(text)
            result['pii_count'] = len(findings)

            # Redact
            policy = {f['entity_type']: 'block' for f in findings}
            redacted_text = self.text_redactor.redact_text(text, findings, policy)

            # Save output
            output_path = os.path.join(output_dir, f"redacted_{filename}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(redacted_text)

            result['output_path'] = output_path
            result['findings'] = findings

        # Images
        elif ext in self.image_formats:
            loader = ImageLoader()
            text = loader.load(file_path)

            # Detect PII in text
            findings = self.detector.detect(text)
            result['pii_count'] = len(findings)

            # Redact image
            output_path = os.path.join(output_dir, f"redacted_{filename}")
            self.image_redactor.redact_image(file_path, findings, output_path)

            result['output_path'] = output_path
            result['findings'] = findings

        # Videos
        elif ext in self.video_formats:
            output_path = os.path.join(output_dir, f"redacted_{filename}")
            self.video_redactor.redact_faces(file_path, output_path)

            result['output_path'] = output_path
            result['type'] = 'video'

        return result
