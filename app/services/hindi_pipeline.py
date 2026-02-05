"""
Complete pipeline for Hindi document processing and PII redaction.
"""

from app.services.ocr.ocr_engine import OCREngine
from app.services.ocr.image_preprocessor import ImagePreprocessor
from app.services.pii.hindi_ensemble_detector import HindiEnsembleDetector
from app.services.redaction.redactor import Redactor
from typing import Dict, Any, Optional
import os


class HindiPIIRedactionPipeline:
    """
    Complete pipeline for Hindi document processing:
    1. Image preprocessing (optional)
    2. OCR with Hindi support
    3. PII detection using ensemble detector
    4. Text redaction

    Supports:
    - Pure Hindi text
    - Mixed Hindi-English (Hinglish)
    - Scanned documents
    - Digital PDFs with Hindi text
    """

    def __init__(
        self,
        ocr_engine: str = 'paddle',  # 'paddle' better for Hindi, 'tesseract' also works
        preprocess: bool = True,
        use_hindi_regex: bool = True
    ):
        """
        Initialize Hindi redaction pipeline.

        Args:
            ocr_engine: OCR engine ('paddle' or 'tesseract')
            preprocess: Enable image preprocessing
            use_hindi_regex: Use Hindi-specific regex patterns
        """
        # Initialize OCR with Hindi support
        self.ocr = OCREngine(
            engine=ocr_engine,
            languages=['hi', 'en'],  # Hindi + English for mixed text
            preprocess=preprocess
        )

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor() if preprocess else None

        # Initialize Hindi ensemble detector
        self.detector = HindiEnsembleDetector(
            use_ner=True,
            use_regex=True,
            use_presidio=True,
            use_hindi_regex=use_hindi_regex,
            load_hindi=True
        )

        # Initialize redactor
        self.redactor = Redactor()

    def process_image(
        self,
        image_path: str,
        redaction_mode: str = 'block',
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process Hindi image document.

        Args:
            image_path: Path to image with Hindi text
            redaction_mode: 'block', 'mask', or 'label'
            min_confidence: Minimum confidence for PII detection

        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Processing: {image_path}")

        # Step 1: Extract text using OCR
        print("  Step 1/3: Extracting Hindi text with OCR...")
        text = self.ocr.extract_text(image_path, preprocess=True)

        if not text or len(text.strip()) < 10:
            print("  Warning: Very little text extracted from image")

        # Step 2: Detect PII
        print("  Step 2/3: Detecting Hindi PII...")
        result = self.detector.detect_multilingual(text, min_confidence=min_confidence)
        findings = result['entities']

        # Step 3: Redact text
        print("  Step 3/3: Redacting PII...")
        policy = {f['entity_type']: redaction_mode for f in findings}
        redacted_text = self.redactor.redact_text(text, findings, policy)

        print(f"  ✓ Found {len(findings)} PII entities")

        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'findings': findings,
            'pii_count': len(findings),
            'language_distribution': result['language_distribution'],
            'statistics': self._calculate_statistics(findings),
            'success': True
        }

    def process_text(
        self,
        text: str,
        language: str = 'hi',
        redaction_mode: str = 'block',
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process Hindi text directly (no OCR).

        Args:
            text: Hindi text with PII
            language: Language code ('hi' for Hindi, 'auto' for mixed)
            redaction_mode: 'block', 'mask', or 'label'
            min_confidence: Minimum confidence for PII detection

        Returns:
            Dictionary with processing results
        """
        print("Processing Hindi text...")

        # Detect PII
        print("  Step 1/2: Detecting Hindi PII...")

        if language == 'auto':
            result = self.detector.detect_multilingual(text, min_confidence=min_confidence)
            findings = result['entities']
            lang_dist = result['language_distribution']
        else:
            findings = self.detector.detect(text, language=language, min_confidence=min_confidence)
            lang_dist = None

        # Redact
        print("  Step 2/2: Redacting PII...")
        policy = {f['entity_type']: redaction_mode for f in findings}
        redacted_text = self.redactor.redact_text(text, findings, policy)

        print(f"  ✓ Found {len(findings)} PII entities")

        result = {
            'original_text': text,
            'redacted_text': redacted_text,
            'findings': findings,
            'pii_count': len(findings),
            'statistics': self._calculate_statistics(findings),
            'success': True
        }

        if lang_dist:
            result['language_distribution'] = lang_dist

        return result

    def process_batch(
        self,
        file_paths: list,
        redaction_mode: str = 'block',
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process multiple Hindi documents.

        Args:
            file_paths: List of image file paths
            redaction_mode: 'block', 'mask', or 'label'
            min_confidence: Minimum confidence for PII detection

        Returns:
            Dictionary with batch results
        """
        print(f"Processing {len(file_paths)} documents...")

        results = []
        total_pii = 0

        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] {os.path.basename(file_path)}")

            try:
                result = self.process_image(file_path, redaction_mode, min_confidence)
                results.append({
                    'file': file_path,
                    'success': True,
                    'pii_count': result['pii_count'],
                    'result': result
                })
                total_pii += result['pii_count']
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'file': file_path,
                    'success': False,
                    'error': str(e)
                })

        success_count = sum(1 for r in results if r['success'])

        return {
            'total_files': len(file_paths),
            'successful': success_count,
            'failed': len(file_paths) - success_count,
            'total_pii_found': total_pii,
            'results': results
        }

    def _calculate_statistics(self, findings: list) -> Dict[str, Any]:
        """Calculate detection statistics."""
        stats = {
            'total': len(findings),
            'by_type': {},
            'by_source': {},
            'by_language': {},
            'confidence_distribution': {
                'high': 0,    # >= 0.8
                'medium': 0,  # >= 0.5
                'low': 0      # < 0.5
            }
        }

        for finding in findings:
            # Count by type
            entity_type = finding['entity_type']
            stats['by_type'][entity_type] = stats['by_type'].get(entity_type, 0) + 1

            # Count by source
            source = finding.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1

            # Count by language
            language = finding.get('language', 'unknown')
            stats['by_language'][language] = stats['by_language'].get(language, 0) + 1

            # Confidence distribution
            conf = finding.get('confidence', 0.5)
            if conf >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif conf >= 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1

        return stats

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return ['hi', 'en', 'mixed']

    def get_supported_redaction_modes(self) -> list:
        """Get list of supported redaction modes."""
        return ['block', 'mask', 'label']


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = HindiPIIRedactionPipeline(
        ocr_engine='paddle',  # Better for Hindi
        preprocess=True,
        use_hindi_regex=True
    )

    # Example 1: Process Hindi text
    hindi_text = """
    व्यक्तिगत जानकारी:
    नाम: राजेश कुमार शर्मा
    ईमेल: rajesh.sharma@example.com
    फ़ोन नंबर: +91-9876543210
    पैन नंबर: ABCDE1234F
    आधार संख्या: 1234 5678 9012
    """

    result = pipeline.process_text(hindi_text, language='hi', redaction_mode='block')

    print("\n" + "="*60)
    print("Original Text:")
    print("="*60)
    print(result['original_text'])

    print("\n" + "="*60)
    print("Redacted Text:")
    print("="*60)
    print(result['redacted_text'])

    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Total PII found: {result['pii_count']}")
    print(f"By type: {result['statistics']['by_type']}")
    print(f"By source: {result['statistics']['by_source']}")

    # Example 2: Process Hindi image (if you have one)
    # result = pipeline.process_image('hindi_document.png', redaction_mode='block')
