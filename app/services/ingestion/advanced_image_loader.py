"""
Advanced image loader with OCR preprocessing and layout analysis.
"""

from .base import BaseLoader
from PIL import Image
from typing import Dict, Any, Optional, List
from app.services.ocr.ocr_engine import OCREngine
from app.services.ocr.image_preprocessor import ImagePreprocessor
from app.services.ocr.layout_analyzer import LayoutAnalyzer
import numpy as np
import cv2


class AdvancedImageLoader(BaseLoader):
    """
    Advanced image loader with:
    - Multiple OCR engines (Tesseract, PaddleOCR)
    - Image preprocessing (de-skewing, denoising, contrast enhancement)
    - Layout-aware extraction (paragraphs, tables)
    - Multi-language support (English + Indian languages)
    """

    def __init__(
        self,
        ocr_engine: str = 'tesseract',
        languages: List[str] = None,
        preprocess: bool = True,
        layout_analysis: bool = False
    ):
        """
        Initialize advanced image loader.

        Args:
            ocr_engine: OCR engine to use ('tesseract' or 'paddle')
            languages: List of language codes (e.g., ['eng', 'hin'])
            preprocess: Enable image preprocessing
            layout_analysis: Enable layout-aware extraction
        """
        self.ocr_engine = OCREngine(
            engine=ocr_engine,
            languages=languages or ['eng'],
            preprocess=preprocess
        )
        self.preprocessor = ImagePreprocessor() if preprocess else None
        self.layout_analyzer = LayoutAnalyzer() if layout_analysis else None
        self.layout_analysis_enabled = layout_analysis

    def load(self, file_path: str, **kwargs) -> str:
        """
        Load and extract text from image.

        Args:
            file_path: Path to image file
            **kwargs: Additional options
                - preprocess: Override preprocessing setting
                - deskew: Enable deskewing
                - denoise: Enable denoising
                - enhance_contrast: Enable contrast enhancement
                - layout_aware: Enable layout analysis

        Returns:
            Extracted text
        """
        preprocess = kwargs.get('preprocess', True)
        layout_aware = kwargs.get('layout_aware', self.layout_analysis_enabled)

        if layout_aware and self.layout_analyzer:
            # Use layout-aware extraction
            result = self.layout_analyzer.analyze_layout(file_path)
            return result['full_text']
        else:
            # Standard OCR extraction
            return self.ocr_engine.extract_text(file_path, preprocess=preprocess)

    def load_with_preprocessing(
        self,
        file_path: str,
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        binarize: bool = True
    ) -> str:
        """
        Load image with custom preprocessing options.

        Args:
            file_path: Path to image
            deskew: Enable deskewing
            denoise: Enable denoising
            enhance_contrast: Enable contrast enhancement
            binarize: Enable binarization

        Returns:
            Extracted text
        """
        if not self.preprocessor:
            self.preprocessor = ImagePreprocessor()

        # Load image
        image = cv2.imread(file_path)

        # Preprocess
        processed = self.preprocessor.preprocess(
            image,
            deskew=deskew,
            denoise=denoise,
            enhance_contrast=enhance_contrast,
            binarize=binarize
        )

        # Convert to PIL Image
        pil_image = Image.fromarray(processed)

        # Extract text
        return self.ocr_engine.extract_text(pil_image, preprocess=False)

    def load_with_layout(self, file_path: str) -> Dict[str, Any]:
        """
        Load image with layout analysis.

        Args:
            file_path: Path to image

        Returns:
            Dictionary with structured layout data
        """
        if not self.layout_analyzer:
            self.layout_analyzer = LayoutAnalyzer()

        return self.layout_analyzer.analyze_layout(file_path)

    def load_multi_language(
        self,
        file_path: str,
        languages: List[str] = None
    ) -> Dict[str, str]:
        """
        Extract text with multiple languages.

        Args:
            file_path: Path to image
            languages: List of language codes to try

        Returns:
            Dictionary with results per language
        """
        return self.ocr_engine.multi_language_ocr(file_path, languages)

    def get_ocr_data(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed OCR data with bounding boxes.

        Args:
            file_path: Path to image

        Returns:
            Dictionary with detailed OCR information
        """
        return self.ocr_engine.extract_with_details(file_path)

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract only tables from image.

        Args:
            file_path: Path to image

        Returns:
            List of table dictionaries
        """
        if not self.layout_analyzer:
            self.layout_analyzer = LayoutAnalyzer()

        return self.layout_analyzer.extract_tables(file_path)

    def extract_paragraphs(self, file_path: str) -> List[str]:
        """
        Extract only paragraphs from image.

        Args:
            file_path: Path to image

        Returns:
            List of paragraph texts
        """
        if not self.layout_analyzer:
            self.layout_analyzer = LayoutAnalyzer()

        return self.layout_analyzer.extract_paragraphs(file_path)

    def preprocess_and_save(
        self,
        file_path: str,
        output_path: str,
        **preprocessing_options
    ):
        """
        Preprocess image and save result.

        Args:
            file_path: Input image path
            output_path: Output path for preprocessed image
            **preprocessing_options: Preprocessing parameters
        """
        if not self.preprocessor:
            self.preprocessor = ImagePreprocessor()

        image = cv2.imread(file_path)
        processed = self.preprocessor.preprocess(image, **preprocessing_options)

        cv2.imwrite(output_path, processed)

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Load image metadata.

        Args:
            file_path: Path to image

        Returns:
            Dictionary with metadata
        """
        image = Image.open(file_path)

        # Detect language from extracted text (sample)
        try:
            sample_text = self.ocr_engine.extract_text(file_path, preprocess=False)
            detected_lang = self.ocr_engine.detect_language(sample_text[:200])
        except:
            detected_lang = 'unknown'

        return {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "ocr_engine": self.ocr_engine.engine,
            "languages": self.ocr_engine.languages,
            "detected_language": detected_lang,
            "preprocessing_enabled": self.preprocessor is not None,
            "layout_analysis_enabled": self.layout_analysis_enabled
        }

    @staticmethod
    def get_supported_languages(engine: str = 'tesseract') -> List[str]:
        """
        Get list of supported languages.

        Args:
            engine: OCR engine ('tesseract' or 'paddle')

        Returns:
            List of language codes
        """
        return OCREngine.get_supported_languages(engine)
