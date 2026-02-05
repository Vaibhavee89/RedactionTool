"""
Advanced OCR Engine with multi-engine support.
Supports Tesseract and PaddleOCR with language detection.
"""

import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from app.core.config import Config
from .image_preprocessor import ImagePreprocessor
import cv2


class OCREngine:
    """
    Advanced OCR engine supporting multiple OCR backends and languages.

    Supported engines:
    - Tesseract (default)
    - PaddleOCR (for better Asian language support)

    Supported languages:
    - English
    - Hindi
    - Other Indian languages (via PaddleOCR)
    """

    def __init__(
        self,
        engine: str = 'tesseract',
        languages: List[str] = None,
        preprocess: bool = True
    ):
        """
        Initialize OCR engine.

        Args:
            engine: OCR engine to use ('tesseract' or 'paddle')
            languages: List of language codes (e.g., ['eng', 'hin'])
            preprocess: Enable automatic image preprocessing
        """
        self.engine = engine
        self.languages = languages or ['eng']
        self.preprocess_enabled = preprocess
        self.preprocessor = ImagePreprocessor() if preprocess else None

        # Set tesseract path from config
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

        # Initialize PaddleOCR if selected
        self.paddle_ocr = None
        if engine == 'paddle':
            self._init_paddle()

    def _init_paddle(self):
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR

            # Determine language
            # PaddleOCR uses different language codes
            lang_map = {
                'eng': 'en',
                'hin': 'hi',
                'hindi': 'hi',
                'english': 'en'
            }

            paddle_lang = 'en'
            if self.languages:
                first_lang = self.languages[0].lower()
                paddle_lang = lang_map.get(first_lang, 'en')

            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,  # Enable angle classification
                lang=paddle_lang
            )
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
            )

    def extract_text(
        self,
        image_input,
        preprocess: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Extract text from image.

        Args:
            image_input: Image path (str), PIL Image, or numpy array
            preprocess: Override default preprocessing setting
            **kwargs: Additional OCR parameters

        Returns:
            Extracted text
        """
        # Load and preprocess image
        image = self._load_image(image_input)

        should_preprocess = preprocess if preprocess is not None else self.preprocess_enabled

        if should_preprocess and self.preprocessor:
            if isinstance(image, np.ndarray):
                image = self.preprocessor.preprocess(image)
            else:
                image_array = np.array(image)
                image_array = self.preprocessor.preprocess(image_array)
                image = Image.fromarray(image_array)

        # Extract text based on engine
        if self.engine == 'tesseract':
            return self._extract_with_tesseract(image, **kwargs)
        elif self.engine == 'paddle':
            return self._extract_with_paddle(image, **kwargs)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")

    def extract_with_details(
        self,
        image_input,
        preprocess: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Extract text with detailed information (bounding boxes, confidence).

        Args:
            image_input: Image path, PIL Image, or numpy array
            preprocess: Override preprocessing setting

        Returns:
            Dictionary with text, bounding boxes, and confidence scores
        """
        image = self._load_image(image_input)

        should_preprocess = preprocess if preprocess is not None else self.preprocess_enabled

        if should_preprocess and self.preprocessor:
            if isinstance(image, np.ndarray):
                image = self.preprocessor.preprocess(image)
            else:
                image_array = np.array(image)
                image_array = self.preprocessor.preprocess(image_array)
                image = Image.fromarray(image_array)

        if self.engine == 'tesseract':
            return self._extract_details_tesseract(image)
        elif self.engine == 'paddle':
            return self._extract_details_paddle(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")

    def _load_image(self, image_input):
        """Load image from various input types."""
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            return image_input
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _extract_with_tesseract(self, image, **kwargs) -> str:
        """Extract text using Tesseract OCR."""
        lang_string = '+'.join(self.languages)

        config = kwargs.get('config', '')

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        text = pytesseract.image_to_string(
            image,
            lang=lang_string,
            config=config
        )

        return text

    def _extract_with_paddle(self, image, **kwargs) -> str:
        """Extract text using PaddleOCR."""
        if self.paddle_ocr is None:
            self._init_paddle()

        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Run OCR
        result = self.paddle_ocr.ocr(image, cls=True)

        # Extract text from results
        if not result or not result[0]:
            return ""

        text_lines = []
        for line in result[0]:
            if line and len(line) > 1:
                text_lines.append(line[1][0])  # line[1][0] is the text

        return '\n'.join(text_lines)

    def _extract_details_tesseract(self, image) -> Dict[str, Any]:
        """Extract detailed OCR data with Tesseract."""
        lang_string = '+'.join(self.languages)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Get detailed data
        data = pytesseract.image_to_data(
            image,
            lang=lang_string,
            output_type=pytesseract.Output.DICT
        )

        # Combine into structured format
        words = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                words.append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'bbox': {
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    },
                    'block_num': data['block_num'][i],
                    'par_num': data['par_num'][i],
                    'line_num': data['line_num'][i]
                })

        return {
            'engine': 'tesseract',
            'languages': self.languages,
            'text': pytesseract.image_to_string(image, lang=lang_string),
            'words': words,
            'raw_data': data
        }

    def _extract_details_paddle(self, image) -> Dict[str, Any]:
        """Extract detailed OCR data with PaddleOCR."""
        if self.paddle_ocr is None:
            self._init_paddle()

        if isinstance(image, Image.Image):
            image = np.array(image)

        result = self.paddle_ocr.ocr(image, cls=True)

        if not result or not result[0]:
            return {
                'engine': 'paddle',
                'languages': self.languages,
                'text': '',
                'words': []
            }

        words = []
        text_lines = []

        for line in result[0]:
            if line and len(line) > 1:
                bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]     # (text, confidence)

                text = text_info[0]
                confidence = text_info[1]

                # Convert bbox to x, y, width, height
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x = min(xs)
                y = min(ys)
                width = max(xs) - x
                height = max(ys) - y

                words.append({
                    'text': text,
                    'confidence': confidence * 100,  # Convert to percentage
                    'bbox': {
                        'left': int(x),
                        'top': int(y),
                        'width': int(width),
                        'height': int(height)
                    },
                    'bbox_points': bbox_points
                })

                text_lines.append(text)

        return {
            'engine': 'paddle',
            'languages': self.languages,
            'text': '\n'.join(text_lines),
            'words': words
        }

    def detect_language(self, text: str) -> str:
        """
        Detect the language of extracted text.

        Args:
            text: Text to analyze

        Returns:
            Detected language code
        """
        try:
            from langdetect import detect
            return detect(text)
        except:
            return 'unknown'

    def multi_language_ocr(
        self,
        image_input,
        languages: List[str] = None
    ) -> Dict[str, str]:
        """
        Perform OCR with multiple languages and return best result.

        Args:
            image_input: Image to process
            languages: List of languages to try

        Returns:
            Dictionary with results per language
        """
        if languages is None:
            languages = ['eng', 'hin']  # English and Hindi

        results = {}

        for lang in languages:
            original_langs = self.languages
            self.languages = [lang]

            try:
                text = self.extract_text(image_input, preprocess=True)
                results[lang] = text
            except Exception as e:
                results[lang] = f"Error: {e}"

            self.languages = original_langs

        return results

    @staticmethod
    def get_supported_languages(engine: str = 'tesseract') -> List[str]:
        """
        Get list of supported languages for specified engine.

        Args:
            engine: OCR engine ('tesseract' or 'paddle')

        Returns:
            List of language codes
        """
        if engine == 'tesseract':
            try:
                langs = pytesseract.get_languages()
                return langs
            except:
                return ['eng', 'hin', 'san', 'mar', 'tam', 'tel']
        elif engine == 'paddle':
            return ['en', 'ch', 'ta', 'te', 'ka', 'hi']
        else:
            return []
