from .base import BaseLoader
import pdfplumber
from typing import Dict, Any
from PIL import Image
import pytesseract
from app.core.config import Config
from pdf2image import convert_from_path
import tempfile
import os

class PDFLoader(BaseLoader):
    def __init__(self):
        # Set tesseract cmd from config for OCR fallback
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

    def load(self, file_path: str, **kwargs) -> str:
        """
        Load text from PDF. Supports both digital and scanned PDFs.
        Falls back to OCR if text extraction yields minimal content.

        Args:
            file_path: Path to PDF file
            force_ocr: If True, skip text extraction and use OCR directly

        Returns:
            Extracted text content
        """
        force_ocr = kwargs.get('force_ocr', False)
        text = ""

        if not force_ocr:
            # Try standard text extraction first
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Check if extraction was successful (threshold: >50 chars)
            if len(text.strip()) > 50:
                return text

        # Fallback to OCR for scanned PDFs
        return self._extract_with_ocr(file_path)

    def _extract_with_ocr(self, file_path: str) -> str:
        """Extract text using OCR from PDF images."""
        text = ""
        try:
            # Convert PDF to images
            images = convert_from_path(file_path)

            # OCR each page
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    text += f"--- Page {i+1} ---\n{page_text}\n"
        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {e}")

        return text

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}
            metadata['num_pages'] = len(pdf.pages)
            return metadata
