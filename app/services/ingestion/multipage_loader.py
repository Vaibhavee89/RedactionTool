from .base import BaseLoader
from typing import Dict, Any, List
from PIL import Image
import pytesseract
from app.core.config import Config
from pdf2image import convert_from_path
import os

class MultiPageDocumentLoader(BaseLoader):
    """
    Loader for multi-page scanned documents.
    Supports:
    - PDF files (converted to images)
    - TIFF files (multi-page)
    - Multiple separate image files
    """

    def __init__(self):
        # Set tesseract cmd from config
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

    def load(self, file_path: str, **kwargs) -> str:
        """
        Load and OCR multi-page documents.

        Args:
            file_path: Path to file or directory of images

        Returns:
            Combined text from all pages
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext in ['.tiff', '.tif']:
            return self._load_tiff(file_path)
        elif os.path.isdir(file_path):
            return self._load_image_directory(file_path)
        else:
            raise ValueError(f"Unsupported file type for multi-page loader: {ext}")

    def _load_pdf(self, file_path: str) -> str:
        """Convert PDF to images and OCR each page."""
        text = ""
        images = convert_from_path(file_path)

        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            text += f"--- Page {i+1} ---\n{page_text}\n"

        return text

    def _load_tiff(self, file_path: str) -> str:
        """Load multi-page TIFF and OCR each page."""
        text = ""
        image = Image.open(file_path)

        page_num = 1
        try:
            while True:
                page_text = pytesseract.image_to_string(image)
                text += f"--- Page {page_num} ---\n{page_text}\n"
                page_num += 1
                image.seek(page_num - 1)
        except EOFError:
            pass  # End of TIFF pages

        return text

    def _load_image_directory(self, dir_path: str) -> str:
        """Load and OCR all images in a directory."""
        text = ""
        supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

        # Get all image files, sorted by name
        image_files = sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in supported_exts
        ])

        for i, img_path in enumerate(image_files):
            image = Image.open(img_path)
            page_text = pytesseract.image_to_string(image)
            text += f"--- Page {i+1} ({os.path.basename(img_path)}) ---\n{page_text}\n"

        return text

    def get_page_images(self, file_path: str) -> List[Image.Image]:
        """
        Get list of page images for visual redaction.

        Returns:
            List of PIL Image objects
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return convert_from_path(file_path)
        elif ext in ['.tiff', '.tif']:
            images = []
            image = Image.open(file_path)
            page_num = 0
            try:
                while True:
                    images.append(image.copy())
                    page_num += 1
                    image.seek(page_num)
            except EOFError:
                pass
            return images
        elif os.path.isdir(file_path):
            supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
            image_files = sorted([
                os.path.join(file_path, f)
                for f in os.listdir(file_path)
                if os.path.splitext(f)[1].lower() in supported_exts
            ])
            return [Image.open(img) for img in image_files]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load metadata from multi-page document."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            images = convert_from_path(file_path)
            return {
                "type": "pdf",
                "num_pages": len(images),
                "page_sizes": [img.size for img in images]
            }
        elif ext in ['.tiff', '.tif']:
            image = Image.open(file_path)
            page_count = 0
            try:
                while True:
                    page_count += 1
                    image.seek(page_count)
            except EOFError:
                pass
            return {
                "type": "tiff",
                "num_pages": page_count,
                "size": image.size
            }
        elif os.path.isdir(file_path):
            supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
            image_files = [
                f for f in os.listdir(file_path)
                if os.path.splitext(f)[1].lower() in supported_exts
            ]
            return {
                "type": "image_directory",
                "num_pages": len(image_files),
                "files": sorted(image_files)
            }

        return {}
