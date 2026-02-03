from .base import BaseLoader
import pytesseract
from PIL import Image
from typing import Dict, Any
from app.core.config import Config

class ImageLoader(BaseLoader):
    def __init__(self):
        # Set tesseract cmd from config
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

    def load(self, file_path: str, **kwargs) -> str:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        image = Image.open(file_path)
        return {
            "format": image.format,
            "size": image.size,
            "mode": image.mode
        }
