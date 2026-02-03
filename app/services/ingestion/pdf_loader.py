from .base import BaseLoader
import pdfplumber
from typing import Dict, Any

class PDFLoader(BaseLoader):
    def load(self, file_path: str, **kwargs) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        with pdfplumber.open(file_path) as pdf:
            return pdf.metadata
