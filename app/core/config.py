import os
from pathlib import Path

class Config:
    APP_NAME = "RedactionTool Enterprise"
    VERSION = "2.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Tesseract
    # Try different common paths or environment variable
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", '/opt/homebrew/bin/tesseract')
    
    # Models
    SPACY_MODEL_EN = "en_core_web_sm"
    SPACY_MODEL_MULTI = "xx_ent_wiki_sm"

    @classmethod
    def setup_paths(cls):
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
