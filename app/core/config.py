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
    # Default to 'tesseract' (PATH) if specific path doesn't exist
    _MAC_PATH = '/opt/homebrew/bin/tesseract'
    _DEFAULT_CMD = _MAC_PATH if os.path.exists(_MAC_PATH) else 'tesseract'
    
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", _DEFAULT_CMD)
    
    # Models
    SPACY_MODEL_EN = "en_core_web_sm"
    SPACY_MODEL_MULTI = "xx_ent_wiki_sm"

    @classmethod
    def setup_paths(cls):
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
