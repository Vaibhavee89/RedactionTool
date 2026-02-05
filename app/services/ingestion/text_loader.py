from .base import BaseLoader
from typing import Dict, Any
import os

class TextLoader(BaseLoader):
    """Loader for plain text files (.txt)"""

    def load(self, file_path: str, **kwargs) -> str:
        """Load content from a plain text file."""
        encoding = kwargs.get('encoding', 'utf-8')
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load metadata from text file."""
        stat = os.stat(file_path)
        return {
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "encoding": "utf-8"
        }
