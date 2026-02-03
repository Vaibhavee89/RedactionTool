from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLoader(ABC):
    """Abstract base class for all file loaders."""

    @abstractmethod
    def load(self, file_path: str, **kwargs) -> str:
        """
        Load content from a file.
        
        Args:
            file_path: Path to the file to load.
            **kwargs: Additional arguments specific to the loader.
            
        Returns:
            Extracted text content as a string.
        """
        pass
    
    @abstractmethod
    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Load metadata from a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Dictionary containing metadata.
        """
        pass
