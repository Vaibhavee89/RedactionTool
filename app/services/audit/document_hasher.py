"""
Document Hasher - Generate secure hashed IDs for documents.
No raw PII is stored in audit logs.
"""

import hashlib
import secrets
from typing import Optional, Dict
from pathlib import Path


class DocumentHasher:
    """
    Generates secure hashed document IDs to prevent PII leakage in audit logs.

    Uses SHA-256 with optional salt for enhanced security.
    """

    def __init__(self, use_salt: bool = True, salt_length: int = 32):
        """
        Initialize DocumentHasher.

        Args:
            use_salt: Whether to use salt for hashing
            salt_length: Length of salt in bytes
        """
        self.use_salt = use_salt
        self.salt_length = salt_length
        self._salt_cache: Dict[str, str] = {}  # Cache salts per session

    def hash_document_id(
        self,
        document_path: str,
        content: Optional[str] = None,
        salt: Optional[str] = None
    ) -> str:
        """
        Generate a hashed document ID.

        Args:
            document_path: Path to the document (used as primary identifier)
            content: Optional document content (for additional entropy)
            salt: Optional custom salt (if not provided, generates new one)

        Returns:
            Hashed document ID (hex string)

        Example:
            >>> hasher = DocumentHasher()
            >>> doc_id = hasher.hash_document_id("/path/to/sensitive.pdf")
            >>> print(doc_id)
            'a3f5b8c2d1e4...' (64 character hex string)
        """
        # Create hash object
        hash_obj = hashlib.sha256()

        # Add document path (normalized)
        normalized_path = str(Path(document_path).resolve())
        hash_obj.update(normalized_path.encode('utf-8'))

        # Add content hash if provided (for extra entropy)
        if content:
            content_hash = hashlib.sha256(content.encode('utf-8')).digest()
            hash_obj.update(content_hash)

        # Add salt if enabled
        if self.use_salt:
            if salt is None:
                # Use cached salt for this document or generate new one
                if normalized_path in self._salt_cache:
                    salt = self._salt_cache[normalized_path]
                else:
                    salt = secrets.token_hex(self.salt_length)
                    self._salt_cache[normalized_path] = salt

            hash_obj.update(salt.encode('utf-8'))

        return hash_obj.hexdigest()

    def hash_filename_only(self, filename: str) -> str:
        """
        Generate a hash from filename only (no path).
        Useful for consistent hashing across different environments.

        Args:
            filename: Name of the file

        Returns:
            Hashed filename (hex string)
        """
        hash_obj = hashlib.sha256()
        hash_obj.update(filename.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Shorter hash for filename-only

    def mask_filepath(self, filepath: str, show_last_n: int = 1) -> str:
        """
        Mask filepath to show only filename (or parts of it).
        Prevents full path leakage in logs.

        Args:
            filepath: Full file path
            show_last_n: Number of path components to show (default: 1 = filename only)

        Returns:
            Masked path string

        Example:
            >>> hasher.mask_filepath("/home/user/sensitive/document.pdf")
            '****/document.pdf'
        """
        path = Path(filepath)
        parts = path.parts

        if len(parts) <= show_last_n:
            return str(path)

        visible_parts = parts[-show_last_n:]
        masked = "****/" + "/".join(visible_parts)
        return masked

    def get_document_metadata(
        self,
        document_path: str,
        include_hash: bool = True,
        include_masked_path: bool = True
    ) -> Dict[str, str]:
        """
        Get privacy-safe document metadata for audit logs.

        Args:
            document_path: Path to document
            include_hash: Include hashed document ID
            include_masked_path: Include masked filepath

        Returns:
            Dictionary with safe metadata
        """
        path = Path(document_path)
        metadata = {
            "filename": path.name,
            "extension": path.suffix.lower()
        }

        if include_hash:
            metadata["document_id"] = self.hash_document_id(document_path)

        if include_masked_path:
            metadata["masked_path"] = self.mask_filepath(document_path)

        return metadata

    def hash_entity_text(self, entity_text: str) -> str:
        """
        Hash detected entity text (for tracking without storing PII).

        Args:
            entity_text: The detected PII text

        Returns:
            SHA-256 hash of the entity text

        Example:
            >>> hasher.hash_entity_text("john.doe@example.com")
            'e3b0c44298fc1c14...'
        """
        return hashlib.sha256(entity_text.encode('utf-8')).hexdigest()

    def clear_cache(self):
        """Clear the salt cache (useful between sessions)."""
        self._salt_cache.clear()


# Singleton instance for convenience
_hasher_instance = None


def get_hasher(use_salt: bool = True) -> DocumentHasher:
    """
    Get a singleton DocumentHasher instance.

    Args:
        use_salt: Whether to use salt for hashing

    Returns:
        DocumentHasher instance
    """
    global _hasher_instance
    if _hasher_instance is None:
        _hasher_instance = DocumentHasher(use_salt=use_salt)
    return _hasher_instance
