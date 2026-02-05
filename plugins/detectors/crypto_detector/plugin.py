"""
Cryptocurrency Address Detector Plugin

Detects cryptocurrency wallet addresses for Bitcoin, Ethereum, and other popular chains.
"""

import re
from typing import List, Dict, Any, Optional

from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    PluginMetadata,
    PluginType,
    DetectedEntity
)


class CryptoDetectorPlugin(DetectorPlugin):
    """
    Detector for cryptocurrency wallet addresses.

    Supports:
    - Bitcoin (BTC) - Legacy, SegWit, Bech32
    - Ethereum (ETH)
    - Litecoin (LTC)
    - Ripple (XRP)
    - Bitcoin Cash (BCH)
    - Cardano (ADA)
    - Dogecoin (DOGE)
    """

    # Cryptocurrency patterns
    CRYPTO_PATTERNS = {
        'CRYPTO_BTC': {
            'pattern': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|bc1[a-z0-9]{39,87}\b',
            'name': 'Bitcoin',
            'confidence': 0.85
        },
        'CRYPTO_ETH': {
            'pattern': r'\b0x[a-fA-F0-9]{40}\b',
            'name': 'Ethereum',
            'confidence': 0.90
        },
        'CRYPTO_LTC': {
            'pattern': r'\b[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}\b',
            'name': 'Litecoin',
            'confidence': 0.80
        },
        'CRYPTO_XRP': {
            'pattern': r'\br[0-9a-zA-Z]{24,34}\b',
            'name': 'Ripple',
            'confidence': 0.75
        },
        'CRYPTO_BCH': {
            'pattern': r'\bq[a-z0-9]{41}\b|p[a-z0-9]{41}\b',
            'name': 'Bitcoin Cash',
            'confidence': 0.80
        },
        'CRYPTO_ADA': {
            'pattern': r'\baddr1[a-z0-9]{58,}\b',
            'name': 'Cardano',
            'confidence': 0.85
        },
        'CRYPTO_DOGE': {
            'pattern': r'\bD[5-9A-HJ-NP-U][1-9A-HJ-NP-Za-km-z]{32}\b',
            'name': 'Dogecoin',
            'confidence': 0.80
        }
    }

    # Context keywords that indicate cryptocurrency
    CRYPTO_CONTEXTS = [
        'wallet', 'address', 'send', 'receive', 'transfer',
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
        'blockchain', 'coin', 'token', 'payment', 'transaction'
    ]

    def __init__(self):
        """Initialize crypto detector plugin."""
        super().__init__()
        self.compiled_patterns = {}

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="crypto_detector",
            version="1.0.0",
            description="Detects cryptocurrency wallet addresses (BTC, ETH, LTC, XRP, etc.)",
            author="RedactionTool Team",
            plugin_type=PluginType.DETECTOR,
            supported_entity_types=list(self.CRYPTO_PATTERNS.keys()),
            supported_languages=["*"],  # Language-independent
            priority=4,  # Higher than standard regex (more specific)
            dependencies=[],
            min_confidence=0.7,
            max_confidence=0.95,
            requires_network=False,
            timeout_seconds=10
        )

    def initialize(self) -> None:
        """Initialize plugin by compiling regex patterns."""
        super().initialize()

        for entity_type, config in self.CRYPTO_PATTERNS.items():
            self.compiled_patterns[entity_type] = re.compile(config['pattern'])

    def detect(
        self,
        text: str,
        language: str = 'en',
        entity_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedEntity]:
        """
        Detect cryptocurrency addresses in text.

        Args:
            text: Text to analyze
            language: Language code (not used, crypto addresses are language-independent)
            entity_types: Specific crypto types to detect (None = all)
            context: Additional context (not used)

        Returns:
            List of detected cryptocurrency addresses
        """
        if not text:
            return []

        entities = []

        # Filter patterns by requested entity types
        patterns_to_check = self.CRYPTO_PATTERNS.items()
        if entity_types:
            patterns_to_check = [
                (et, config) for et, config in patterns_to_check
                if et in entity_types
            ]

        # Detect each cryptocurrency type
        for entity_type, config in patterns_to_check:
            pattern = self.compiled_patterns[entity_type]

            for match in pattern.finditer(text):
                address = match.group(0)
                start = match.start()
                end = match.end()

                # Validate address (additional checks)
                if self._validate_address(address, entity_type):
                    # Check context for confidence boost
                    confidence = config['confidence']
                    if self._has_crypto_context(text, start, end):
                        confidence = min(confidence + 0.05, 0.95)

                    entities.append(DetectedEntity(
                        entity_type=entity_type,
                        text=address,
                        start=start,
                        end=end,
                        confidence=confidence,
                        source="crypto_detector",
                        metadata={
                            'crypto_name': config['name'],
                            'has_context': self._has_crypto_context(text, start, end)
                        }
                    ))

        return entities

    def _validate_address(self, address: str, entity_type: str) -> bool:
        """
        Additional validation for crypto addresses.

        Args:
            address: Address string
            entity_type: Type of cryptocurrency

        Returns:
            True if address passes validation
        """
        # Length checks
        if entity_type == 'CRYPTO_BTC':
            # Bitcoin addresses: 26-35 characters (legacy), 42+ (bech32)
            if address.startswith('bc1'):
                return 42 <= len(address) <= 90
            return 26 <= len(address) <= 35

        elif entity_type == 'CRYPTO_ETH':
            # Ethereum: exactly 42 characters (0x + 40 hex)
            return len(address) == 42

        elif entity_type == 'CRYPTO_LTC':
            # Litecoin: 26-35 characters
            return 26 <= len(address) <= 35

        elif entity_type == 'CRYPTO_XRP':
            # Ripple: 25-35 characters
            return 25 <= len(address) <= 35

        elif entity_type == 'CRYPTO_BCH':
            # Bitcoin Cash: 42 characters
            return len(address) == 42

        elif entity_type == 'CRYPTO_ADA':
            # Cardano: 59+ characters
            return len(address) >= 59

        elif entity_type == 'CRYPTO_DOGE':
            # Dogecoin: 34 characters
            return len(address) == 34

        return True

    def _has_crypto_context(self, text: str, start: int, end: int, window: int = 50) -> bool:
        """
        Check if address has cryptocurrency context nearby.

        Args:
            text: Full text
            start: Start position of address
            end: End position of address
            window: Context window size

        Returns:
            True if crypto context found
        """
        # Extract context before and after
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()

        # Check for crypto keywords
        return any(keyword in context for keyword in self.CRYPTO_CONTEXTS)

    def validate(self) -> Dict[str, Any]:
        """
        Validate plugin configuration.

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        # Check patterns compile
        for entity_type, config in self.CRYPTO_PATTERNS.items():
            try:
                re.compile(config['pattern'])
            except re.error as e:
                errors.append(f"Invalid pattern for {entity_type}: {str(e)}")

        # Check metadata
        try:
            metadata = self.get_metadata()
            if not metadata.supported_entity_types:
                warnings.append("No entity types defined")
        except Exception as e:
            errors.append(f"Failed to get metadata: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


def register_plugin() -> DetectorPlugin:
    """
    Register function called by plugin registry.

    Returns:
        CryptoDetectorPlugin instance
    """
    return CryptoDetectorPlugin()
