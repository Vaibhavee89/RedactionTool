# Extensibility Framework - Testing Guide

## Test Structure

```
tests/
├── extensions/
│   ├── test_detector_plugin.py       # Plugin interface tests
│   ├── test_language_pack.py         # Language pack interface tests
│   ├── test_llm_provider.py          # LLM provider interface tests
│   ├── test_plugin_registry.py       # Plugin registry tests
│   ├── test_language_registry.py     # Language registry tests
│   ├── test_llm_registry.py          # LLM registry tests
│   ├── test_plugin_validator.py      # Validator tests
│   └── test_cache_manager.py         # Cache manager tests
├── integration/
│   ├── test_enhanced_ensemble_detector.py  # Enhanced detector tests
│   ├── test_plugin_execution.py            # Plugin execution tests
│   ├── test_conflict_resolution.py         # Conflict resolution tests
│   └── test_llm_detection.py               # LLM detection tests
├── plugins/
│   ├── test_crypto_detector.py       # Crypto plugin tests
│   ├── test_medical_codes_detector.py  # Medical codes tests
│   └── test_custom_regex_detector.py  # Custom regex tests
├── languages/
│   ├── test_french_pack.py           # French pack tests
│   ├── test_german_pack.py           # German pack tests
│   └── test_arabic_pack.py           # Arabic pack tests
├── llm/
│   ├── test_openai_provider.py       # OpenAI tests
│   ├── test_anthropic_provider.py    # Anthropic tests
│   └── test_ollama_provider.py       # Ollama tests
├── api/
│   └── test_extensions_endpoints.py  # API tests
└── e2e/
    ├── test_plugin_workflow.py       # End-to-end plugin workflow
    ├── test_language_workflow.py     # End-to-end language workflow
    └── test_llm_workflow.py          # End-to-end LLM workflow
```

---

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test Suites
```bash
# Unit tests
pytest tests/extensions/

# Integration tests
pytest tests/integration/

# Plugin tests
pytest tests/plugins/

# API tests
pytest tests/api/

# E2E tests
pytest tests/e2e/
```

### With Coverage
```bash
pytest --cov=app/extensions --cov-report=html
```

---

## Unit Test Examples

### Plugin Registry Test

```python
# tests/extensions/test_plugin_registry.py

import pytest
from app.extensions.registry.plugin_registry import PluginRegistry
from app.extensions.interfaces.detector_plugin import DetectorPlugin, PluginMetadata

class MockPlugin(DetectorPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="mock_plugin",
            version="1.0.0",
            supported_entity_types=["MOCK"],
            priority=3
        )

    def detect(self, text, language='en', entity_types=None, context=None):
        return []

    def validate(self):
        return {"valid": True, "errors": [], "warnings": []}

def test_plugin_registration():
    registry = PluginRegistry()
    plugin = MockPlugin()

    # Register plugin
    name = registry.register(plugin)
    assert name == "mock_plugin"

    # Check if registered
    assert registry.get_plugin("mock_plugin") is not None
    assert registry.is_enabled("mock_plugin")

def test_plugin_enable_disable():
    registry = PluginRegistry()
    plugin = MockPlugin()
    registry.register(plugin)

    # Disable plugin
    registry.disable_plugin("mock_plugin")
    assert not registry.is_enabled("mock_plugin")

    # Enable plugin
    registry.enable_plugin("mock_plugin")
    assert registry.is_enabled("mock_plugin")

def test_plugin_discovery():
    registry = PluginRegistry()

    # Discover plugins
    discovered = registry.discover_plugins("plugins/detectors")

    # Should find at least 3 built-in plugins
    assert len(discovered) >= 3
    assert "crypto_detector" in discovered
```

### Enhanced Detector Test

```python
# tests/integration/test_enhanced_ensemble_detector.py

import pytest
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

def test_plugin_detection():
    detector = EnhancedEnsembleDetector(
        use_ner=False,
        use_regex=False,
        use_presidio=False,
        enable_plugins=True
    )

    # Test crypto detection
    text = "Bitcoin address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    results = detector.detect(text)

    assert len(results) > 0
    assert any(r['entity_type'] == 'CRYPTO_BTC' for r in results)

def test_backward_compatibility():
    # Existing code should work unchanged
    detector = EnhancedEnsembleDetector()

    text = "Email: john@example.com"
    results = detector.detect(text)

    assert len(results) > 0

def test_conflict_resolution():
    detector = EnhancedEnsembleDetector(
        use_regex=True,
        enable_plugins=True
    )

    # Text that might match both regex and plugin
    text = "Test detection"
    results = detector.detect(text)

    # Should resolve conflicts without duplicates
    positions = [(r['start'], r['end']) for r in results]
    assert len(positions) == len(set(positions))
```

### Crypto Plugin Test

```python
# tests/plugins/test_crypto_detector.py

import pytest
from plugins.detectors.crypto_detector.plugin import CryptoDetectorPlugin

def test_bitcoin_detection():
    plugin = CryptoDetectorPlugin()
    plugin.initialize()

    text = "Send to: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    entities = plugin.detect(text)

    assert len(entities) == 1
    assert entities[0].entity_type == "CRYPTO_BTC"
    assert entities[0].confidence > 0.8

def test_ethereum_detection():
    plugin = CryptoDetectorPlugin()
    plugin.initialize()

    text = "ETH address: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
    entities = plugin.detect(text)

    assert len(entities) == 1
    assert entities[0].entity_type == "CRYPTO_ETH"

def test_multiple_cryptocurrencies():
    plugin = CryptoDetectorPlugin()
    plugin.initialize()

    text = """
    BTC: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
    ETH: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
    LTC: LMXn6vT3cqXqJQQaCQSqQ5QL9qJbZzKqQw
    """
    entities = plugin.detect(text)

    assert len(entities) == 3
    types = [e.entity_type for e in entities]
    assert "CRYPTO_BTC" in types
    assert "CRYPTO_ETH" in types
    assert "CRYPTO_LTC" in types
```

### Language Pack Test

```python
# tests/languages/test_french_pack.py

import pytest
from plugins.languages.fr.language import FrenchLanguagePack

def test_french_patterns():
    pack = FrenchLanguagePack()
    patterns = pack.get_regex_patterns()

    assert "INSEE" in patterns
    assert "PHONE_FR" in patterns
    assert "IBAN_FR" in patterns

def test_insee_detection():
    pack = FrenchLanguagePack()
    pack.initialize()

    # Test INSEE pattern
    import re
    insee_pattern = pack.get_regex_patterns()["INSEE"]
    text = "Mon numéro INSEE est 1 89 05 49 588 157 80"

    matches = list(re.finditer(insee_pattern, text))
    assert len(matches) == 1

def test_insee_validation():
    pack = FrenchLanguagePack()

    # Valid INSEE
    assert pack._validate_insee("1 89 05 49 588 157 80")

    # Invalid INSEE (wrong format)
    assert not pack._validate_insee("0 00 00 00 000 000 00")
```

---

## Integration Test Examples

### Plugin Workflow

```python
# tests/e2e/test_plugin_workflow.py

import pytest
from app.extensions.registry.plugin_registry import get_plugin_registry
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

def test_complete_plugin_workflow():
    # Step 1: Discover plugins
    registry = get_plugin_registry()
    discovered = registry.discover_plugins("plugins/detectors")
    assert len(discovered) > 0

    # Step 2: Enable specific plugin
    registry.enable_plugin("crypto_detector")
    assert registry.is_enabled("crypto_detector")

    # Step 3: Use enhanced detector
    detector = EnhancedEnsembleDetector(enable_plugins=True)

    # Step 4: Detect with plugin
    text = "Bitcoin: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    results = detector.detect(text)

    # Step 5: Verify results
    assert len(results) > 0
    crypto_entities = [r for r in results if r['entity_type'].startswith('CRYPTO_')]
    assert len(crypto_entities) > 0

    # Step 6: Disable plugin
    registry.disable_plugin("crypto_detector")
    assert not registry.is_enabled("crypto_detector")
```

---

## API Test Examples

```python
# tests/api/test_extensions_endpoints.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_list_plugins():
    response = client.get("/extensions/plugins")
    assert response.status_code == 200
    plugins = response.json()
    assert isinstance(plugins, list)

def test_plugin_details():
    response = client.get("/extensions/plugins/crypto_detector")
    assert response.status_code == 200
    plugin = response.json()
    assert plugin["name"] == "crypto_detector"
    assert "version" in plugin

def test_enable_disable_plugin():
    # Enable
    response = client.post("/extensions/plugins/crypto_detector/enable")
    assert response.status_code == 200

    # Disable
    response = client.post("/extensions/plugins/crypto_detector/disable")
    assert response.status_code == 200

def test_discovery():
    response = client.post("/extensions/discover", json={
        "plugins_dir": "plugins/detectors",
        "languages_dir": "plugins/languages"
    })
    assert response.status_code == 200
    result = response.json()
    assert "plugins_discovered" in result
    assert "languages_discovered" in result

def test_extension_stats():
    response = client.get("/extensions/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "total_plugins" in stats
    assert "enabled_plugins" in stats
```

---

## Performance Tests

```python
# tests/performance/test_plugin_performance.py

import pytest
import time
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

def test_plugin_overhead():
    detector_without = EnhancedEnsembleDetector(enable_plugins=False)
    detector_with = EnhancedEnsembleDetector(enable_plugins=True)

    text = "Test text for performance" * 100

    # Without plugins
    start = time.time()
    detector_without.detect(text)
    time_without = time.time() - start

    # With plugins
    start = time.time()
    detector_with.detect(text)
    time_with = time.time() - start

    # Plugin overhead should be reasonable
    overhead = time_with - time_without
    assert overhead < 0.05  # Less than 50ms overhead

def test_cache_effectiveness():
    from app.extensions.utils.cache_manager import get_cache_manager

    cache_manager = get_cache_manager()
    cache_manager.clear()

    # First call (cache miss)
    @cache_manager.cached
    def expensive_operation(text):
        time.sleep(0.1)  # Simulate expensive operation
        return len(text)

    start = time.time()
    result1 = expensive_operation("test")
    time1 = time.time() - start

    # Second call (cache hit)
    start = time.time()
    result2 = expensive_operation("test")
    time2 = time.time() - start

    # Cache hit should be much faster
    assert time2 < time1 / 10
    assert result1 == result2

    # Check cache stats
    stats = cache_manager.get_stats()
    assert stats['hits'] >= 1
```

---

## Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=app/extensions
    --cov-report=html
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    api: API tests
    e2e: End-to-end tests
    slow: Slow tests
```

### conftest.py

```python
# tests/conftest.py

import pytest
from app.extensions.registry.plugin_registry import get_plugin_registry
from app.extensions.registry.language_registry import get_language_registry
from app.extensions.registry.llm_registry import get_llm_registry

@pytest.fixture(autouse=True)
def reset_registries():
    """Reset all registries before each test"""
    get_plugin_registry().clear()
    get_language_registry().clear()
    get_llm_registry().clear()
    yield

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    John Doe
    Email: john@example.com
    Phone: +1-555-123-4567
    SSN: 123-45-6789
    Bitcoin: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
    """
```

---

## Coverage Goals

- **Overall**: 90%+
- **Core Interfaces**: 95%+
- **Registries**: 95%+
- **Plugins**: 85%+
- **API**: 90%+

---

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=app/extensions --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Next Steps

1. **Implement Unit Tests**: Start with interfaces and registries
2. **Add Integration Tests**: Test plugin execution and conflict resolution
3. **Create API Tests**: Test all endpoints
4. **Add E2E Tests**: Test complete workflows
5. **Performance Tests**: Benchmark and optimize
6. **Security Tests**: Validate input sanitization

---

For actual test implementation, see the test structure above and follow the examples provided.
