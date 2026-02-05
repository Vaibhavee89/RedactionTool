# Extensibility Framework - Getting Started Guide

## Quick Start (5 minutes)

### 1. Enable Plugins

```python
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

# Enable plugins
detector = EnhancedEnsembleDetector(enable_plugins=True)

# Detect with plugins
text = "Send Bitcoin to: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
results = detector.detect(text)

print(f"Detected {len(results)} entities")
for entity in results:
    print(f"- {entity['entity_type']}: {entity['text']}")
```

### 2. Discover Plugins

```python
from app.extensions.registry.plugin_registry import get_plugin_registry

registry = get_plugin_registry()

# Auto-discover plugins
discovered = registry.discover_plugins("plugins/detectors")
print(f"Discovered {len(discovered)} plugins: {discovered}")

# Get stats
stats = registry.get_stats()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Enabled: {stats['enabled_plugins']}")
```

### 3. Use Language Packs

```python
from app.extensions.registry.language_registry import get_language_registry

registry = get_language_registry()

# Discover language packs
discovered = registry.discover_language_packs("plugins/languages")
print(f"Discovered languages: {discovered}")

# Detect French PII
detector = EnhancedEnsembleDetector(enable_plugins=True)
french_text = "Mon numéro de sécurité sociale est 1 89 05 49 588 157 80"
results = detector.detect(french_text, language='fr')
```

---

## Built-in Plugins

### Cryptocurrency Detector
Detects wallet addresses for:
- Bitcoin (BTC)
- Ethereum (ETH)
- Litecoin (LTC)
- Ripple (XRP)
- Bitcoin Cash (BCH)
- Cardano (ADA)
- Dogecoin (DOGE)

### Medical Codes Detector
Detects medical classification codes:
- ICD-10 codes
- CPT codes
- NDC codes
- LOINC codes
- HCPCS codes

### Custom Regex Detector
User-configurable patterns via JSON:
- Edit `plugins/detectors/custom_regex_detector/custom_patterns.json`
- Add your own patterns without coding

---

## Built-in Language Packs

### French (fr)
- INSEE (Social Security)
- French IBAN
- French phone numbers
- SIRET/SIREN (Company IDs)

### German (de)
- German ID cards
- Social Security numbers
- Tax IDs
- German IBAN
- German phone numbers

### Arabic (ar)
- National IDs (Saudi, UAE, Egypt)
- Arabic phone numbers
- Arabic IBAN
- Arabic script support

---

## Creating Your First Plugin

### Step 1: Create Plugin Directory

```bash
mkdir -p plugins/detectors/my_detector
cd plugins/detectors/my_detector
touch plugin.py __init__.py
```

### Step 2: Implement Plugin

```python
# plugins/detectors/my_detector/plugin.py

from app.extensions.interfaces.detector_plugin import (
    DetectorPlugin,
    PluginMetadata,
    DetectedEntity
)
import re

class MyDetectorPlugin(DetectorPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="my_detector",
            version="1.0.0",
            description="My custom detector",
            supported_entity_types=["CUSTOM_ID"],
            priority=4
        )

    def detect(self, text, language='en', entity_types=None, context=None):
        entities = []

        # Your detection logic
        pattern = r'\bCUST-\d{6}\b'
        for match in re.finditer(pattern, text):
            entities.append(DetectedEntity(
                entity_type="CUSTOM_ID",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                source="my_detector"
            ))

        return entities

    def validate(self):
        return {"valid": True, "errors": [], "warnings": []}

def register_plugin():
    return MyDetectorPlugin()
```

### Step 3: Test Your Plugin

```python
from app.extensions.registry.plugin_registry import get_plugin_registry

registry = get_plugin_registry()

# Discover your plugin
discovered = registry.discover_plugins("plugins/detectors")
print(f"Discovered: {discovered}")

# Use enhanced detector
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

detector = EnhancedEnsembleDetector(enable_plugins=True)
text = "Customer CUST-123456 placed an order"
results = detector.detect(text)

for entity in results:
    if entity['entity_type'] == 'CUSTOM_ID':
        print(f"Found custom ID: {entity['text']}")
```

---

## Creating a Language Pack

### Step 1: Create Language Pack Directory

```bash
mkdir -p plugins/languages/es  # Spanish
cd plugins/languages/es
touch language.py __init__.py
```

### Step 2: Implement Language Pack

```python
# plugins/languages/es/language.py

from app.extensions.interfaces.language_pack import (
    LanguagePack,
    LanguagePackMetadata,
    RedactionPolicy,
    Script
)

class SpanishLanguagePack(LanguagePack):
    def get_metadata(self):
        return LanguagePackMetadata(
            language_code="es",
            language_name="Spanish",
            script=Script.LATIN,
            supported_entity_types=["DNI_ES", "PHONE_ES", "NIE_ES"],
            description="Spanish language PII detection"
        )

    def get_regex_patterns(self):
        return {
            "DNI_ES": r'\b\d{8}[A-Z]\b',  # Spanish DNI
            "PHONE_ES": r'\b[6789]\d{8}\b',  # Spanish mobile
            "NIE_ES": r'\b[XYZ]\d{7}[A-Z]\b'  # Spanish NIE
        }

    def get_redaction_policy(self):
        return RedactionPolicy(
            full_redaction=["DNI_ES", "NIE_ES"],
            partial_redaction={"PHONE_ES": 4},
            preserve_format=["PHONE_ES"]
        )

    def validate(self):
        return {"valid": True, "errors": [], "warnings": []}

def register_language_pack():
    return SpanishLanguagePack()
```

### Step 3: Use Your Language Pack

```python
from app.extensions.registry.language_registry import get_language_registry

registry = get_language_registry()

# Discover language pack
discovered = registry.discover_language_packs("plugins/languages")
print(f"Discovered languages: {discovered}")

# Get Spanish patterns
patterns = registry.get_regex_patterns("es")
print(f"Spanish patterns: {list(patterns.keys())}")

# Detect Spanish PII
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

detector = EnhancedEnsembleDetector(enable_plugins=True)
text = "Mi DNI es 12345678Z y mi teléfono es 612345678"
results = detector.detect(text, language='es')
```

---

## Using LLM Providers (Advanced)

### Setup OpenAI

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"
```

```python
from app.extensions.llm_providers.openai_provider import create_provider
from app.extensions.registry.llm_registry import get_llm_registry

# Create and register provider
provider = create_provider(api_key="your-key", model="gpt-4")
registry = get_llm_registry()
registry.register(provider, set_as_default=True)

# Use with enhanced detector
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

detector = EnhancedEnsembleDetector(
    enable_plugins=True,
    enable_llm=True,
    llm_provider="openai"
)

text = "Complex document with subtle PII..."
results = detector.detect(text)
```

### Setup Ollama (Local, Privacy-First)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama2:13b

# Run Ollama
ollama serve
```

```python
from app.extensions.llm_providers.ollama_provider import create_provider
from app.extensions.registry.llm_registry import get_llm_registry

# Create and register provider (no API key needed!)
provider = create_provider(model="llama2:13b")
registry = get_llm_registry()
registry.register(provider, set_as_default=True)

# Use with enhanced detector (100% local, no API costs)
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

detector = EnhancedEnsembleDetector(
    enable_plugins=True,
    enable_llm=True,
    llm_provider="ollama"
)

text = "Sensitive document..."
results = detector.detect(text)
```

---

## Using the REST API

### Discover Extensions

```bash
curl -X POST http://localhost:8000/extensions/discover \
  -H "Content-Type: application/json" \
  -d '{
    "plugins_dir": "plugins/detectors",
    "languages_dir": "plugins/languages"
  }'
```

### List Plugins

```bash
curl http://localhost:8000/extensions/plugins
```

### Enable/Disable Plugin

```bash
# Enable plugin
curl -X POST http://localhost:8000/extensions/plugins/crypto_detector/enable

# Disable plugin
curl -X POST http://localhost:8000/extensions/plugins/crypto_detector/disable
```

### Get Extension Stats

```bash
curl http://localhost:8000/extensions/stats
```

### Configure LLM Provider

```bash
curl -X POST http://localhost:8000/extensions/llm/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "api_key": "your-key",
    "model": "gpt-4",
    "set_as_default": true
  }'
```

### Detect with LLM

```bash
curl -X POST http://localhost:8000/extensions/llm/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Doe, SSN: 123-45-6789",
    "language": "en"
  }'
```

---

## Configuration

### Environment Variables

```bash
# Enable plugins
ENABLE_PLUGINS=true
PLUGINS_DIR=plugins/detectors

# Enable language packs
ENABLE_LANGUAGE_PACKS=true
LANGUAGES_DIR=plugins/languages

# Enable LLM
ENABLE_LLM=false
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key

# LLM caching
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL=3600
LLM_RATE_LIMIT=10
```

### YAML Configuration

Edit `config/extensions.yaml`:

```yaml
plugins:
  enabled: true
  directory: "plugins/detectors"
  enabled_plugins:
    - crypto_detector
    - medical_codes_detector
    - my_custom_detector

llm:
  enabled: false
  default_provider: "openai"
  cache_enabled: true
  rate_limit: 10
```

---

## Troubleshooting

### Plugin Not Found

```python
# Check discovery
from app.extensions.registry.plugin_registry import get_plugin_registry
registry = get_plugin_registry()
discovered = registry.discover_plugins("plugins/detectors")
print(f"Discovered: {discovered}")

# Check if plugin registered
all_plugins = registry.get_all_plugins()
print(f"Registered plugins: {list(all_plugins.keys())}")
```

### Plugin Not Detecting

```python
# Check if plugin is enabled
from app.extensions.registry.plugin_registry import get_plugin_registry
registry = get_plugin_registry()

is_enabled = registry.is_enabled("my_detector")
print(f"Plugin enabled: {is_enabled}")

# Enable if needed
if not is_enabled:
    registry.enable_plugin("my_detector")
```

### LLM Rate Limit

```python
# Check rate limit status
from app.extensions.registry.llm_registry import get_llm_registry
registry = get_llm_registry()

can_call = registry.check_rate_limit("openai")
if not can_call:
    wait_time = registry.get_wait_time("openai")
    print(f"Rate limited. Wait {wait_time:.1f}s")
```

---

## Best Practices

### 1. Start Simple
- Use built-in plugins first
- Test with small datasets
- Enable features incrementally

### 2. Plugin Development
- Follow naming conventions (lowercase, underscores)
- Provide comprehensive metadata
- Implement validation
- Handle errors gracefully

### 3. LLM Usage
- Cache aggressively (reduce costs)
- Set appropriate rate limits
- Monitor token usage
- Use Ollama for development/privacy

### 4. Performance
- Profile plugin execution
- Set appropriate timeouts
- Monitor memory usage
- Use async where possible

### 5. Security
- Validate all inputs
- Sanitize file uploads
- Use API keys securely
- Review plugins before deployment

---

## Next Steps

1. **Explore Examples**: Check `demo_extensibility.py`
2. **Read Documentation**: See `docs/` directory
3. **Create Custom Plugin**: Start with simple regex patterns
4. **Join Community**: Contribute your plugins!

---

## Support

- **Documentation**: See `docs/` directory
- **Examples**: Check `demo_extensibility.py`
- **Issues**: Report at GitHub
- **API Docs**: Visit `/docs` endpoint

---

## Quick Reference

### Common Commands

```python
# Discover plugins
registry.discover_plugins("plugins/detectors")

# Enable/disable plugin
registry.enable_plugin("plugin_name")
registry.disable_plugin("plugin_name")

# Get plugin stats
stats = registry.get_stats()

# Use enhanced detector
detector = EnhancedEnsembleDetector(enable_plugins=True)
results = detector.detect(text)
```

### File Locations

- **Plugins**: `plugins/detectors/`
- **Language Packs**: `plugins/languages/`
- **Config**: `config/extensions.yaml`
- **Environment**: `.env`
- **Docs**: `docs/`

---

Happy Extending! 🚀
