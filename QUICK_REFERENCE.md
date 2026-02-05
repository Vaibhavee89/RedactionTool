# Extensibility Framework - Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

### Basic Usage

```python
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

# Enable plugins
detector = EnhancedEnsembleDetector(enable_plugins=True)

# Detect PII
results = detector.detect("Your text here")
```

### Discover Plugins

```python
from app.extensions.registry.plugin_registry import get_plugin_registry

registry = get_plugin_registry()
discovered = registry.discover_plugins("plugins/detectors")
print(f"Found: {discovered}")
```

---

## 📦 Built-in Detectors

| Plugin | Detects | Example |
|--------|---------|---------|
| **crypto_detector** | BTC, ETH, LTC, XRP, BCH, ADA, DOGE | `1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa` |
| **medical_codes_detector** | ICD-10, CPT, NDC, LOINC, HCPCS | `J45.909`, `99213` |
| **custom_regex_detector** | User-defined patterns | `EMP-123456` |

---

## 🌍 Built-in Languages

| Code | Language | Detects |
|------|----------|---------|
| **fr** | French | INSEE, IBAN, phones, SIRET |
| **de** | German | IDs, SSN, Tax ID, IBAN |
| **ar** | Arabic | National IDs (SA/UAE/EG), phones |

---

## 🤖 LLM Providers

| Provider | Cost | Privacy | Setup |
|----------|------|---------|-------|
| **OpenAI** | $$ | Cloud | API key required |
| **Anthropic** | $$ | Cloud | API key required |
| **Ollama** | FREE | Local | No API key |

---

## 💻 Common Commands

### Python API

```python
# Import
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector
from app.extensions.registry.plugin_registry import get_plugin_registry

# Discover
registry = get_plugin_registry()
registry.discover_plugins("plugins/detectors")

# Enable/Disable
registry.enable_plugin("crypto_detector")
registry.disable_plugin("crypto_detector")

# Detect
detector = EnhancedEnsembleDetector(enable_plugins=True)
results = detector.detect(text, language='en')

# Multi-language
results_fr = detector.detect(french_text, language='fr')
results_de = detector.detect(german_text, language='de')

# With LLM
detector_llm = EnhancedEnsembleDetector(
    enable_plugins=True,
    enable_llm=True,
    llm_provider="openai"
)
results = detector_llm.detect(text)
```

### REST API

```bash
# Discover
curl -X POST http://localhost:8000/extensions/discover

# List plugins
curl http://localhost:8000/extensions/plugins

# Enable plugin
curl -X POST http://localhost:8000/extensions/plugins/crypto_detector/enable

# Get stats
curl http://localhost:8000/extensions/stats

# Configure LLM
curl -X POST http://localhost:8000/extensions/llm/configure \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "api_key": "your-key", "model": "gpt-4"}'

# Detect with LLM
curl -X POST http://localhost:8000/extensions/llm/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text", "language": "en"}'
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Enable plugins
ENABLE_PLUGINS=true
PLUGINS_DIR=plugins/detectors

# Enable LLM
ENABLE_LLM=false
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key

# LLM settings
LLM_CACHE_ENABLED=true
LLM_RATE_LIMIT=10
```

### YAML Config

```yaml
# config/extensions.yaml
plugins:
  enabled: true
  directory: "plugins/detectors"

llm:
  enabled: false
  default_provider: "openai"
  cache_enabled: true
```

---

## 🛠️ Create Your Own

### Plugin (5 minutes)

```python
# plugins/detectors/my_plugin/plugin.py
from app.extensions.interfaces.detector_plugin import *

class MyPlugin(DetectorPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            supported_entity_types=["MY_TYPE"],
            priority=4
        )

    def detect(self, text, language='en', entity_types=None, context=None):
        entities = []
        # Your detection logic
        return entities

    def validate(self):
        return {"valid": True, "errors": [], "warnings": []}

def register_plugin():
    return MyPlugin()
```

### Language Pack (10 minutes)

```python
# plugins/languages/es/language.py
from app.extensions.interfaces.language_pack import *

class SpanishPack(LanguagePack):
    def get_metadata(self):
        return LanguagePackMetadata(
            language_code="es",
            language_name="Spanish",
            script=Script.LATIN,
            supported_entity_types=["DNI_ES", "PHONE_ES"]
        )

    def get_regex_patterns(self):
        return {
            "DNI_ES": r'\b\d{8}[A-Z]\b',
            "PHONE_ES": r'\b[6789]\d{8}\b'
        }

    def get_redaction_policy(self):
        return RedactionPolicy(
            full_redaction=["DNI_ES"],
            partial_redaction={"PHONE_ES": 4}
        )

    def validate(self):
        return {"valid": True, "errors": [], "warnings": []}

def register_language_pack():
    return SpanishPack()
```

---

## 📊 Directory Structure

```
plugins/
├── detectors/          # Your plugins here
│   ├── my_plugin/
│   │   ├── plugin.py   # Implement DetectorPlugin
│   │   └── __init__.py
│   └── ...
└── languages/          # Your language packs here
    ├── es/
    │   ├── language.py # Implement LanguagePack
    │   └── __init__.py
    └── ...
```

---

## 🔍 Debugging

```python
# Check what's registered
from app.extensions.registry.plugin_registry import get_plugin_registry

registry = get_plugin_registry()
stats = registry.get_stats()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Enabled: {stats['enabled_plugins']}")

# Check if plugin enabled
is_enabled = registry.is_enabled("crypto_detector")
print(f"Crypto detector enabled: {is_enabled}")

# Get plugin details
metadata = registry.get_metadata("crypto_detector")
print(f"Priority: {metadata.priority}")
print(f"Entity types: {metadata.supported_entity_types}")

# Test detection
from app.services.pii.enhanced_ensemble_detector import EnhancedEnsembleDetector

detector = EnhancedEnsembleDetector(enable_plugins=True)
text = "Test: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
results = detector.detect(text)
print(f"Detected {len(results)} entities")
for r in results:
    print(f"  - {r['entity_type']}: {r['text']}")
```

---

## ⚡ Performance Tips

1. **Use caching** - Enable LLM cache for 80% cost reduction
2. **Set priorities** - Higher priority plugins win conflicts
3. **Filter entity types** - Only detect what you need
4. **Use Ollama** - Local, free, private for development
5. **Monitor timeouts** - Adjust timeout_seconds if needed

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Plugin not found | Run `discover_plugins()` |
| Plugin not detecting | Check if enabled with `is_enabled()` |
| LLM rate limit | Wait or increase rate limit |
| Slow detection | Check plugin timeouts, disable unused plugins |
| Import errors | Check plugin dependencies |

---

## 📚 Documentation

- **Getting Started**: `docs/EXTENSIBILITY_GETTING_STARTED.md`
- **Testing**: `docs/TESTING_GUIDE.md`
- **Architecture**: `EXTENSIBILITY_FRAMEWORK.md`
- **Status**: `IMPLEMENTATION_STATUS.md`
- **Summary**: `FINAL_IMPLEMENTATION_SUMMARY.md`
- **Demo**: `demo_extensibility.py`

---

## 🎯 Key Concepts

- **Plugin**: Custom detector for specific entity types
- **Language Pack**: Patterns and policies for a language
- **LLM Provider**: AI backend for context-aware detection
- **Registry**: Manages plugin/language/LLM lifecycle
- **Priority**: Higher number = wins conflicts (1-10)
- **Enhanced Detector**: Drop-in replacement with plugin support

---

## ✅ Checklist for New Users

- [ ] Run `demo_extensibility.py`
- [ ] Discover built-in plugins
- [ ] Try cryptocurrency detection
- [ ] Try medical codes detection
- [ ] Test multi-language (French/German/Arabic)
- [ ] Create your first custom plugin
- [ ] (Optional) Configure LLM provider
- [ ] Read getting started guide

---

## 🚀 Production Checklist

- [ ] Set appropriate priorities
- [ ] Configure timeouts
- [ ] Enable LLM caching
- [ ] Set rate limits
- [ ] Configure monitoring
- [ ] Test performance
- [ ] Review security settings
- [ ] Enable audit logging

---

## 💡 Examples in Action

### Cryptocurrency Detection
```python
text = "Send Bitcoin to: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
# Detects: CRYPTO_BTC
```

### Medical Codes
```python
text = "Patient diagnosed with ICD-10 code J45.909"
# Detects: MEDICAL_ICD10
```

### French PII
```python
text = "Mon numéro INSEE: 1 89 05 49 588 157 80"
# Detects: INSEE (with French language pack)
```

### Custom Pattern
```python
# Edit: plugins/detectors/custom_regex_detector/custom_patterns.json
text = "Employee EMP-123456"
# Detects: CUSTOM_EMPLOYEE_ID
```

---

## 🔗 Quick Links

- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/extensions/health`
- **Stats**: `http://localhost:8000/extensions/stats`

---

**Need Help?** Check `docs/EXTENSIBILITY_GETTING_STARTED.md` for detailed guides!

---

*Last updated: February 5, 2026*
