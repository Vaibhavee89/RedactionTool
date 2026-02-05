# Policy-Based Redaction System Guide

## Overview

The Policy-Based Redaction System provides enterprise-grade, configurable PII redaction with built-in compliance policies and advanced redaction strategies.

## Features

### ✅ Implemented Features

1. **YAML Policy Configuration**
   - Define rules for each entity type
   - Set confidence thresholds
   - Configure redaction actions
   - Global policy settings

2. **Pre-built Compliance Policies**
   - `india_finance.yaml` - RBI and IT Act compliance
   - `gdpr_basic.yaml` - GDPR Article 4 and 9 compliance
   - `hipaa_like.yaml` - HIPAA Safe Harbor method

3. **Advanced Redaction Strategies**
   - **Block**: Full redaction with block characters (█)
   - **Mask**: Partial masking (show first/last N characters)
   - **Partial Mask**: Custom masking with format preservation
   - **Label**: Replace with entity type label
   - **Hash**: Cryptographic hashing (SHA-256, MD5, SHA-1)
   - **Tokenize**: Replace with unique reversible tokens
   - **Allow**: No redaction

4. **Per-Entity Controls**
   - Different strategies per entity type
   - Confidence threshold per entity
   - Custom patterns and formats

5. **Policy Management**
   - Load from YAML files
   - Create custom policies programmatically
   - Policy validation
   - Save custom policies

## Quick Start

### Basic Usage

```python
from app.services.redaction.policy_manager import PolicyManager
from app.services.redaction.enhanced_redactor import EnhancedRedactor

# Initialize
policy_manager = PolicyManager()
redactor = EnhancedRedactor(policy_manager)

# List available policies
policies = policy_manager.list_policies()
print(f"Available policies: {policies}")

# Redact with a policy
text = "Name: John Doe, PAN: ABCDE1234F, Phone: 9876543210"
findings = [
    {"entity_type": "PERSON", "start": 6, "end": 14, "confidence": 0.85},
    {"entity_type": "PAN", "start": 21, "end": 31, "confidence": 0.95},
    {"entity_type": "PHONE", "start": 40, "end": 50, "confidence": 0.90},
]

redacted = redactor.redact_text(
    text,
    findings,
    policy="India Finance Compliance"
)
print(redacted)
```

### Using Pre-built Policies

#### India Finance Policy

Designed for Indian financial documents with RBI compliance:

```python
# Redact financial data
result = redactor.redact_with_metadata(
    text,
    findings,
    policy="India Finance Compliance"
)

print(result['redacted_text'])
print(f"Redacted {result['redaction_count']} entities")
print(f"Actions used: {result['by_action']}")
```

**Key features:**
- PAN: Full redaction
- Aadhaar: Show last 4 digits (as per regulations)
- Credit Card: Show last 4 digits
- Phone: Mask with last 4 visible
- Bank Account: Mask with last 4 visible
- IFSC: Show bank code and branch
- Names: Show first character only
- Email: Mask username, keep domain

#### GDPR Basic Policy

For European personal data protection:

```python
result = redactor.redact_text(
    text,
    findings,
    policy="GDPR Basic Compliance"
)
```

**Key features:**
- Names: Tokenized for audit trails
- Email: Masked local part
- IP Address: Hashed for pseudonymization
- Credit Card: PCI-DSS + GDPR compliant masking
- Special categories (Article 9): Full redaction
- Organizations: Allowed

#### HIPAA-Like Policy

For healthcare data (Safe Harbor method):

```python
result = redactor.redact_text(
    text,
    findings,
    policy="HIPAA-Like Healthcare Compliance"
)
```

**Key features:**
- All 18 HIPAA identifiers: Full redaction
- Dates: Labeled (keep year if needed)
- Medical Record Numbers: Tokenized
- Maximum privacy by default

## Creating Custom Policies

### Method 1: Programmatically

```python
policy_manager = PolicyManager()

custom_rules = {
    'PAN': {
        'action': 'block',
        'min_confidence': 0.9
    },
    'PHONE': {
        'action': 'mask',
        'show_last': 4,
        'mask_char': 'X',
        'min_confidence': 0.7
    },
    'EMAIL': {
        'action': 'hash',
        'algorithm': 'sha256',
        'truncate': 10,
        'prefix': 'EMAIL_',
        'min_confidence': 0.8
    },
    'PERSON': {
        'action': 'partial_mask',
        'show_first': 1,
        'show_last': 1,
        'mask_char': '*',
        'min_confidence': 0.7
    }
}

policy = policy_manager.create_custom_policy(
    name="My Custom Policy",
    rules=custom_rules,
    description="Custom policy for my use case",
    global_config={'min_confidence': 0.6}
)

# Save for reuse
policy_manager.save_policy(policy)
```

### Method 2: YAML File

Create `my_policy.yaml`:

```yaml
name: My Custom Policy
description: Custom redaction policy for specific requirements
version: "1.0"

global:
  min_confidence: 0.6
  preserve_format: true

rules:
  PAN:
    action: block
    min_confidence: 0.9
    description: "Full redaction of PAN"

  PHONE:
    action: mask
    show_last: 4
    mask_char: "X"
    min_confidence: 0.7

  EMAIL:
    action: partial_mask
    preserve_format: true
    show_first: 2
    mask_char: "*"
    min_confidence: 0.8

  PERSON:
    action: partial_mask
    show_first: 1
    show_last: 1
    mask_char: "*"
    min_confidence: 0.7

  CREDIT_CARD:
    action: partial_mask
    pattern: "XXXX-XXXX-XXXX-1234"
    min_confidence: 0.9

  "*":
    action: block
    min_confidence: 0.5
```

Load and use:

```python
policy = policy_manager.load_policy_from_file('my_policy.yaml')
redactor.redact_text(text, findings, policy="My Custom Policy")
```

## Advanced Redaction Strategies

### 1. Block (Full Redaction)

```python
rule = {
    'action': 'block',
    'char': '█',              # Character to use
    'preserve_length': True   # Keep original length
}
```

**Example:**
- Original: `ABCDE1234F`
- Redacted: `██████████`

### 2. Mask (Partial Visibility)

```python
rule = {
    'action': 'mask',
    'show_first': 2,     # Show first 2 characters
    'show_last': 4,      # Show last 4 characters
    'mask_char': '*'     # Masking character
}
```

**Example:**
- Original: `rajesh@example.com`
- Redacted: `ra*******m.com`

### 3. Partial Mask (Format Preservation)

```python
rule = {
    'action': 'partial_mask',
    'pattern': 'XXXX-XXXX-XXXX-1234',  # Explicit pattern
    'preserve_format': True             # Keep dashes, spaces
}
```

**Examples:**
- Credit Card: `4532-1234-5678-9010` → `XXXX-XXXX-XXXX-9010`
- Aadhaar: `1234 5678 9012` → `XXXX XXXX 9012`
- Phone: `+91-9876543210` → `+XX-XXXXXX3210`

### 4. Hash (Pseudonymization)

```python
rule = {
    'action': 'hash',
    'algorithm': 'sha256',  # md5, sha1, sha256
    'salt': 'my_secret',    # Optional salt
    'prefix': 'HASH_',      # Prefix for readability
    'truncate': 8           # Show first 8 chars
}
```

**Example:**
- Original: `user@example.com`
- Redacted: `HASH_b4c9a289`

### 5. Tokenize (Reversible)

```python
rule = {
    'action': 'tokenize',
    'prefix': 'TOKEN_',
    'preserve_mapping': True  # Allow de-tokenization
}
```

**Example:**
- Original: `John Doe`
- Redacted: `TOKEN_PERSON_0001`
- Same value always gets same token

### 6. Label (Semantic)

```python
rule = {
    'action': 'label',
    'format': '[{entity_type}]'  # Template
}
```

**Example:**
- Original: `John Doe`
- Redacted: `[PERSON]`

### 7. Allow (No Redaction)

```python
rule = {
    'action': 'allow'
}
```

Use for entity types that should not be redacted.

## Per-Entity Control Examples

### Financial Documents

```python
rules = {
    'PAN': {
        'action': 'block'  # Full redaction
    },
    'AADHAAR': {
        'action': 'partial_mask',
        'pattern': 'XXXX-XXXX-1234'  # Show last 4
    },
    'PHONE': {
        'action': 'mask',
        'show_last': 4  # Show last 4 digits
    },
    'CREDIT_CARD': {
        'action': 'partial_mask',
        'pattern': 'XXXX-XXXX-XXXX-****'
    },
    'BANK_ACCOUNT': {
        'action': 'mask',
        'show_last': 4,
        'mask_char': 'X'
    }
}
```

### Healthcare Records

```python
rules = {
    'PERSON': {
        'action': 'tokenize',  # Reversible for research
        'prefix': 'PT_'
    },
    'SSN': {
        'action': 'block'  # Full redaction
    },
    'MEDICAL_RECORD': {
        'action': 'tokenize',
        'prefix': 'MRN_'
    },
    'DATE': {
        'action': 'label',  # Replace with [DATE]
        'format': '[DATE_REDACTED]'
    }
}
```

### Marketing Data (GDPR)

```python
rules = {
    'EMAIL': {
        'action': 'hash',  # Pseudonymize
        'algorithm': 'sha256',
        'truncate': 10
    },
    'IP_ADDRESS': {
        'action': 'hash',  # Pseudonymize
        'prefix': 'IP_'
    },
    'PERSON': {
        'action': 'tokenize',  # For analytics
        'preserve_mapping': True
    },
    'ORGANIZATION': {
        'action': 'allow'  # Keep company names
    }
}
```

## Confidence Thresholds

Control redaction based on detection confidence:

```python
rules = {
    'PAN': {
        'action': 'block',
        'min_confidence': 0.9  # Only redact if 90%+ confident
    },
    'PERSON': {
        'action': 'mask',
        'min_confidence': 0.7  # Lower threshold for names
    }
}

global_config = {
    'min_confidence': 0.6  # Global minimum
}
```

**How it works:**
- Entity confidence must meet both rule-specific AND global thresholds
- Higher thresholds = fewer false positives, more false negatives
- Lower thresholds = catch more PII, but more false positives

**Recommended thresholds:**
- High-risk PII (SSN, PAN, Credit Card): 0.85-0.95
- Medium-risk (Phone, Email): 0.70-0.85
- Low-risk (Names, Organizations): 0.60-0.75

## Policy Validation

Validate policies before use:

```python
policy = policy_manager.get_policy("My Custom Policy")
validation = policy_manager.validate_policy(policy)

if not validation['valid']:
    print("Policy errors:")
    for error in validation['errors']:
        print(f"  - {error}")
```

**Validation checks:**
- Valid action types
- Confidence values between 0.0 and 1.0
- Required fields present
- Rule syntax correctness

## Integration with Existing System

### With Hindi Pipeline

```python
from app.services.hindi_pipeline import HindiPIIRedactionPipeline
from app.services.redaction.policy_manager import PolicyManager
from app.services.redaction.enhanced_redactor import EnhancedRedactor

# Create Hindi pipeline
pipeline = HindiPIIRedactionPipeline()

# Detect PII
result = pipeline.detector.detect(hindi_text, language='hi')

# Apply policy-based redaction
policy_manager = PolicyManager()
redactor = EnhancedRedactor(policy_manager)

redacted_text = redactor.redact_text(
    hindi_text,
    result,
    policy="India Finance Compliance"
)
```

### With Ensemble Detector

```python
from app.services.pii.ensemble_detector import EnsembleDetector
from app.services.redaction.enhanced_redactor import EnhancedRedactor

detector = EnsembleDetector()
findings = detector.detect(text)

redactor = EnhancedRedactor()
result = redactor.redact_with_metadata(
    text,
    findings,
    policy="GDPR Basic Compliance"
)
```

## Best Practices

### 1. Policy Selection

- **Financial services**: Use `india_finance.yaml` as base
- **Healthcare**: Use `hipaa_like.yaml` as base
- **European operations**: Use `gdpr_basic.yaml` as base
- **Multi-jurisdiction**: Create custom policy combining requirements

### 2. Confidence Tuning

Start with higher thresholds and gradually lower:

```python
# Start conservative
global_config = {'min_confidence': 0.8}

# Monitor false negatives
# Adjust per entity type
rules = {
    'PAN': {'min_confidence': 0.9},      # Keep high
    'PERSON': {'min_confidence': 0.7},   # Lower for names
}
```

### 3. Testing Policies

Always test policies with representative data:

```python
# Test with sample data
test_cases = [
    "PAN: ABCDE1234F",
    "Phone: +91-9876543210",
    "Email: user@example.com"
]

for test in test_cases:
    # Detect entities
    findings = detector.detect(test)

    # Apply policy
    redacted = redactor.redact_text(test, findings, policy="My Policy")

    print(f"Original: {test}")
    print(f"Redacted: {redacted}")
    print()
```

### 4. Audit Trails

Use tokenization for reversibility:

```python
rule = {
    'action': 'tokenize',
    'preserve_mapping': True
}

# Get token mapping
token_map = redactor.get_token_mapping()
# Save securely for de-tokenization if needed
```

### 5. Multi-stage Redaction

Apply different policies at different stages:

```python
# Stage 1: Detect with low threshold
findings_all = detector.detect(text, min_confidence=0.5)

# Stage 2: High-risk PII with strict policy
high_risk = [f for f in findings_all
             if f['entity_type'] in ['PAN', 'SSN', 'CREDIT_CARD']]
text = redactor.redact_text(text, high_risk, policy="Strict Policy")

# Stage 3: Low-risk PII with lenient policy
low_risk = [f for f in findings_all
            if f['entity_type'] not in ['PAN', 'SSN', 'CREDIT_CARD']]
text = redactor.redact_text(text, low_risk, policy="Lenient Policy")
```

## Troubleshooting

### Issue: Too much/too little redaction

**Solution:** Adjust confidence thresholds

```python
# Too much redaction - increase threshold
global_config = {'min_confidence': 0.8}

# Too little - decrease threshold
global_config = {'min_confidence': 0.5}
```

### Issue: Wrong redaction method

**Solution:** Check policy rules for entity type

```python
policy = policy_manager.get_policy("My Policy")
rule = policy.get_rule('PAN')
print(f"PAN action: {rule['action']}")
```

### Issue: Policy not loading

**Solution:** Validate YAML syntax

```bash
# Check YAML syntax
python3 -c "import yaml; yaml.safe_load(open('policy.yaml'))"
```

### Issue: Format not preserved

**Solution:** Enable format preservation

```python
rule = {
    'action': 'partial_mask',
    'preserve_format': True  # Keep spaces, dashes, etc.
}
```

## API Reference

### PolicyManager

```python
PolicyManager(policy_dir: Optional[str] = None)
```

**Methods:**
- `load_policy_from_file(file_path: str) -> RedactionPolicy`
- `load_policy_from_dict(policy_data: Dict) -> RedactionPolicy`
- `set_policy(policy_name: str)`
- `get_policy(policy_name: Optional[str]) -> RedactionPolicy`
- `list_policies() -> List[str]`
- `validate_policy(policy: RedactionPolicy) -> Dict`
- `create_custom_policy(...) -> RedactionPolicy`
- `save_policy(policy: RedactionPolicy, file_path: Optional[str])`

### EnhancedRedactor

```python
EnhancedRedactor(policy_manager: Optional[PolicyManager] = None)
```

**Methods:**
- `redact_text(text: str, findings: List[Dict], policy: Optional[str], custom_rules: Optional[Dict]) -> str`
- `redact_with_metadata(text: str, findings: List[Dict], policy: Optional[str]) -> Dict`
- `get_token_mapping() -> Dict[str, str]`
- `clear_token_mapping()`

### RedactionPolicy

```python
RedactionPolicy(policy_data: Dict[str, Any])
```

**Methods:**
- `get_rule(entity_type: str) -> Dict[str, Any]`
- `should_redact(entity_type: str, confidence: float) -> bool`

## Examples

See `test_policy_redaction.py` for comprehensive examples of:
- Loading and using pre-built policies
- Creating custom policies
- Per-entity control demonstrations
- Policy validation
- Advanced redaction strategies

## Support

For issues or questions:
1. Check this guide
2. Review test examples in `test_policy_redaction.py`
3. Validate policy configuration
4. Test with sample data

## License

Part of RedactionTool project.
