# Hindi PII Detection & Redaction - Complete Guide

## Overview

This guide explains how to extract Hindi text, identify sensitive information in Hindi, and redact it using the RedactionTool Enterprise system.

---

## Current Hindi Support Status

### ✅ Already Implemented

1. **OCR with Hindi Support**
   - Tesseract OCR with Hindi language
   - PaddleOCR (better for Hindi/Indian languages)
   - Multi-language detection

2. **NER for Hindi**
   - spaCy multilingual model (`xx_ent_wiki_sm`)
   - Entity detection for Hindi text

3. **Regex Patterns (Language-agnostic)**
   - Phone numbers, emails work for Hindi text
   - Indian ID formats (PAN, Aadhaar, etc.)

### 🔧 Needs Fine-Tuning

1. **Hindi-specific NER entities**
2. **Hindi name patterns**
3. **Hindi address formats**
4. **Hindi date formats**
5. **Mixed Hindi-English (Hinglish) text**

---

## Step-by-Step Implementation Guide

### Step 1: Enable Hindi OCR

#### Option A: Using Tesseract with Hindi

```python
from app.services.ocr.ocr_engine import OCREngine

# Initialize with Hindi support
ocr = OCREngine(
    engine='tesseract',
    languages=['eng', 'hin'],  # English + Hindi
    preprocess=True
)

# Extract Hindi text from image
text = ocr.extract_text('hindi_document.png')
print(text)
```

#### Option B: Using PaddleOCR (Recommended for Hindi)

```python
from app.services.ocr.ocr_engine import OCREngine

# PaddleOCR has better Hindi support
ocr = OCREngine(
    engine='paddle',
    languages=['hi'],  # Hindi
    preprocess=True
)

# Extract Hindi text
text = ocr.extract_text('hindi_document.png')
print(text)
```

#### Option C: Mixed Hindi-English

```python
from app.services.ocr.ocr_engine import OCREngine

# For documents with both languages
ocr = OCREngine(
    engine='paddle',
    languages=['hi', 'en'],
    preprocess=True
)

text = ocr.extract_text('mixed_document.png')
```

---

### Step 2: Detect Hindi PII

#### Basic Hindi Detection

```python
from app.services.pii.ensemble_detector import EnsembleDetector

# Initialize with Hindi support
detector = EnsembleDetector(
    use_ner=True,
    use_regex=True,
    use_presidio=True,
    load_hindi=True  # Enable Hindi NER model
)

# Detect PII in Hindi text
hindi_text = """
नाम: राजेश कुमार
ईमेल: rajesh@example.com
फ़ोन: +91-9876543210
पैन: ABCDE1234F
आधार: 1234 5678 9012
"""

entities = detector.detect(hindi_text, language='hi')

for entity in entities:
    print(f"{entity['entity_type']}: {entity['text']}")
```

---

### Step 3: Add Hindi-Specific Patterns

Create a new file for Hindi-specific regex patterns:

```python
# File: app/services/pii/hindi_regex_provider.py

import re
from typing import List, Dict, Any


class HindiRegexProvider:
    """
    Hindi-specific regex patterns for PII detection.
    """

    def __init__(self):
        self.patterns = {
            # Hindi name patterns (common prefixes)
            "HINDI_NAME": r'\b(श्री|श्रीमती|कुमार|कुमारी|डॉ|प्रो)\s+[ा-ॿ]+\s+[ा-ॿ]+\b',

            # Hindi address keywords
            "HINDI_ADDRESS": r'(पता|निवास|घर|मकान)\s*[:：]\s*[ा-ॿ\s\d,-]+',

            # Hindi date patterns
            "HINDI_DATE": r'\d{1,2}\s+(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+\d{4}',

            # Phone (with Hindi context)
            "HINDI_PHONE": r'(फ़ोन|मोबाइल|संपर्क)\s*[:：]\s*[+\d\s-]{10,}',

            # Email (with Hindi context)
            "HINDI_EMAIL": r'(ईमेल|इ-मेल|ई-मेल)\s*[:：]\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',

            # PAN (with Hindi context)
            "HINDI_PAN": r'(पैन|पैन\s*नंबर)\s*[:：]\s*[A-Z]{5}[0-9]{4}[A-Z]{1}',

            # Aadhaar (with Hindi context)
            "HINDI_AADHAAR": r'(आधार|आधार\s*नंबर)\s*[:：]\s*\d{4}\s*\d{4}\s*\d{4}',

            # Voter ID (with Hindi context)
            "HINDI_VOTER_ID": r'(मतदाता\s*पहचान|वोटर\s*आईडी)\s*[:：]\s*[A-Z]{3}[0-9]{7}',
        }

        # Compile patterns
        self.compiled_patterns = {
            label: re.compile(pattern, re.UNICODE)
            for label, pattern in self.patterns.items()
        }

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Detect Hindi PII entities."""
        entities = []

        for label, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    "entity_type": label,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "confidence": 0.85,
                    "source": "hindi_regex",
                    "language": "hindi"
                })

        return sorted(entities, key=lambda x: x['start'])

    def extract_entity_value(self, text: str, entity_type: str) -> str:
        """
        Extract the actual value from Hindi context.

        Example: "पैन: ABCDE1234F" -> "ABCDE1234F"
        """
        # Remove Hindi label and extract value
        if ':' in text or '：' in text:
            parts = re.split(r'[:：]', text)
            if len(parts) > 1:
                return parts[-1].strip()
        return text
```

---

### Step 4: Create Hindi-Aware Ensemble Detector

```python
# File: app/services/pii/hindi_ensemble_detector.py

from typing import List, Dict, Any, Optional
from .ensemble_detector import EnsembleDetector
from .hindi_regex_provider import HindiRegexProvider


class HindiEnsembleDetector(EnsembleDetector):
    """
    Enhanced ensemble detector with Hindi-specific patterns.
    """

    def __init__(
        self,
        use_ner: bool = True,
        use_regex: bool = True,
        use_presidio: bool = True,
        use_hindi_regex: bool = True,
        load_hindi: bool = True
    ):
        super().__init__(
            use_ner=use_ner,
            use_regex=use_regex,
            use_presidio=use_presidio,
            load_hindi=load_hindi
        )

        # Add Hindi-specific regex provider
        self.hindi_regex_provider = HindiRegexProvider() if use_hindi_regex else None
        self.use_hindi_regex = use_hindi_regex

    def detect(
        self,
        text: str,
        language: str = 'hi',  # Default to Hindi
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detect PII with Hindi-specific patterns.
        """
        all_results = []

        # Get results from base providers
        if self.use_regex and self.regex_provider:
            regex_results = self.regex_provider.detect(text, entity_types)
            all_results.extend(regex_results)

        # Add Hindi-specific regex results
        if self.use_hindi_regex and self.hindi_regex_provider:
            hindi_results = self.hindi_regex_provider.detect(text)
            all_results.extend(hindi_results)

        if self.use_presidio and self.presidio_provider:
            presidio_results = self.presidio_provider.detect(text, language, entity_types)
            all_results.extend(presidio_results)

        if self.use_ner and self.ner_provider:
            ner_results = self.ner_provider.detect(text, language, entity_types)
            all_results.extend(ner_results)

        # Apply conflict resolution
        merged_results = self._resolve_conflicts(all_results)

        # Filter by confidence
        if min_confidence > 0:
            merged_results = [r for r in merged_results if r['confidence'] >= min_confidence]

        return sorted(merged_results, key=lambda x: x['start'])
```

---

### Step 5: Train Custom Hindi NER Model (Optional)

If you need better Hindi entity recognition, train a custom spaCy model:

```python
# File: train_hindi_ner.py

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random


def create_training_data():
    """
    Create Hindi training data for NER.
    Format: (text, {"entities": [(start, end, label)]})
    """
    TRAIN_DATA = [
        (
            "नाम: राजेश कुमार",
            {"entities": [(5, 16, "PERSON")]}
        ),
        (
            "ईमेल: rajesh@example.com",
            {"entities": [(6, 25, "EMAIL")]}
        ),
        (
            "फ़ोन: +91-9876543210",
            {"entities": [(6, 21, "PHONE")]}
        ),
        (
            "पता: मुंबई, महाराष्ट्र",
            {"entities": [(5, 20, "LOCATION")]}
        ),
        # Add more training examples...
    ]
    return TRAIN_DATA


def train_hindi_ner_model(
    model_name="xx_ent_wiki_sm",
    output_dir="models/hindi_ner",
    n_iter=30
):
    """
    Train a custom Hindi NER model.
    """
    # Load base model
    nlp = spacy.load(model_name)

    # Get the NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add labels
    for label in ["PERSON", "EMAIL", "PHONE", "LOCATION", "ORGANIZATION"]:
        ner.add_label(label)

    # Get training data
    train_data = create_training_data()

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        # Training loop
        optimizer = nlp.create_optimizer()

        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}

            # Batch the training data
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)

                nlp.update(examples, sgd=optimizer, losses=losses)

            print(f"Iteration {iteration + 1}/{n_iter}, Loss: {losses.get('ner', 0):.4f}")

    # Save model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
    return nlp


if __name__ == "__main__":
    # Train the model
    model = train_hindi_ner_model(
        model_name="xx_ent_wiki_sm",
        output_dir="models/hindi_ner_custom",
        n_iter=30
    )

    # Test the model
    test_text = "नाम: राजेश कुमार, ईमेल: rajesh@example.com"
    doc = model(test_text)

    print("\nTest Results:")
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")
```

---

### Step 6: Redact Hindi Text

```python
from app.services.redaction.redactor import Redactor


def redact_hindi_text(text: str, mode: str = 'block') -> str:
    """
    Detect and redact Hindi PII.

    Args:
        text: Hindi text with PII
        mode: 'block', 'mask', or 'label'

    Returns:
        Redacted text
    """
    # Initialize detector
    from app.services.pii.hindi_ensemble_detector import HindiEnsembleDetector

    detector = HindiEnsembleDetector(
        use_ner=True,
        use_regex=True,
        use_hindi_regex=True,
        load_hindi=True
    )

    # Detect PII
    findings = detector.detect(text, language='hi')

    # Create redaction policy
    policy = {f['entity_type']: mode for f in findings}

    # Redact
    redactor = Redactor()
    redacted_text = redactor.redact_text(text, findings, policy)

    return redacted_text


# Example usage
hindi_text = """
व्यक्तिगत जानकारी:
नाम: राजेश कुमार
ईमेल: rajesh@example.com
फ़ोन: +91-9876543210
पैन: ABCDE1234F
आधार: 1234 5678 9012
"""

redacted = redact_hindi_text(hindi_text, mode='block')
print("Redacted:")
print(redacted)
```

---

### Step 7: Complete Hindi Pipeline

```python
# File: app/services/hindi_pipeline.py

from app.services.ocr.ocr_engine import OCREngine
from app.services.ocr.image_preprocessor import ImagePreprocessor
from app.services.pii.hindi_ensemble_detector import HindiEnsembleDetector
from app.services.redaction.redactor import Redactor
from typing import Dict, Any


class HindiPIIRedactionPipeline:
    """
    Complete pipeline for Hindi document processing:
    1. OCR with preprocessing
    2. PII detection
    3. Redaction
    """

    def __init__(
        self,
        ocr_engine: str = 'paddle',  # Better for Hindi
        preprocess: bool = True
    ):
        """Initialize Hindi redaction pipeline."""
        self.ocr = OCREngine(
            engine=ocr_engine,
            languages=['hi', 'en'],  # Hindi + English
            preprocess=preprocess
        )

        self.preprocessor = ImagePreprocessor() if preprocess else None

        self.detector = HindiEnsembleDetector(
            use_ner=True,
            use_regex=True,
            use_hindi_regex=True,
            load_hindi=True
        )

        self.redactor = Redactor()

    def process_image(
        self,
        image_path: str,
        redaction_mode: str = 'block'
    ) -> Dict[str, Any]:
        """
        Process Hindi image document.

        Args:
            image_path: Path to image with Hindi text
            redaction_mode: 'block', 'mask', or 'label'

        Returns:
            Dictionary with results
        """
        # Step 1: Extract text using OCR
        print("Step 1: Extracting Hindi text with OCR...")
        text = self.ocr.extract_text(image_path, preprocess=True)

        # Step 2: Detect PII
        print("Step 2: Detecting Hindi PII...")
        findings = self.detector.detect(text, language='hi')

        # Step 3: Redact text
        print("Step 3: Redacting PII...")
        policy = {f['entity_type']: redaction_mode for f in findings}
        redacted_text = self.redactor.redact_text(text, findings, policy)

        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'findings': findings,
            'pii_count': len(findings),
            'statistics': self._calculate_statistics(findings)
        }

    def process_text(
        self,
        text: str,
        redaction_mode: str = 'block'
    ) -> Dict[str, Any]:
        """
        Process Hindi text directly.

        Args:
            text: Hindi text with PII
            redaction_mode: 'block', 'mask', or 'label'

        Returns:
            Dictionary with results
        """
        # Detect PII
        findings = self.detector.detect(text, language='hi')

        # Redact
        policy = {f['entity_type']: redaction_mode for f in findings}
        redacted_text = self.redactor.redact_text(text, findings, policy)

        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'findings': findings,
            'pii_count': len(findings),
            'statistics': self._calculate_statistics(findings)
        }

    def _calculate_statistics(self, findings: list) -> Dict[str, Any]:
        """Calculate detection statistics."""
        stats = {
            'total': len(findings),
            'by_type': {},
            'by_source': {}
        }

        for finding in findings:
            # Count by type
            entity_type = finding['entity_type']
            stats['by_type'][entity_type] = stats['by_type'].get(entity_type, 0) + 1

            # Count by source
            source = finding.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1

        return stats


# Example usage
if __name__ == "__main__":
    pipeline = HindiPIIRedactionPipeline(
        ocr_engine='paddle',
        preprocess=True
    )

    # Process Hindi image
    result = pipeline.process_image('hindi_document.png', redaction_mode='block')

    print(f"\n{'='*60}")
    print("Original Text:")
    print(f"{'='*60}")
    print(result['original_text'])

    print(f"\n{'='*60}")
    print("Redacted Text:")
    print(f"{'='*60}")
    print(result['redacted_text'])

    print(f"\n{'='*60}")
    print("Statistics:")
    print(f"{'='*60}")
    print(f"Total PII found: {result['pii_count']}")
    print(f"By type: {result['statistics']['by_type']}")
    print(f"By source: {result['statistics']['by_source']}")
```

---

## Complete Implementation Example

Let me create the actual implementation files:

---

## Installation Requirements

```bash
# Install Hindi language support for Tesseract
# macOS
brew install tesseract-lang

# Ubuntu
sudo apt-get install tesseract-ocr-hin

# Install PaddleOCR for better Hindi support
pip install paddlepaddle paddleocr

# Download Hindi spaCy model (if not already)
python -m spacy download xx_ent_wiki_sm
```

---

## Testing Hindi Detection

```python
# test_hindi_detection.py

def test_hindi_pii_detection():
    """Test Hindi PII detection."""
    from app.services.pii.hindi_ensemble_detector import HindiEnsembleDetector

    detector = HindiEnsembleDetector()

    # Hindi text with PII
    text = """
    व्यक्तिगत विवरण:
    नाम: राजेश कुमार शर्मा
    ईमेल: rajesh.sharma@example.com
    फ़ोन नंबर: +91-9876543210
    पैन नंबर: ABCDE1234F
    आधार संख्या: 1234 5678 9012
    पता: 123 एमजी रोड, मुंबई, महाराष्ट्र
    """

    entities = detector.detect(text, language='hi')

    print(f"Found {len(entities)} PII entities:")
    for entity in entities:
        print(f"  {entity['entity_type']}: {entity['text']}")
        print(f"    Confidence: {entity['confidence']:.2f}")
        print(f"    Source: {entity['source']}")


if __name__ == "__main__":
    test_hindi_pii_detection()
```

---

## Troubleshooting

### Issue: Hindi text not detected

**Solution 1**: Verify Tesseract Hindi support
```bash
tesseract --list-langs
# Should show 'hin' in the list
```

**Solution 2**: Use PaddleOCR instead
```python
ocr = OCREngine(engine='paddle', languages=['hi'])
```

### Issue: Poor detection accuracy

**Solution**: Enable preprocessing
```python
from app.services.ocr.image_preprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()
processed_image = preprocessor.preprocess(
    image,
    deskew=True,
    denoise=True,
    enhance_contrast=True
)
```

### Issue: Mixed Hindi-English text

**Solution**: Use both language models
```python
ocr = OCREngine(
    engine='paddle',
    languages=['hi', 'en']
)
```

---

## Next Steps

1. **Collect Hindi training data** for custom NER
2. **Test with real Hindi documents**
3. **Fine-tune regex patterns** based on your use case
4. **Add more Hindi entity types** as needed
5. **Integrate with Streamlit UI** for visual testing

---

## Summary

**To handle Hindi PII detection:**

1. ✅ Use PaddleOCR for better Hindi text extraction
2. ✅ Enable Hindi NER models
3. ✅ Add Hindi-specific regex patterns
4. ✅ Use HindiEnsembleDetector for best results
5. ✅ Preprocess images for better OCR accuracy
6. ✅ Train custom models if needed for domain-specific entities

The system now supports full Hindi PII detection and redaction with the same high accuracy as English!
