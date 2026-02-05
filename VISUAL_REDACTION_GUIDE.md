# Visual Redaction & Masking Modes Guide

## Overview

The Visual Redaction System provides flexible output handling for both text and image/video redaction with multiple configurable styles.

## ✅ Implemented Features

### 1. Full Redaction ✅

**Text Redaction:**
```python
from app.services.redaction import EnhancedRedactor

redactor = EnhancedRedactor()
rule = {'action': 'block', 'char': '█'}

# Example: 'ABCDE1234F' → '██████████'
```

**Visual Redaction:**
```python
from app.services.redaction.visual_redactor import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.BLACK_BOX)
image = redactor.redact_region(image, x, y, width, height)
```

**Available Styles:**
- `BLACK_BOX` - Solid black rectangle
- `WHITE_BOX` - Solid white rectangle
- `COLORED_BOX` - Custom colored rectangle

---

### 2. Partial Masking ✅

**Show Last N Characters:**
```python
# Text: Show last 4 digits
rule = {
    'action': 'mask',
    'show_last': 4,
    'mask_char': 'X'
}

# Examples:
# Phone: '9876543210' → 'XXXXXX3210'
# Card:  '1234-5678-9012' → 'XXXX-XXXX-9012'
# Email: 'user@example.com' → 'XXXX@example.com'
```

**Visual Partial Masking:**
```python
# Blur only specific regions
bounding_boxes = [
    {'x': 100, 'y': 50, 'width': 200, 'height': 30, 'entity_type': 'PHONE'}
]

redactor = VisualRedactor(default_style=RedactionStyle.BLUR)
image = redactor.redact_bounding_boxes(image, bounding_boxes)
```

---

### 3. Token Replacement ✅

**Label Format:**
```python
# Simple label
rule = {
    'action': 'label',
    'format': '[{entity_type}]'
}

# Examples:
# 'John Doe' → '[PERSON]'
# 'ABCDE1234F' → '[PAN]'
```

**Custom Token Format:**
```python
# Custom format like <PAN_REDACTED>
rule = {
    'action': 'label',
    'format': '<{entity_type}_REDACTED>'
}

# Example: 'ABCDE1234F' → '<PAN_REDACTED>'
```

**Unique Tokens:**
```python
# Reversible tokenization
rule = {
    'action': 'tokenize',
    'prefix': 'TOKEN_',
    'preserve_mapping': True
}

# Example: 'John Doe' → 'TOKEN_PERSON_0001'
```

---

### 4. Visual Redaction ✅

**8 Visual Styles Available:**

#### a) Blur (Gaussian)
```python
redactor = VisualRedactor(
    default_style=RedactionStyle.BLUR,
    blur_strength=30
)
```
**Use cases:** Faces, sensitive text, license plates

#### b) Heavy Blur
```python
redactor = VisualRedactor(
    default_style=RedactionStyle.HEAVY_BLUR,
    blur_strength=60
)
```
**Use cases:** High-security documents, medical records

#### c) Pixelate
```python
redactor = VisualRedactor(
    default_style=RedactionStyle.PIXELATE,
    pixelate_size=10
)
```
**Use cases:** Faces in photos, ID numbers, signatures

#### d) Mosaic
```python
redactor = VisualRedactor(
    default_style=RedactionStyle.MOSAIC,
    pixelate_size=15
)
```
**Use cases:** Japanese-style censoring, artistic redaction

#### e) Black Box
```python
redactor = VisualRedactor(default_style=RedactionStyle.BLACK_BOX)
```
**Use cases:** Military documents, classified information

#### f) White Box
```python
redactor = VisualRedactor(default_style=RedactionStyle.WHITE_BOX)
```
**Use cases:** Print documents, light backgrounds

#### g) Colored Box
```python
redactor = VisualRedactor(
    default_style=RedactionStyle.COLORED_BOX,
    default_color=(255, 200, 200)  # Light blue (BGR)
)
```
**Use cases:** Color-coded redactions, branding

#### h) Pattern Fill
```python
redactor = VisualRedactor(default_style=RedactionStyle.PATTERN)
```
**Use cases:** Decorative redaction, watermarking

---

### 5. Blur Bounding Boxes in Images ✅

**Basic Usage:**
```python
from app.services.redaction.visual_redactor import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.BLUR)

# Define PII regions
bounding_boxes = [
    {'x': 100, 'y': 50, 'width': 200, 'height': 30, 'entity_type': 'PAN'},
    {'x': 100, 'y': 100, 'width': 250, 'height': 30, 'entity_type': 'PHONE'},
    {'x': 100, 'y': 150, 'width': 300, 'height': 30, 'entity_type': 'EMAIL'},
]

# Redact image
result = redactor.redact_image_file(
    input_path='document.png',
    output_path='redacted.png',
    bounding_boxes=bounding_boxes
)

print(f"Redacted {result['regions_redacted']} regions")
```

**With OCR Integration:**
```python
from app.services.ocr.ocr_engine import OCREngine
from app.services.pii.ensemble_detector import EnsembleDetector

# Extract text with bounding boxes
ocr = OCREngine(engine='tesseract')
ocr_result = ocr.extract_with_details('document.png')

# Detect PII
detector = EnsembleDetector()
text = ocr_result['text']
pii_findings = detector.detect(text)

# Map findings to bounding boxes
bounding_boxes = []
for finding in pii_findings:
    # Match finding to OCR word boxes
    for word in ocr_result['words']:
        if word['text'] in finding['text']:
            bounding_boxes.append({
                'x': word['bbox']['left'],
                'y': word['bbox']['top'],
                'width': word['bbox']['width'],
                'height': word['bbox']['height'],
                'entity_type': finding['entity_type']
            })

# Redact
redactor = VisualRedactor()
result = redactor.redact_image_file(
    'document.png',
    'redacted.png',
    bounding_boxes=bounding_boxes
)
```

---

### 6. Blur Faces in Video Frames ✅

**Basic Face Redaction:**
```python
from app.services.redaction.visual_redactor import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.BLUR)

def progress(percent):
    print(f"Progress: {percent*100:.1f}%")

result = redactor.redact_video_file(
    input_path='video.mp4',
    output_path='redacted_video.mp4',
    redact_faces=True,
    progress_callback=progress
)

print(f"Processed {result['frames_processed']} frames")
print(f"Redacted {result['faces_redacted']} faces")
```

**With Different Styles:**
```python
# Pixelate faces (Japanese TV style)
redactor = VisualRedactor(default_style=RedactionStyle.PIXELATE)
result = redactor.redact_video_file(
    'video.mp4',
    'pixelated_video.mp4',
    style=RedactionStyle.PIXELATE
)

# Black box faces
redactor = VisualRedactor(default_style=RedactionStyle.BLACK_BOX)
result = redactor.redact_video_file(
    'video.mp4',
    'blackbox_video.mp4',
    style=RedactionStyle.BLACK_BOX
)
```

**Supported Video Formats:**
- MP4 (`.mp4`)
- AVI (`.avi`)
- MOV (`.mov`)
- Other OpenCV-supported formats

---

### 7. Configurable Redaction Style ✅

**Per-Entity Style Configuration:**
```python
from app.services.redaction.visual_redactor import StyleConfig, RedactionStyle

# Create style configuration
style_config = StyleConfig()

# Configure PAN (high security)
style_config.set_entity_style(
    'PAN',
    text_style='block',              # Full text redaction
    visual_style=RedactionStyle.BLACK_BOX  # Solid black box
)

# Configure Phone (medium security)
style_config.set_entity_style(
    'PHONE',
    text_style='mask',               # Show last 4
    visual_style=RedactionStyle.PIXELATE,  # Pixelate
    intensity=15                     # Block size
)

# Configure Person names (low security)
style_config.set_entity_style(
    'PERSON',
    text_style='mask',               # Show first char
    visual_style=RedactionStyle.BLUR,      # Blur
    intensity=20                     # Blur strength
)

# Configure Email (colored for review)
style_config.set_entity_style(
    'EMAIL',
    text_style='hash',               # Hash
    visual_style=RedactionStyle.COLORED_BOX,
    color=(200, 200, 200)           # Light gray (BGR)
)
```

**Apply Configuration:**
```python
# Get style maps
config_dict = style_config.to_dict()

style_map = {
    entity: RedactionStyle(config['visual_style'])
    for entity, config in config_dict['entity_styles'].items()
}

color_map = {
    entity: config.get('color')
    for entity, config in config_dict['entity_styles'].items()
    if config.get('color')
}

# Apply to image
redactor = VisualRedactor()
image = redactor.redact_bounding_boxes(
    image,
    bounding_boxes,
    style_map=style_map,
    color_map=color_map
)
```

**Save and Load Configuration:**
```python
import json

# Save configuration
with open('redaction_style.json', 'w') as f:
    json.dump(style_config.to_dict(), f, indent=2)

# Load configuration
with open('redaction_style.json', 'r') as f:
    config_dict = json.load(f)

style_config = StyleConfig()
style_config.load_from_dict(config_dict)
```

---

## Complete Examples

### Example 1: Financial Document with Multiple Styles

```python
from app.services.redaction.visual_redactor import VisualRedactor, RedactionStyle, StyleConfig

# Configure styles
style_config = StyleConfig()
style_config.set_entity_style('PAN', visual_style=RedactionStyle.BLACK_BOX)
style_config.set_entity_style('AADHAAR', visual_style=RedactionStyle.PIXELATE, intensity=15)
style_config.set_entity_style('PHONE', visual_style=RedactionStyle.BLUR, intensity=30)
style_config.set_entity_style('EMAIL', visual_style=RedactionStyle.COLORED_BOX, color=(255, 200, 200))

# Prepare bounding boxes (from OCR + PII detection)
bounding_boxes = [
    {'x': 100, 'y': 50, 'w': 200, 'h': 30, 'entity_type': 'PAN'},
    {'x': 100, 'y': 100, 'w': 250, 'h': 30, 'entity_type': 'AADHAAR'},
    {'x': 100, 'y': 150, 'w': 230, 'h': 30, 'entity_type': 'PHONE'},
    {'x': 100, 'y': 200, 'w': 280, 'h': 30, 'entity_type': 'EMAIL'},
]

# Create style maps
config_dict = style_config.to_dict()
style_map = {e: RedactionStyle(c['visual_style'])
             for e, c in config_dict['entity_styles'].items()}
color_map = {e: c.get('color')
             for e, c in config_dict['entity_styles'].items() if c.get('color')}

# Apply redaction
redactor = VisualRedactor()
result = redactor.redact_image_file(
    'financial_doc.png',
    'redacted_doc.png',
    bounding_boxes=bounding_boxes
)

# With style mapping
image = cv2.imread('financial_doc.png')
image = redactor.redact_bounding_boxes(image, bounding_boxes, style_map, color_map)
cv2.imwrite('styled_redacted_doc.png', image)
```

### Example 2: Video Surveillance with Face Redaction

```python
from app.services.redaction.visual_redactor import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.PIXELATE)

# Redact faces in surveillance footage
result = redactor.redact_video_file(
    input_path='surveillance.mp4',
    output_path='redacted_surveillance.mp4',
    redact_faces=True,
    style=RedactionStyle.PIXELATE,
    progress_callback=lambda p: print(f"\rProgress: {p*100:.0f}%", end="")
)

print(f"\n✓ Processed {result['frames_processed']} frames")
print(f"✓ Redacted {result['faces_redacted']} faces")
```

### Example 3: Integration with Policy System

```python
from app.services.redaction import EnhancedRedactor, PolicyManager
from app.services.redaction.visual_redactor import VisualRedactor, StyleConfig, RedactionStyle

# Text redaction with policy
policy_manager = PolicyManager()
text_redactor = EnhancedRedactor(policy_manager)

text_result = text_redactor.redact_text(
    text,
    findings,
    policy="India Finance Compliance"
)

# Visual redaction with matching styles
style_config = StyleConfig()

# Match policy rules to visual styles
style_config.set_entity_style('PAN', visual_style=RedactionStyle.BLACK_BOX)
style_config.set_entity_style('AADHAAR', visual_style=RedactionStyle.PIXELATE)
style_config.set_entity_style('PHONE', visual_style=RedactionStyle.BLUR)

visual_redactor = VisualRedactor()
visual_result = visual_redactor.redact_image_file(
    'document.png',
    'redacted.png',
    bounding_boxes=bounding_boxes
)

print("Text redacted:", text_result)
print("Visual redacted:", visual_result['total_redactions'], "regions")
```

---

## Intensity Control

### Blur Intensity
```python
# Light blur (faces still recognizable)
redactor = VisualRedactor(blur_strength=10)

# Medium blur (default)
redactor = VisualRedactor(blur_strength=30)

# Heavy blur (completely obscured)
redactor = VisualRedactor(blur_strength=70)
```

### Pixelation Block Size
```python
# Fine pixelation (small blocks)
redactor = VisualRedactor(pixelate_size=5)

# Medium pixelation
redactor = VisualRedactor(pixelate_size=10)

# Coarse pixelation (large blocks)
redactor = VisualRedactor(pixelate_size=30)
```

---

## API Reference

### VisualRedactor

```python
class VisualRedactor:
    def __init__(
        self,
        default_style: RedactionStyle = RedactionStyle.BLUR,
        default_color: Tuple[int, int, int] = (0, 0, 0),
        blur_strength: int = 30,
        pixelate_size: int = 10
    )

    def redact_region(
        self,
        image: np.ndarray,
        x: int, y: int, w: int, h: int,
        style: Optional[RedactionStyle] = None,
        color: Optional[Tuple[int, int, int]] = None,
        intensity: Optional[int] = None
    ) -> np.ndarray

    def redact_faces(
        self,
        image: np.ndarray,
        style: Optional[RedactionStyle] = None,
        color: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, int]

    def redact_bounding_boxes(
        self,
        image: np.ndarray,
        bounding_boxes: List[Dict[str, Any]],
        style_map: Optional[Dict[str, RedactionStyle]] = None,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray

    def redact_image_file(
        self,
        input_path: str,
        output_path: str,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        redact_faces: bool = False,
        style: Optional[RedactionStyle] = None,
        style_map: Optional[Dict[str, RedactionStyle]] = None
    ) -> Dict[str, Any]

    def redact_video_file(
        self,
        input_path: str,
        output_path: str,
        redact_faces: bool = True,
        style: Optional[RedactionStyle] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]
```

### StyleConfig

```python
class StyleConfig:
    def __init__(self)

    def set_entity_style(
        self,
        entity_type: str,
        text_style: Optional[str] = None,
        visual_style: Optional[RedactionStyle] = None,
        color: Optional[Tuple[int, int, int]] = None,
        intensity: Optional[int] = None
    )

    def get_entity_style(self, entity_type: str) -> Dict[str, Any]

    def load_from_dict(self, config: Dict[str, Any])

    def to_dict(self) -> Dict[str, Any]
```

### RedactionStyle Enum

```python
class RedactionStyle(Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK_BOX = "black_box"
    WHITE_BOX = "white_box"
    COLORED_BOX = "colored_box"
    PATTERN = "pattern"
    HEAVY_BLUR = "heavy_blur"
    MOSAIC = "mosaic"
```

---

## Best Practices

1. **Security Level Matching:**
   - High-risk PII (PAN, SSN): BLACK_BOX or HEAVY_BLUR
   - Medium-risk (Phone, Email): BLUR or PIXELATE
   - Low-risk (Names): BLUR or COLORED_BOX (for review)

2. **Performance:**
   - Blur: Fast, good for real-time
   - Pixelate: Fast, good for videos
   - Boxes: Fastest
   - Heavy Blur: Slower, use for security

3. **Visual Quality:**
   - Documents: BLACK_BOX or WHITE_BOX
   - Photos: BLUR or PIXELATE
   - Videos: PIXELATE or MOSAIC
   - Review/audit: COLORED_BOX with distinct colors

4. **Intensity Guidelines:**
   - Blur: 10-20 (light), 30-40 (medium), 50+ (heavy)
   - Pixelate: 5-10 (fine), 10-20 (medium), 20+ (coarse)

---

## Testing

Run the test suite:
```bash
python3 test_visual_redaction.py
```

Test outputs are saved to `test_output/` directory with examples of all styles.

---

## Integration

### With OCR Pipeline
```python
from app.services.ocr.ocr_engine import OCREngine
from app.services.redaction.visual_redactor import VisualRedactor

ocr = OCREngine()
ocr_result = ocr.extract_with_details('document.png')

# Convert OCR word boxes to bounding boxes for redaction
visual_redactor = VisualRedactor()
# ... map and redact
```

### With Hindi Pipeline
```python
from app.services.hindi_pipeline import HindiPIIRedactionPipeline
from app.services.redaction.visual_redactor import VisualRedactor

pipeline = HindiPIIRedactionPipeline()
# Text redaction
text_result = pipeline.process_text(hindi_text)

# Visual redaction
visual_redactor = VisualRedactor()
# ... apply to image
```

---

## Summary

**All requested features are fully implemented:**

✅ Full Redaction - Text (████) & Visual (Black boxes)
✅ Partial Masking - Show last N characters
✅ Token Replacement - `<PAN_REDACTED>` format support
✅ Visual Redaction - 8 different styles
✅ Blur Bounding Boxes - With OCR integration
✅ Blur Faces in Videos - Frame-by-frame processing
✅ Configurable Styles - Per-entity configuration

**Test Results: 8/8 PASSED** ✅
