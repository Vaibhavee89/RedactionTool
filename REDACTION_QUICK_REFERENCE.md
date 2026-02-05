# Redaction & Masking Modes - Quick Reference

## ✅ Implementation: COMPLETE

All requested features are implemented and tested.

---

## 🎯 Feature Checklist

| Feature | Status | Files |
|---------|--------|-------|
| ✅ Full Redaction (████) | Complete | `enhanced_redactor.py`, `visual_redactor.py` |
| ✅ Partial Masking (last N) | Complete | `enhanced_redactor.py` |
| ✅ Token Replacement | Complete | `enhanced_redactor.py` |
| ✅ Blur Boxes (Images) | Complete | `visual_redactor.py` |
| ✅ Blur Faces (Videos) | Complete | `visual_redactor.py` |
| ✅ Configurable Styles | Complete | `visual_redactor.py` |

---

## 🚀 Quick Examples

### 1. Full Redaction

**Text:**
```python
from app.services.redaction import EnhancedRedactor

redactor = EnhancedRedactor()
rules = {'PAN': {'action': 'block'}}
# 'ABCDE1234F' → '██████████'
```

**Visual:**
```python
from app.services.redaction import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.BLACK_BOX)
# Creates solid black rectangle over region
```

---

### 2. Partial Masking

```python
# Show last 4 characters
rules = {
    'PHONE': {
        'action': 'mask',
        'show_last': 4,
        'mask_char': 'X'
    }
}

# '9876543210' → 'XXXXXX3210'
```

---

### 3. Token Replacement

```python
# Label format
rules = {
    'PAN': {
        'action': 'label',
        'format': '<{entity_type}_REDACTED>'
    }
}

# 'ABCDE1234F' → '<PAN_REDACTED>'
```

---

### 4. Visual Styles (8 Options)

```python
from app.services.redaction import RedactionStyle

# Available styles:
RedactionStyle.BLUR           # Gaussian blur
RedactionStyle.HEAVY_BLUR     # Extra strong blur
RedactionStyle.PIXELATE       # Pixelation
RedactionStyle.MOSAIC         # Mosaic effect
RedactionStyle.BLACK_BOX      # Solid black
RedactionStyle.WHITE_BOX      # Solid white
RedactionStyle.COLORED_BOX    # Custom color
RedactionStyle.PATTERN        # Pattern fill
```

---

### 5. Image Redaction

```python
from app.services.redaction import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.BLUR)

bounding_boxes = [
    {'x': 100, 'y': 50, 'width': 200, 'height': 30, 'entity_type': 'PAN'}
]

result = redactor.redact_image_file(
    'document.png',
    'redacted.png',
    bounding_boxes=bounding_boxes
)
```

---

### 6. Video Face Redaction

```python
from app.services.redaction import VisualRedactor, RedactionStyle

redactor = VisualRedactor(default_style=RedactionStyle.PIXELATE)

result = redactor.redact_video_file(
    'video.mp4',
    'redacted_video.mp4',
    redact_faces=True
)
```

---

### 7. Per-Entity Styles

```python
from app.services.redaction import StyleConfig, RedactionStyle

style_config = StyleConfig()

# PAN: Black box
style_config.set_entity_style(
    'PAN',
    text_style='block',
    visual_style=RedactionStyle.BLACK_BOX
)

# Phone: Pixelate
style_config.set_entity_style(
    'PHONE',
    text_style='mask',
    visual_style=RedactionStyle.PIXELATE,
    intensity=15
)

# Email: Colored box
style_config.set_entity_style(
    'EMAIL',
    text_style='hash',
    visual_style=RedactionStyle.COLORED_BOX,
    color=(200, 200, 200)
)
```

---

## 📊 Visual Styles Comparison

| Style | Appearance | Best For |
|-------|-----------|----------|
| BLUR | Smooth blur | Faces, general PII |
| HEAVY_BLUR | Very blurred | High security |
| PIXELATE | Block pixels | Japanese-style censoring |
| MOSAIC | Color blocks | Artistic effect |
| BLACK_BOX | Solid black | Military/classified |
| WHITE_BOX | Solid white | Print documents |
| COLORED_BOX | Custom color | Color-coding by type |
| PATTERN | Diagonal lines | Decorative |

---

## 🎨 Text vs Visual

| Feature | Text Implementation | Visual Implementation |
|---------|-------------------|----------------------|
| **Full Redaction** | `action: 'block'` → `████` | `BLACK_BOX` → solid rectangle |
| **Partial Mask** | `show_last: 4` → `XXX3210` | Selective regions |
| **Token** | `label` → `[PAN]` | N/A (text only) |
| **Blur** | N/A (visual only) | `BLUR` → blurred region |
| **Pixelate** | N/A (visual only) | `PIXELATE` → pixel blocks |

---

## 📁 Test Outputs

Run tests to see examples:
```bash
python3 test_visual_redaction.py
```

Test images saved to `test_output/`:
- `test_blur.png` - Gaussian blur example
- `test_pixelate.png` - Pixelation example
- `test_black_box.png` - Black box example
- `test_per_entity_styles.png` - Multiple styles
- And 16+ more examples

---

## 🔧 Integration

### With PII Detection
```python
from app.services.pii.ensemble_detector import EnsembleDetector
from app.services.redaction import EnhancedRedactor

detector = EnsembleDetector()
redactor = EnhancedRedactor()

findings = detector.detect(text)
redacted = redactor.redact_text(text, findings)
```

### With Policy System
```python
from app.services.redaction import EnhancedRedactor, PolicyManager

policy_manager = PolicyManager()
redactor = EnhancedRedactor(policy_manager)

redacted = redactor.redact_text(
    text,
    findings,
    policy="India Finance Compliance"
)
```

### With OCR + Visual Redaction
```python
from app.services.ocr.ocr_engine import OCREngine
from app.services.redaction import VisualRedactor

ocr = OCREngine()
visual_redactor = VisualRedactor()

ocr_result = ocr.extract_with_details('document.png')
# Map PII to bounding boxes
result = visual_redactor.redact_image_file(
    'document.png',
    'redacted.png',
    bounding_boxes=boxes
)
```

---

## 📚 Full Documentation

- **VISUAL_REDACTION_GUIDE.md** - Complete guide with examples
- **REDACTION_MODES_IMPLEMENTATION_SUMMARY.md** - Detailed implementation info
- **test_visual_redaction.py** - Working code examples

---

## ✅ Test Results

```
Full Redaction: ✅ PASS
Partial Masking: ✅ PASS
Token Replacement: ✅ PASS
Visual Redaction Styles: ✅ PASS
Blur Bounding Boxes: ✅ PASS
Configurable Styles: ✅ PASS
Intensity Levels: ✅ PASS
Integration Summary: ✅ PASS

8/8 TESTS PASSED ✅
```

---

## 🎉 Summary

**All features implemented:**
- ✅ Text redaction (7 strategies)
- ✅ Visual redaction (8 styles)
- ✅ Video face blurring
- ✅ Configurable per-entity styles
- ✅ Integration with existing systems

**Ready for production!** 🚀
