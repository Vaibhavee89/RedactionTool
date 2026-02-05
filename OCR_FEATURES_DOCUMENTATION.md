# OCR & Preprocessing Features - Complete Documentation

## ✅ Implementation Status

All requested OCR & Preprocessing features have been successfully implemented and tested!

---

## 📋 Feature Overview

### ✅ 1. OCR Engines
- **Tesseract OCR** - Default, fast, widely supported
- **PaddleOCR** - Better for Asian/Indian languages

### ✅ 2. Language Support
- **English** (eng)
- **Hindi** (hin)
- **Other Indian Languages** - Tamil, Telugu, Kannada, etc. (via PaddleOCR)
- **Multi-language detection** - Automatic language detection

### ✅ 3. Image Preprocessing
- **De-skewing** - Automatic rotation correction
- **Noise removal** - Denoising with Non-local Means
- **Contrast enhancement** - CLAHE (Adaptive Histogram Equalization)
- **Binarization** - Adaptive thresholding
- **Border removal** - Remove image borders

### ✅ 4. Layout-Aware Extraction
- **Paragraph detection** - Intelligent paragraph grouping
- **Table detection** - Detect and extract tables
- **Heading detection** - Identify document headings
- **Reading order** - Maintains proper text flow
- **Structured output** - JSON format with block types

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Image Input                          │
│              (PNG, JPG, TIFF, PDF)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             Image Preprocessor (NEW)                    │
├─────────────────────────────────────────────────────────┤
│  • De-skewing (auto rotation)                           │
│  • Noise removal (denoising)                            │
│  • Contrast enhancement (CLAHE)                         │
│  • Binarization (adaptive threshold)                    │
│  • Border removal                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              OCR Engine (NEW)                           │
├─────────────────────────────────────────────────────────┤
│  • Tesseract (default)                                  │
│  • PaddleOCR (for Asian languages)                      │
│  • Multi-language support                               │
│  • Confidence scoring                                   │
│  • Bounding box detection                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Layout Analyzer (NEW)                        │
├─────────────────────────────────────────────────────────┤
│  • Paragraph detection                                  │
│  • Table extraction                                     │
│  • Heading identification                               │
│  • Reading order detection                              │
│  • Structured output                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Structured Text Output                          │
│  (Plain text / JSON / Layout-aware)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 New Files Created

### Core Modules (4 files):

1. **`app/services/ocr/__init__.py`** - OCR package
2. **`app/services/ocr/ocr_engine.py`** - Advanced OCR engine (352 lines)
3. **`app/services/ocr/image_preprocessor.py`** - Image preprocessing (280 lines)
4. **`app/services/ocr/layout_analyzer.py`** - Layout analysis (432 lines)

### Enhanced Loaders (1 file):

5. **`app/services/ingestion/advanced_image_loader.py`** - Advanced image loader (221 lines)

### Testing & Documentation (2 files):

6. **`test_ocr_features.py`** - Comprehensive test suite
7. **`OCR_FEATURES_DOCUMENTATION.md`** - This file

**Total**: 7 new files, ~1,300 lines of code

---

## 🚀 Usage Examples

### Example 1: Basic OCR with Preprocessing

```python
from app.services.ocr.ocr_engine import OCREngine

# Create OCR engine with preprocessing
ocr = OCREngine(
    engine='tesseract',
    languages=['eng'],
    preprocess=True  # Enable automatic preprocessing
)

# Extract text
text = ocr.extract_text('document.png')
print(text)
```

### Example 2: Advanced Image Preprocessing

```python
from app.services.ocr.image_preprocessor import ImagePreprocessor
import cv2

# Create preprocessor
preprocessor = ImagePreprocessor()

# Load image
image = cv2.imread('skewed_document.png')

# Preprocess with all features
processed = preprocessor.preprocess(
    image,
    deskew=True,           # Correct rotation
    denoise=True,          # Remove noise
    enhance_contrast=True, # Improve contrast
    binarize=True,         # Convert to black/white
    remove_borders=False
)

# Save preprocessed image
cv2.imwrite('processed.png', processed)

# Use for OCR
ocr = OCREngine()
text = ocr.extract_text(processed)
```

### Example 3: Multi-Language OCR

```python
from app.services.ocr.ocr_engine import OCREngine

# English + Hindi OCR
ocr = OCREngine(
    engine='tesseract',
    languages=['eng', 'hin'],  # English and Hindi
    preprocess=True
)

text = ocr.extract_text('hindi_document.png')
print(text)

# Or use PaddleOCR for better Asian language support
ocr_paddle = OCREngine(
    engine='paddle',
    languages=['hi'],  # Hindi (PaddleOCR format)
    preprocess=True
)

text = ocr_paddle.extract_text('hindi_document.png')
```

### Example 4: Layout-Aware Extraction

```python
from app.services.ocr.layout_analyzer import LayoutAnalyzer

# Create analyzer
analyzer = LayoutAnalyzer()

# Analyze document layout
result = analyzer.analyze_layout('complex_document.png')

# Access structured data
print(f"Found {result['layout_metadata']['num_paragraphs']} paragraphs")
print(f"Found {result['layout_metadata']['num_tables']} tables")
print(f"Found {result['layout_metadata']['num_headings']} headings")

# Get full text with structure preserved
print(result['full_text'])

# Get text grouped by type
for block_type, texts in result['text_by_type'].items():
    print(f"\n{block_type.upper()}:")
    for text in texts:
        print(f"  - {text}")
```

### Example 5: Extract Only Tables

```python
from app.services.ocr.layout_analyzer import LayoutAnalyzer

analyzer = LayoutAnalyzer()

# Extract only tables
tables = analyzer.extract_tables('invoice.png')

for i, table in enumerate(tables):
    print(f"\nTable {i+1}:")
    print(table['text'])
    print(f"Location: {table['bbox']}")
    print(f"Confidence: {table['confidence']}")
```

### Example 6: Extract Only Paragraphs

```python
from app.services.ocr.layout_analyzer import LayoutAnalyzer

analyzer = LayoutAnalyzer()

# Extract only paragraphs
paragraphs = analyzer.extract_paragraphs('article.png')

for i, para in enumerate(paragraphs):
    print(f"\nParagraph {i+1}:")
    print(para)
```

### Example 7: Advanced Image Loader (All-in-One)

```python
from app.services.ingestion.advanced_image_loader import AdvancedImageLoader

# Create advanced loader with all features
loader = AdvancedImageLoader(
    ocr_engine='tesseract',
    languages=['eng', 'hin'],
    preprocess=True,
    layout_analysis=True
)

# Simple text extraction (with preprocessing)
text = loader.load('document.png')

# Layout-aware extraction
layout_result = loader.load_with_layout('complex_doc.png')
print(layout_result['full_text'])

# Extract tables only
tables = loader.extract_tables('invoice.png')

# Extract paragraphs only
paragraphs = loader.extract_paragraphs('article.png')

# Multi-language extraction
results = loader.load_multi_language('mixed_lang.png', languages=['eng', 'hin'])
```

### Example 8: Custom Preprocessing Pipeline

```python
from app.services.ocr.image_preprocessor import ImagePreprocessor
from app.services.ocr.ocr_engine import OCREngine
import cv2

# Create preprocessor
preprocessor = ImagePreprocessor()

# Load image
image = cv2.imread('poor_quality.png')

# Custom preprocessing
# 1. Remove borders
image = preprocessor.remove_border(image, border_size=20)

# 2. Resize for better OCR
image = preprocessor.resize_for_ocr(image, target_height=2000)

# 3. Denoise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised = preprocessor.denoise_image(gray)

# 4. Enhance contrast
enhanced = preprocessor.enhance_contrast(denoised)

# 5. Deskew
deskewed = preprocessor.deskew_image(enhanced)

# 6. Binarize
binary = preprocessor.binarize_image(deskewed)

# 7. Save preprocessed
cv2.imwrite('fully_processed.png', binary)

# 8. OCR
ocr = OCREngine()
text = ocr.extract_text(binary, preprocess=False)
```

### Example 9: Get Detailed OCR Data with Bounding Boxes

```python
from app.services.ocr.ocr_engine import OCREngine

ocr = OCREngine(engine='tesseract', preprocess=True)

# Get detailed data
details = ocr.extract_with_details('document.png')

print(f"Engine: {details['engine']}")
print(f"Languages: {details['languages']}")
print(f"Full text: {details['text']}")

# Access individual words with bounding boxes
for word in details['words']:
    print(f"Word: {word['text']}")
    print(f"  Confidence: {word['confidence']}")
    print(f"  Position: {word['bbox']}")
    print(f"  Block: {word['block_num']}, Line: {word['line_num']}")
```

---

## 🎯 Feature Comparison

### OCR Engines

| Feature | Tesseract | PaddleOCR |
|---------|-----------|-----------|
| English | ✅ Excellent | ✅ Good |
| Hindi | ✅ Good | ✅ Excellent |
| Other Indian Languages | ⚠️ Limited | ✅ Excellent |
| Speed | Fast | Moderate |
| Accuracy (English) | High | High |
| Accuracy (Hindi) | Moderate | High |
| Installation | Easy | Moderate |

### Preprocessing Techniques

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| De-skewing | Correct rotation | Scanned documents, photos |
| Denoising | Remove noise | Low-quality scans, camera photos |
| Contrast Enhancement | Improve readability | Faded documents, low contrast |
| Binarization | Convert to B&W | General OCR improvement |
| Border Removal | Remove edges | Scanned documents with borders |

---

## 📊 Test Results

All features tested and working:

```
============================================================
📋 FINAL SUMMARY
============================================================
Module Imports: ✅ PASS
Image Preprocessing: ✅ PASS
OCR Engine: ✅ PASS
Layout Analysis: ✅ PASS
Advanced Loader: ✅ PASS

🎉 ALL TESTS PASSED!
============================================================
```

**Test Coverage:**
- ✅ Image preprocessing (all techniques)
- ✅ OCR with Tesseract
- ✅ Layout-aware extraction
- ✅ Paragraph detection
- ✅ Heading detection
- ✅ Table detection
- ✅ Advanced image loader
- ✅ Multi-language support

---

## 🔧 Configuration

### Supported Languages (Tesseract)

```python
# Check supported languages
from app.services.ocr.ocr_engine import OCREngine

langs = OCREngine.get_supported_languages('tesseract')
print(langs)  # ['eng', 'hin', 'san', 'mar', 'tam', 'tel', ...]
```

### Supported Languages (PaddleOCR)

```python
langs = OCREngine.get_supported_languages('paddle')
print(langs)  # ['en', 'ch', 'ta', 'te', 'ka', 'hi', ...]
```

### Language Codes

| Language | Tesseract | PaddleOCR |
|----------|-----------|-----------|
| English | eng | en |
| Hindi | hin | hi |
| Tamil | tam | ta |
| Telugu | tel | te |
| Kannada | kan | ka |
| Marathi | mar | - |
| Sanskrit | san | - |

---

## 🚀 Integration with Existing System

### Update Existing Image Loader

```python
# OLD: Basic image loader
from app.services.ingestion.image_loader import ImageLoader

loader = ImageLoader()
text = loader.load('image.png')

# NEW: Advanced image loader with preprocessing
from app.services.ingestion.advanced_image_loader import AdvancedImageLoader

loader = AdvancedImageLoader(
    ocr_engine='tesseract',
    languages=['eng', 'hin'],
    preprocess=True,
    layout_analysis=True
)
text = loader.load('image.png')
```

### Add to Batch Processing

```python
from app.services.ingestion.batch_processor import BatchProcessor
from app.services.ingestion.advanced_image_loader import AdvancedImageLoader

# Enhance batch processor with advanced OCR
class EnhancedBatchProcessor(BatchProcessor):
    def __init__(self):
        super().__init__()
        # Replace image loader
        self.loaders['.png'] = AdvancedImageLoader(preprocess=True)
        self.loaders['.jpg'] = AdvancedImageLoader(preprocess=True)

processor = EnhancedBatchProcessor()
results = processor.process_directory('input/', 'output/')
```

---

## 📈 Performance Tips

### For Best OCR Results:

1. **Enable preprocessing** for scanned documents
2. **Use higher resolution** images (min 300 DPI)
3. **Choose the right engine**:
   - English documents → Tesseract
   - Hindi/Indian languages → PaddleOCR
4. **Enable layout analysis** for structured documents
5. **Pre-crop images** to remove unnecessary borders

### Performance Optimization:

```python
# Fast (no preprocessing)
ocr = OCREngine(engine='tesseract', preprocess=False)
text = ocr.extract_text('clean_document.png')

# Balanced (selective preprocessing)
ocr = OCREngine(engine='tesseract', preprocess=True)
text = ocr.extract_text('scanned_doc.png', preprocess=True)

# Maximum accuracy (full preprocessing + layout)
loader = AdvancedImageLoader(
    ocr_engine='paddle',
    preprocess=True,
    layout_analysis=True
)
result = loader.load_with_layout('complex_doc.png')
```

---

## 🔍 Debugging & Troubleshooting

### View Preprocessed Images

```python
from app.services.ocr.image_preprocessor import ImagePreprocessor
import cv2

preprocessor = ImagePreprocessor()

# Preprocess and save each step
image = cv2.imread('input.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save each step for inspection
cv2.imwrite('1_denoised.png', preprocessor.denoise_image(gray))
cv2.imwrite('2_enhanced.png', preprocessor.enhance_contrast(gray))
cv2.imwrite('3_deskewed.png', preprocessor.deskew_image(gray))
cv2.imwrite('4_binary.png', preprocessor.binarize_image(gray))
```

### Check OCR Confidence

```python
ocr = OCREngine()
details = ocr.extract_with_details('document.png')

# Check overall confidence
avg_conf = sum(w['confidence'] for w in details['words']) / len(details['words'])
print(f"Average confidence: {avg_conf:.2f}%")

# Find low confidence words
low_conf_words = [w for w in details['words'] if w['confidence'] < 60]
print(f"Low confidence words: {len(low_conf_words)}")
```

### Compare OCR Engines

```python
from app.services.ocr.ocr_engine import OCREngine

# Try both engines
tesseract = OCREngine(engine='tesseract', preprocess=True)
paddle = OCREngine(engine='paddle', preprocess=True)

text_tess = tesseract.extract_text('hindi_doc.png')
text_paddle = paddle.extract_text('hindi_doc.png')

print("Tesseract:", text_tess)
print("PaddleOCR:", text_paddle)
```

---

## 📚 Additional Documentation

- **Image Preprocessor API**: See `app/services/ocr/image_preprocessor.py`
- **OCR Engine API**: See `app/services/ocr/ocr_engine.py`
- **Layout Analyzer API**: See `app/services/ocr/layout_analyzer.py`
- **Advanced Loader API**: See `app/services/ingestion/advanced_image_loader.py`

---

## ✨ Summary

**All OCR & Preprocessing features are implemented and tested:**

✅ **OCR Engines**: Tesseract + PaddleOCR
✅ **Language Support**: English + Hindi + Indian languages
✅ **Preprocessing**: De-skewing, denoising, contrast, binarization
✅ **Layout Analysis**: Paragraphs, tables, headings
✅ **Advanced Integration**: Ready to use in existing workflows

**Key Benefits:**
- 🚀 Improved OCR accuracy
- 🌍 Multi-language support
- 📄 Structure-aware extraction
- 🔧 Flexible preprocessing pipeline
- 📊 Detailed confidence scoring

**Test Results**: 5/5 tests passed ✅

Start using now with `AdvancedImageLoader`!
