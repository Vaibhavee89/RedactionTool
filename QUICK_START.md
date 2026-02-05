# Quick Start Guide - RedactionTool Enterprise v2.0

## 🎉 All Features Successfully Implemented!

All requested Input & Ingestion features are now complete and tested. Here's how to use them.

---

## ✅ Feature Status

| Feature Category | Status | Details |
|-----------------|--------|---------|
| **Text Inputs** | ✅ Complete | Plain text, PDF (digital + scanned), DOCX |
| **Image Inputs** | ✅ Complete | PNG/JPG, multi-page TIFF, scanned documents |
| **Video Inputs** | ✅ Complete | MP4/AVI/MOV with frame-level face blurring |
| **Batch Processing** | ✅ Complete | Folder ingestion, mixed file types |
| **Streaming Mode** | ✅ Complete | Large file chunked processing |

---

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages (pdf2image now included)
pip install -r requirements.txt

# Install system dependencies
# macOS:
brew install poppler tesseract

# Ubuntu/Debian:
sudo apt-get install poppler-utils tesseract-ocr
```

### Step 2: Verify Installation
```bash
# Run the test suite
python3 test_new_features.py
```

Expected output: `🎉 ALL TESTS PASSED!`

### Step 3: Choose Your Interface

#### Option A: Enhanced Web UI (Recommended for Beginners)
```bash
streamlit run app/ui/streamlit_app_enhanced.py
```
Opens in browser at http://localhost:8501

**Features:**
- ✅ Single File Mode
- ✅ Batch Processing Mode
- ✅ Streaming Mode (Large Files)
- ✅ Side-by-side comparison
- ✅ Progress tracking
- ✅ ZIP downloads for batch results

#### Option B: Command-Line Interface (For Automation)
```bash
# Batch process a folder
python3 cli_batch.py batch -i /path/to/input -o /path/to/output -r

# Process large file
python3 cli_batch.py stream -i large_file.txt -o output.txt

# See all options
python3 cli_batch.py --help
```

#### Option C: Python API (For Integration)
```python
# Import what you need
from app.services.ingestion.batch_processor import BatchProcessor
from app.services.ingestion.streaming_processor import StreamingProcessor

# Batch processing
processor = BatchProcessor()
results = processor.process_directory('/input', '/output', recursive=True)

# Streaming processing
stream_proc = StreamingProcessor(chunk_size=10000)
result = stream_proc.process_large_text_file('large.txt', 'output.txt')
```

---

## 📖 Common Use Cases

### Use Case 1: Process a Single PDF
```bash
# Start the enhanced UI
streamlit run app/ui/streamlit_app_enhanced.py

# Then:
# 1. Select "Single File" mode
# 2. Upload your PDF
# 3. Check "Force OCR" if it's scanned
# 4. Click "Analyze & Redact"
# 5. Download redacted file
```

### Use Case 2: Batch Process Mixed Files
```bash
# Using CLI
python3 cli_batch.py batch \
  -i /path/to/folder \
  -o /path/to/output \
  -r \
  --types .pdf,.docx,.png

# Or using Web UI:
# 1. Select "Batch Processing" mode
# 2. Upload multiple files
# 3. Click "Process All Files"
# 4. Download ZIP of results
```

### Use Case 3: Process Large Document (>100MB)
```bash
# Using CLI
python3 cli_batch.py stream \
  -i huge_document.txt \
  -o redacted.txt \
  --chunk-size 20000

# Or using Web UI:
# 1. Select "Streaming (Large Files)" mode
# 2. Upload your large file
# 3. Adjust chunk size slider
# 4. Click "Process Large File"
# 5. Wait for progress bar
# 6. Download result
```

### Use Case 4: Process Scanned Multi-page Document
```python
from app.services.ingestion.multipage_loader import MultiPageDocumentLoader
from app.services.pii.detector_engine import DetectorEngine
from app.services.redaction.redactor import Redactor

# Load multi-page document
loader = MultiPageDocumentLoader()
text = loader.load('scanned_document.tiff')

# Detect PII
detector = DetectorEngine()
findings = detector.detect(text)

# Redact
redactor = Redactor()
policy = {f['entity_type']: 'block' for f in findings}
redacted_text = redactor.redact_text(text, findings, policy)

# Save
with open('redacted_output.txt', 'w') as f:
    f.write(redacted_text)
```

### Use Case 5: Process Videos
```bash
# Start the enhanced UI
streamlit run app/ui/streamlit_app_enhanced.py

# Then:
# 1. Upload MP4/AVI/MOV video
# 2. Click "Process Video"
# 3. Watch progress bar
# 4. Download redacted video with blurred faces
```

---

## 🎯 Feature Highlights

### New in v2.0

#### 1. Plain Text Support
```python
from app.services.ingestion.text_loader import TextLoader
loader = TextLoader()
text = loader.load('document.txt')
```

#### 2. Scanned PDF with OCR
```python
from app.services.ingestion.pdf_loader import PDFLoader
loader = PDFLoader()
# Automatic detection
text = loader.load('document.pdf')
# Force OCR
text = loader.load('scanned.pdf', force_ocr=True)
```

#### 3. Multi-page TIFF/Scanned Docs
```python
from app.services.ingestion.multipage_loader import MultiPageDocumentLoader
loader = MultiPageDocumentLoader()
text = loader.load('multi_page.tiff')
pages = loader.get_page_images('multi_page.tiff')
```

#### 4. AVI/MOV Video Support
```python
from app.services.redaction.video_redactor import VideoRedactor
redactor = VideoRedactor()
redactor.redact_faces('video.avi', 'output.avi')  # Now works!
redactor.redact_faces('video.mov', 'output.mov')  # Now works!
```

#### 5. Batch Processing
```python
from app.services.ingestion.batch_processor import BatchProcessor
processor = BatchProcessor()

# Process directory
results = processor.process_directory(
    '/input',
    '/output',
    recursive=True,
    file_types=['.pdf', '.docx']
)

print(f"Processed: {len(results['processed'])} files")
print(f"Total PII found: {results['stats']['total_pii_found']}")
```

#### 6. Streaming for Large Files
```python
from app.services.ingestion.streaming_processor import StreamingProcessor
processor = StreamingProcessor(chunk_size=10000)

# Estimate time first
estimate = processor.estimate_processing_time('huge_file.txt')
print(f"Will take ~{estimate['estimated_minutes']} minutes")

# Process
result = processor.process_large_text_file(
    'huge_file.txt',
    'output.txt',
    overlap=500
)
```

---

## 📊 Performance Tips

### For Best Performance:

1. **Small-Medium Files (<10MB)**: Use Single File mode
2. **Multiple Files**: Use Batch Processing mode
3. **Large Files (>100MB)**: Use Streaming mode
4. **Scanned PDFs**: Enable OCR or use force_ocr=True
5. **Videos**: Be patient - processing is CPU-intensive
6. **Batch Processing**: Use file_types filter to focus on specific formats

### Recommended Settings:

| File Size | Mode | Chunk Size | Notes |
|-----------|------|------------|-------|
| < 10 MB | Single File | N/A | Fast |
| 10-100 MB | Batch/Single | N/A | Moderate |
| > 100 MB | Streaming | 10000-20000 | Memory efficient |
| > 1 GB | Streaming | 5000-10000 | Slower but safe |

---

## 🔧 Troubleshooting

### Issue: "pdf2image not found"
```bash
pip install pdf2image
```

### Issue: "Tesseract not found"
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: "Poppler not found"
```bash
# macOS
brew install poppler

# Ubuntu
sudo apt-get install poppler-utils

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### Issue: Memory error with large files
**Solution**: Use streaming mode instead of single file mode

### Issue: Scanned PDF not extracting text
**Solution**: Enable "Force OCR" option or use force_ocr=True

---

## 📚 Documentation Files

- `README.md` - Project overview
- `FEATURES.md` - Detailed feature documentation
- `IMPLEMENTATION_SUMMARY.md` - What was implemented (this release)
- `DEPLOY.md` - Docker deployment guide
- `QUICK_START.md` - This file

---

## 🎓 Next Steps

1. ✅ Try the enhanced web UI
2. ✅ Test batch processing with sample files
3. ✅ Experiment with streaming mode on large files
4. ✅ Integrate into your workflow via Python API
5. ✅ Read FEATURES.md for advanced usage

---

## 💡 Examples Repository

Create a test folder:
```bash
mkdir test_files
cd test_files

# Create test text file
echo "John Doe lives at 123 Main St. Call him at 555-1234." > sample.txt

# Now process it
cd ..
python3 cli_batch.py batch -i test_files -o output_files
```

---

## 🤝 Need Help?

- Check documentation files (README.md, FEATURES.md)
- Run test suite: `python3 test_new_features.py`
- View examples in FEATURES.md
- Check system requirements in IMPLEMENTATION_SUMMARY.md

---

## ✨ Summary

**You now have:**
- ✅ 7 new file loaders/processors
- ✅ Enhanced web UI with 3 modes
- ✅ CLI tool for automation
- ✅ Python API for integration
- ✅ Support for 10+ file formats
- ✅ Batch processing capabilities
- ✅ Large file streaming
- ✅ Comprehensive documentation

**Start here:**
```bash
streamlit run app/ui/streamlit_app_enhanced.py
```

🎉 **Enjoy RedactionTool Enterprise v2.0!**
