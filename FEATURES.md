# RedactionTool Enterprise - Feature Documentation

## Input & Ingestion Features

### ✅ Text Inputs

#### 1. Plain Text (.txt)
- **Status**: ✅ Implemented
- **Location**: `app/services/ingestion/text_loader.py`
- **Features**:
  - UTF-8 encoding support
  - Error handling for encoding issues
  - Metadata extraction (file size, modification date)

#### 2. PDF (Digital + Scanned)
- **Status**: ✅ Implemented with OCR Fallback
- **Location**: `app/services/ingestion/pdf_loader.py`
- **Features**:
  - Digital PDF text extraction (via pdfplumber)
  - Automatic OCR fallback for scanned PDFs
  - Force OCR mode for low-quality scans
  - Multi-page support
  - Page-level text extraction
  - Metadata extraction (page count, document properties)

#### 3. DOCX
- **Status**: ✅ Implemented
- **Location**: `app/services/ingestion/docx_loader.py`
- **Features**:
  - Paragraph-level text extraction
  - Metadata extraction (author, title)
  - Full document content loading

---

### ✅ Image Inputs

#### 1. PNG / JPG / JPEG
- **Status**: ✅ Implemented
- **Location**: `app/services/ingestion/image_loader.py`
- **Features**:
  - Tesseract OCR text extraction
  - Bounding box detection for visual redaction
  - Word-level coordinate mapping
  - Face detection and blurring
  - PII text region blurring

#### 2. Multi-page Scanned Documents
- **Status**: ✅ Implemented
- **Location**: `app/services/ingestion/multipage_loader.py`
- **Features**:
  - Multi-page TIFF support
  - PDF-to-image conversion with OCR
  - Image directory batch processing
  - Page-level text extraction
  - Individual page image export for redaction

---

### ✅ Video Inputs

#### 1. MP4 / AVI / MOV
- **Status**: ✅ Implemented
- **Location**: `app/services/redaction/video_redactor.py`
- **Features**:
  - MP4 format support
  - AVI format support with XVID codec
  - MOV format support
  - Frame-level processing
  - Real-time progress tracking
  - Face detection using Haar Cascade
  - Gaussian blur for privacy protection

#### 2. Frame-level Processing
- **Status**: ✅ Implemented
- **Features**:
  - Frame-by-frame analysis
  - Face detection per frame
  - Dynamic blur kernel sizing
  - Progress callback support
  - Memory-efficient streaming

---

### ✅ Batch Processing

#### 1. Folder-level Ingestion
- **Status**: ✅ Implemented
- **Location**: `app/services/ingestion/batch_processor.py`
- **Features**:
  - Directory scanning (recursive and non-recursive)
  - File type filtering
  - Progress tracking with callbacks
  - Detailed processing statistics
  - Error handling per file
  - Mixed file type support in single run

#### 2. Mixed File Types in One Run
- **Status**: ✅ Implemented
- **Features**:
  - Automatic format detection
  - Unified processing pipeline
  - Type-specific redaction strategies
  - Aggregate statistics (documents/images/videos)
  - Batch result reporting
  - ZIP export for batch downloads

**Usage Example:**
```python
from app.services.ingestion.batch_processor import BatchProcessor

processor = BatchProcessor()

# Process entire directory
results = processor.process_directory(
    input_dir="/path/to/files",
    output_dir="/path/to/output",
    recursive=True,
    file_types=['.pdf', '.docx', '.png']  # Optional filter
)

# Process specific file list
results = processor.process_file_list(
    file_paths=['file1.pdf', 'file2.png', 'video.mp4'],
    output_dir="/path/to/output"
)
```

**Result Structure:**
```python
{
    'total_files': 10,
    'processed': [
        {
            'filename': 'doc1.pdf',
            'type': '.pdf',
            'pii_count': 5,
            'output_path': '/output/redacted_doc1.pdf.txt',
            'findings': [...]
        },
        ...
    ],
    'failed': [
        {'file': 'corrupt.pdf', 'error': 'Failed to open'}
    ],
    'stats': {
        'text_documents': 5,
        'images': 3,
        'videos': 2,
        'total_pii_found': 47
    }
}
```

---

### ✅ Streaming Mode (Large Files)

#### 1. Chunked Processing
- **Status**: ✅ Implemented
- **Location**: `app/services/ingestion/streaming_processor.py`
- **Features**:
  - Memory-efficient chunked reading
  - Configurable chunk size (default: 10,000 chars)
  - Overlap between chunks to avoid boundary PII misses
  - Progress tracking
  - Support for TXT, PDF, and DOCX
  - Processing time estimation

**Usage Example:**
```python
from app.services.ingestion.streaming_processor import StreamingProcessor

processor = StreamingProcessor(chunk_size=10000)

# Estimate processing time first
estimate = processor.estimate_processing_time('/path/to/large_file.txt')
print(f"Estimated time: {estimate['estimated_minutes']} minutes")

# Process large file
result = processor.process_large_text_file(
    file_path='/path/to/large_file.txt',
    output_path='/path/to/output.txt',
    overlap=500,  # Character overlap between chunks
    progress_callback=lambda p: print(f"Progress: {p*100}%")
)
```

**Benefits:**
- Handles files > 100MB without memory issues
- Constant memory usage regardless of file size
- Real-time progress tracking
- No file size limitations

---

## Feature Matrix

| Feature | Status | File Types | Mode | Location |
|---------|--------|------------|------|----------|
| Plain Text | ✅ | .txt | Single/Batch/Stream | `text_loader.py` |
| Digital PDF | ✅ | .pdf | Single/Batch/Stream | `pdf_loader.py` |
| Scanned PDF (OCR) | ✅ | .pdf | Single/Batch/Stream | `pdf_loader.py` |
| DOCX | ✅ | .docx | Single/Batch/Stream | `docx_loader.py` |
| Images | ✅ | .png/.jpg/.jpeg | Single/Batch | `image_loader.py` |
| Multi-page TIFF | ✅ | .tiff/.tif | Single/Batch | `multipage_loader.py` |
| MP4 Video | ✅ | .mp4 | Single/Batch | `video_redactor.py` |
| AVI Video | ✅ | .avi | Single/Batch | `video_redactor.py` |
| MOV Video | ✅ | .mov | Single/Batch | `video_redactor.py` |
| Batch Processing | ✅ | All | Batch | `batch_processor.py` |
| Streaming | ✅ | .txt/.pdf/.docx | Stream | `streaming_processor.py` |

---

## User Interface Options

### 1. Enhanced Streamlit UI
- **File**: `app/ui/streamlit_app_enhanced.py`
- **Features**:
  - Single file processing mode
  - Batch processing mode with multi-file upload
  - Streaming mode for large files
  - Progress tracking with visual feedback
  - Side-by-side comparison views
  - Detection reports and statistics
  - ZIP download for batch results
  - Real-time PII count metrics

### 2. Legacy Streamlit UI
- **File**: `app/ui/streamlit_app.py`
- **Features**:
  - Single file processing only
  - Basic upload and redaction
  - Original/Redacted comparison

### 3. Command-Line Interface
- **File**: `app/main.py`
- **Features**:
  - CLI-based processing
  - Batch mode support
  - Scripting and automation friendly

---

## Architecture

```
User Input (Files/Folders)
    ↓
Ingestion Layer
    ├── TextLoader (.txt)
    ├── PDFLoader (.pdf, with OCR fallback)
    ├── DocxLoader (.docx)
    ├── ImageLoader (.png, .jpg)
    ├── MultiPageLoader (.tiff, multi-page PDFs)
    └── BatchProcessor (folder/multiple files)
    ↓
Text Extraction & Chunking
    ├── Standard mode (full file)
    └── Streaming mode (chunked)
    ↓
PII Detection Engine
    ├── RegexProvider (patterns)
    ├── NERProvider (spaCy)
    └── PresidioProvider (ML-based)
    ↓
Redaction Layer
    ├── Redactor (text: block/mask/label)
    ├── ImageRedactor (visual: face + text blur)
    └── VideoRedactor (frame-by-frame face blur)
    ↓
Output (Redacted files)
```

---

## Performance Characteristics

| Feature | Memory Usage | Speed | Best For |
|---------|--------------|-------|----------|
| Single File | Moderate | Fast | Individual documents |
| Batch Processing | High | Parallel | Multiple small-medium files |
| Streaming | Low (constant) | Slower | Very large files (>100MB) |

---

## Dependencies

**New Dependencies Added:**
- `pdf2image` - PDF to image conversion for OCR
- All existing dependencies maintained

**System Requirements:**
- Tesseract OCR (for scanned documents)
- Poppler (for pdf2image)
- OpenCV libraries
- Python 3.7+

---

## Testing

Run comprehensive feature tests:
```bash
# Test PII detection
python -m pytest tests/test_pii.py

# Test individual loaders
python -c "from app.services.ingestion.text_loader import TextLoader; print('✅ TextLoader')"
python -c "from app.services.ingestion.multipage_loader import MultiPageDocumentLoader; print('✅ MultiPageLoader')"

# Test batch processor
python -c "from app.services.ingestion.batch_processor import BatchProcessor; print('✅ BatchProcessor')"

# Test streaming processor
python -c "from app.services.ingestion.streaming_processor import StreamingProcessor; print('✅ StreamingProcessor')"
```

---

## Next Steps / Future Enhancements

1. **Cloud Storage Integration**
   - S3, Azure Blob, Google Cloud Storage connectors
   - Direct cloud-to-cloud processing

2. **Advanced Video Processing**
   - Audio transcription + PII redaction
   - License plate blurring
   - Custom object detection

3. **API Endpoint**
   - REST API for integration
   - Webhook notifications
   - API key management

4. **Enhanced Reporting**
   - PDF reports with redaction statistics
   - Compliance audit trails
   - Custom redaction rules

---

## Contact & Support

For issues, feature requests, or contributions:
- GitHub: [RedactionTool Repository]
- Documentation: See README.md and DEPLOY.md
