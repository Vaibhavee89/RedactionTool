# Folder-Level Batch Redaction Guide

## Quick Test - You Just Did This! ✅

You successfully processed **5 files** with **87 PII entities detected and redacted**!

Results are in: `test_output/`

---

## How to Check Folder-Level Redaction

### Method 1: Command-Line (Fastest) ⚡

```bash
# 1. Create test files (already done!)
./test_batch_redaction.sh

# 2. Run batch redaction
python3 cli_batch.py batch -i test_input -o test_output

# 3. Check results
ls -lh test_output/
cat test_output/redacted_sample1.txt.txt
```

**What You'll See:**
```
🚀 Starting Batch Processing...
Processing: sample1.txt - 0%
Processing: sample2.txt - 20%
...
📊 BATCH PROCESSING RESULTS
Total Files: 5
✅ Processed: 5
❌ Failed: 0
Total PII Found: 87
```

---

### Method 2: Enhanced Web UI (Most Visual) 🖥️

```bash
# Start the enhanced UI
streamlit run app/ui/streamlit_app_enhanced.py
```

**Steps:**
1. Open browser to http://localhost:8501
2. In sidebar, select **"Batch Processing"** mode
3. Click **"Upload Multiple Files"**
4. Select all files from `test_input/` folder
5. Click **"Process All Files"** button
6. Watch progress bar
7. Download **ZIP** of all redacted files

**Features:**
- Visual progress tracking
- File type breakdown
- PII statistics
- Side-by-side comparison (for single files)
- Download all as ZIP

---

### Method 3: Python API (For Integration) 🐍

```python
from app.services.ingestion.batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor()

# Method A: Process entire directory
results = processor.process_directory(
    input_dir='test_input',
    output_dir='test_output',
    recursive=True,
    file_types=['.txt', '.pdf', '.docx']  # Optional filter
)

# Method B: Process specific file list
file_list = ['file1.pdf', 'file2.docx', 'file3.txt']
results = processor.process_file_list(
    file_paths=file_list,
    output_dir='test_output'
)

# Check results
print(f"Total files: {results['total_files']}")
print(f"Processed: {len(results['processed'])}")
print(f"Failed: {len(results['failed'])}")
print(f"PII found: {results['stats']['total_pii_found']}")

# Access individual results
for item in results['processed']:
    print(f"{item['filename']}: {item['pii_count']} PII entities")
```

---

## Advanced Options

### Recursive Directory Processing

Process all subdirectories:
```bash
python3 cli_batch.py batch -i /path/to/folder -o output -r
```

### Filter by File Type

Process only specific file types:
```bash
# Only PDFs
python3 cli_batch.py batch -i test_input -o output --types .pdf

# Multiple types
python3 cli_batch.py batch -i test_input -o output --types .pdf,.docx,.txt
```

### With Progress Tracking

```python
from app.services.ingestion.batch_processor import BatchProcessor

processor = BatchProcessor()

def progress_callback(filename, progress):
    print(f"Processing: {filename} - {int(progress*100)}%")

results = processor.process_directory(
    input_dir='test_input',
    output_dir='test_output',
    progress_callback=progress_callback
)
```

---

## What Gets Processed?

### Supported File Types in Batch Mode:

| Type | Extensions | Processing |
|------|-----------|------------|
| Text Documents | .txt, .pdf, .docx | Text extraction + PII detection + redaction |
| Images | .png, .jpg, .jpeg | OCR + PII detection + visual blurring |
| Videos | .mp4, .avi, .mov | Face detection + blurring |

### Mixed File Types

You can process all types in a single batch:
```bash
test_input/
├── document1.pdf
├── document2.docx
├── scan.png
├── photo.jpg
├── video.mp4
└── notes.txt

# Process all at once
python3 cli_batch.py batch -i test_input -o test_output
```

---

## Understanding Results

### Result Structure

```python
{
    'total_files': 10,
    'processed': [
        {
            'filename': 'doc1.pdf',
            'type': '.pdf',
            'pii_count': 5,
            'output_path': '/output/redacted_doc1.pdf.txt',
            'findings': [
                {'entity_type': 'PERSON', 'text': 'John Doe', 'source': 'NER'},
                {'entity_type': 'EMAIL', 'text': 'john@example.com', 'source': 'Regex'},
                ...
            ]
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

### Statistics Breakdown

After processing, you get:
- **Total files** processed
- **Success/failure** count
- **PII entities** found (by type)
- **File type** breakdown (documents/images/videos)
- **Individual file** results with PII details

---

## Real-World Examples

### Example 1: Process HR Documents Folder

```bash
# Setup
hr_docs/
├── employee_records/
│   ├── john_doe.pdf
│   ├── jane_smith.docx
│   └── payroll.txt
└── contracts/
    ├── contract_2024_01.pdf
    └── contract_2024_02.pdf

# Process recursively
python3 cli_batch.py batch \
  -i hr_docs \
  -o hr_docs_redacted \
  -r \
  --types .pdf,.docx,.txt

# Result: All files redacted, subdirectory structure preserved
```

### Example 2: Process Scanned Documents

```bash
# Setup
scans/
├── scan001.png
├── scan002.jpg
├── scan003.tiff
└── multi_page.pdf

# Process all
python3 cli_batch.py batch -i scans -o scans_redacted

# Result: OCR + PII detection + visual redaction
```

### Example 3: Process Mixed Media

```bash
# Setup
media_folder/
├── interview_transcript.txt
├── id_card_scan.png
├── interview_video.mp4
└── contract.pdf

# Process all types
python3 cli_batch.py batch -i media_folder -o redacted_media

# Result:
# - Text: PII redacted
# - Image: Faces + text blurred
# - Video: Faces blurred frame-by-frame
# - PDF: Text extracted and PII redacted
```

---

## Verification Steps

### 1. Check Output Files
```bash
ls -lh test_output/
```

### 2. Compare Original vs Redacted
```bash
echo "=== ORIGINAL ==="
cat test_input/sample1.txt

echo "=== REDACTED ==="
cat test_output/redacted_sample1.txt.txt
```

### 3. Verify PII Was Redacted
Look for:
- Names replaced with ████
- Emails replaced with ████
- Phone numbers redacted
- SSNs blocked out
- Addresses redacted

### 4. Check Statistics
```bash
# CLI shows summary automatically
# or use Python API to access detailed stats
```

---

## Common Use Cases

### Use Case 1: Compliance Audit
**Scenario**: Need to redact all customer data before sharing with auditors

```bash
python3 cli_batch.py batch \
  -i customer_data/ \
  -o auditor_safe_data/ \
  -r
```

### Use Case 2: Data Anonymization for Testing
**Scenario**: Create test data from production data

```bash
python3 cli_batch.py batch \
  -i production_backups/ \
  -o test_data/ \
  --types .sql,.json,.csv
```

### Use Case 3: GDPR Compliance
**Scenario**: Remove personal data from archived documents

```bash
python3 cli_batch.py batch \
  -i archives/2020-2024/ \
  -o gdpr_compliant/ \
  -r \
  --types .pdf,.docx,.txt
```

---

## Performance Tips

### Batch Size Recommendations

| File Count | Total Size | Method | Expected Time |
|------------|-----------|--------|---------------|
| 1-10 files | < 100 MB | Batch | 1-2 minutes |
| 10-50 files | 100 MB - 1 GB | Batch | 5-10 minutes |
| 50-100 files | 1-5 GB | Batch | 15-30 minutes |
| Large single file | > 100 MB | Streaming | Variable |

### Optimize Processing

1. **Filter by type** to process only what you need
2. **Use recursive mode** only when needed
3. **For huge files** (>100MB each), use streaming mode instead
4. **Process videos separately** (they're CPU-intensive)

---

## Troubleshooting

### Issue: "No files processed"
**Solution**: Check input directory path and file extensions

### Issue: "Memory error"
**Solution**:
- Process smaller batches
- Use streaming mode for large files
- Filter by file type to reduce load

### Issue: "Some files failed"
**Solution**: Check the failed files list in results
```python
for failed in results['failed']:
    print(f"Failed: {failed['file']} - {failed['error']}")
```

### Issue: "OCR not working"
**Solution**: Install Tesseract
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr
```

---

## Quick Reference

### CLI Commands

```bash
# Basic batch
python3 cli_batch.py batch -i INPUT -o OUTPUT

# Recursive
python3 cli_batch.py batch -i INPUT -o OUTPUT -r

# Filter types
python3 cli_batch.py batch -i INPUT -o OUTPUT --types .pdf,.docx

# Help
python3 cli_batch.py batch --help
```

### Python API

```python
from app.services.ingestion.batch_processor import BatchProcessor

processor = BatchProcessor()
results = processor.process_directory('input/', 'output/')
```

### Web UI

```bash
streamlit run app/ui/streamlit_app_enhanced.py
# Then select "Batch Processing" mode
```

---

## What You Just Tested ✅

Running `./test_batch_redaction.sh` and then the batch command:

1. ✅ Created 5 test files with PII
2. ✅ Processed all files in one batch
3. ✅ Detected 87 PII entities
4. ✅ Redacted all sensitive information
5. ✅ Saved output to test_output/
6. ✅ Generated statistics report

**You can now:**
- Add more files to `test_input/`
- Run batch processing again
- Compare original vs redacted files
- Check the statistics

---

## Next Steps

1. **Try Web UI**: `streamlit run app/ui/streamlit_app_enhanced.py`
2. **Process your own files**: Replace test_input with your folder
3. **Integrate into workflow**: Use Python API in your scripts
4. **Automate**: Set up cron jobs with CLI commands

---

## Summary

**Folder-level redaction is working perfectly!**

You have **3 ways** to use it:
1. ⚡ CLI (fastest, scriptable)
2. 🖥️ Web UI (most visual, easiest)
3. 🐍 Python API (most flexible, for integration)

All methods support:
- Multiple file types
- Recursive directories
- Progress tracking
- Detailed statistics
- Error handling

**Test Results**: ✅ 5 files, 87 PII entities redacted successfully!
