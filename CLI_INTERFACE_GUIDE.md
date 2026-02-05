# CLI Interface Guide

## 🎯 Overview

The PII Redaction Tool CLI provides powerful automation and scaling capabilities for batch processing of documents with configurable output formats, audit logging, and pipeline integration.

## ✅ What's Implemented

### 1. Command Line Interface ✅
- Simple syntax: `redact input_dir/ output_dir/`
- Configurable options (policy, mode, confidence, formats)
- Help documentation built-in

### 2. Batch Processing ✅
- Process entire directories recursively
- Support for multiple file formats (.txt, .text, .md, .json, .csv, .log)
- Parallel-ready architecture
- Progress tracking and statistics

### 3. Configurable Output Formats ✅
- Text (`.txt`)
- JSON (`.json`)
- HTML (`.html`)
- Markdown (`.md`)
- Multiple formats simultaneously

### 4. JSON Summary Output ✅
- Detailed audit logs
- Processing statistics
- Per-file results
- Error tracking
- Timing information

### 5. Exit Codes for Pipeline Integration ✅
- 0: Success
- 1: General error
- 2: Input error
- 3: Output error
- 4: Policy error
- 5: Processing error

---

## 🚀 Quick Start

### Installation

#### Option 1: Direct Use
```bash
python3 redact_cli.py input_dir/ output_dir/
```

#### Option 2: Install as Command
```bash
# Run setup script
./setup_cli.sh

# Now use from anywhere
redact input_dir/ output_dir/
```

### Basic Usage

```bash
# Process directory
redact test_data/ output/

# Process single file
redact document.txt output/

# With policy
redact input/ output/ --policy policies/india_finance.yaml

# With custom mode
redact input/ output/ --mode mask --confidence 0.8

# Multiple output formats
redact input/ output/ --formats text json html

# With audit log
redact input/ output/ --log audit.json

# Dry run (no files written)
redact input/ output/ --dry-run --verbose
```

---

## 📖 Command Reference

### Positional Arguments

**`input`**
- Input directory or file path
- Supported formats: .txt, .text, .md, .json, .csv, .log
- Processes recursively if directory

**`output`**
- Output directory path
- Created automatically if doesn't exist
- Maintains directory structure from input

### Optional Arguments

**`-p, --policy POLICY`**
- Path to policy YAML file
- Example: `--policy india_finance.yaml`
- Uses policy rules for redaction decisions
- Optional: If not provided, all PII is redacted

**`-m, --mode MODE`**
- Redaction mode
- Choices: `block`, `mask`, `partial_mask`, `label`, `hash`, `tokenize`, `allow`
- Default: `block`
- Example: `--mode mask`

**`-c, --confidence CONFIDENCE`**
- Confidence threshold (0.0 to 1.0)
- Default: `0.5`
- Only entities above this threshold are redacted
- Example: `--confidence 0.8`

**`-f, --formats FORMAT [FORMAT ...]`**
- Output format(s)
- Choices: `text`, `json`, `html`, `markdown`
- Default: `text`
- Can specify multiple: `--formats text json html`

**`-l, --log LOG`**
- Audit log file path (JSON)
- Contains detailed processing summary
- Example: `--log audit.json`

**`-v, --verbose`**
- Enable verbose logging
- Shows detailed processing information
- Useful for debugging

**`-d, --dry-run`**
- Dry run mode
- No output files written
- Useful for testing

**`--version`**
- Show version and exit

**`-h, --help`**
- Show help message and exit

---

## 💡 Usage Examples

### Example 1: Basic Redaction

```bash
redact documents/ redacted_output/
```

**What it does:**
- Processes all files in `documents/` directory
- Detects PII with default settings
- Redacts using block mode (████)
- Outputs to `redacted_output/` as text files

### Example 2: Financial Data with Policy

```bash
redact financial_data/ output/ \
  --policy policies/india_finance.yaml \
  --confidence 0.7 \
  --log financial_audit.json
```

**What it does:**
- Uses India Finance policy (PAN, Aadhaar, etc.)
- Only redacts entities with 70%+ confidence
- Generates audit log in JSON format

### Example 3: Multiple Output Formats

```bash
redact customer_records/ processed/ \
  --formats text json html \
  --mode mask \
  --log customer_audit.json
```

**What it does:**
- Generates 3 formats for each file (.txt, .json, .html)
- Uses masking mode (shows last N characters)
- Creates comprehensive audit log

### Example 4: High Security Redaction

```bash
redact sensitive/ redacted/ \
  --policy policies/hipaa_like.yaml \
  --mode block \
  --confidence 0.3 \
  --formats text json \
  --log hipaa_audit.json \
  --verbose
```

**What it does:**
- Uses HIPAA-like policy for healthcare data
- Low confidence threshold (catches more PII)
- Full blocking redaction
- Verbose logging for compliance tracking

### Example 5: Testing and Validation

```bash
redact test_samples/ output_test/ \
  --dry-run \
  --verbose \
  --log test_report.json
```

**What it does:**
- Runs detection without writing files
- Shows detailed processing information
- Generates report for validation

### Example 6: Pipeline Integration

```bash
#!/bin/bash
# Redaction pipeline script

# Run redaction
redact input/ output/ \
  --policy company_policy.yaml \
  --formats text json \
  --log audit.json

# Check exit code
if [ $? -eq 0 ]; then
    echo "✓ Redaction successful"
    # Continue pipeline
    ./next_step.sh output/
else
    echo "✗ Redaction failed"
    exit 1
fi
```

---

## 📊 Output Formats

### Text Format (`.txt`)

Simple redacted text file:

```
Personal Information Form

Name: ███████████████████
PAN: ██████████
Email: █████████████████
```

### JSON Format (`.json`)

Structured output with metadata:

```json
{
  "original_file": "sample.txt",
  "redacted_text": "Name: ███████...",
  "entities": [
    {
      "entity_type": "PERSON",
      "start": 6,
      "end": 23,
      "text": "Rajesh Kumar Sharma",
      "confidence": 0.95
    }
  ],
  "metadata": {
    "entities_detected": 10,
    "entities_redacted": 10,
    "redaction_mode": "block",
    "confidence_threshold": 0.5,
    "timestamp": "2026-02-05T19:24:28.786279"
  }
}
```

### HTML Format (`.html`)

Formatted HTML document with statistics:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Redacted: sample.txt</title>
    <style>/* styling */</style>
</head>
<body>
    <div class="header">
        <h1>Redacted Document</h1>
        <p><strong>Original:</strong> sample.txt</p>
    </div>
    <div class="stats">
        <h2>Redaction Statistics</h2>
        <ul>
            <li>Entities detected: 10</li>
            <li>Redaction mode: block</li>
        </ul>
    </div>
    <div class="content">
        <pre><!-- redacted content --></pre>
    </div>
</body>
</html>
```

### Markdown Format (`.md`)

Markdown document with detailed information:

```markdown
# Redacted Document

**Original:** sample.txt
**Redacted:** 2026-02-05 19:24:28

## Redaction Statistics

- Entities detected: 10
- Redaction mode: block

## Redacted Content

\```
Name: ███████████████████
PAN: ██████████
\```

## Detected Entities

1. **PERSON** - Confidence: 0.95
2. **PAN** - Confidence: 1.00
```

---

## 📋 Audit Log (JSON Summary)

The audit log (`--log audit.json`) contains comprehensive processing information:

```json
{
  "summary": {
    "total_files": 3,
    "processed_files": 3,
    "failed_files": 0,
    "skipped_files": 0,
    "total_entities": 23,
    "redacted_entities": 23,
    "success_rate": 100.0
  },
  "configuration": {
    "input_path": "test_data",
    "output_path": "output",
    "policy_file": "india_finance.yaml",
    "redaction_mode": "block",
    "confidence_threshold": 0.5,
    "output_formats": ["text", "json"],
    "dry_run": false
  },
  "timing": {
    "start_time": "2026-02-05T19:24:28.600976",
    "end_time": "2026-02-05T19:24:28.786279"
  },
  "errors": [],
  "results": [
    {
      "file": "test_data/sample1.txt",
      "status": "success",
      "entities_detected": 10,
      "entities_redacted": 10,
      "errors": []
    }
  ]
}
```

**Fields:**

- **summary**: Aggregate statistics
  - `total_files`: Total files found
  - `processed_files`: Successfully processed
  - `failed_files`: Failed to process
  - `skipped_files`: Skipped (empty, errors)
  - `total_entities`: Total PII entities detected
  - `redacted_entities`: Entities actually redacted
  - `success_rate`: Percentage (0-100)

- **configuration**: Run configuration
  - `input_path`, `output_path`: Paths used
  - `policy_file`: Policy applied (if any)
  - `redaction_mode`: Mode used
  - `confidence_threshold`: Threshold applied
  - `output_formats`: Formats generated
  - `dry_run`: Whether dry run

- **timing**: Timestamp information
  - `start_time`: When processing started
  - `end_time`: When processing finished

- **errors**: List of error messages

- **results**: Per-file results
  - `file`: File path
  - `status`: success/error/skipped/no_pii
  - `entities_detected`: Count for this file
  - `entities_redacted`: Count redacted
  - `errors`: File-specific errors

---

## 🔧 Exit Codes

The CLI returns specific exit codes for pipeline integration:

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | All files processed successfully |
| 1 | General Error | Unexpected error occurred |
| 2 | Input Error | Input path invalid or inaccessible |
| 3 | Output Error | Cannot create output directory |
| 4 | Policy Error | Policy file invalid or not found |
| 5 | Processing Error | Some files failed to process |

**Example Usage in Script:**

```bash
#!/bin/bash

redact input/ output/ --log audit.json

EXIT_CODE=$?

case $EXIT_CODE in
    0)
        echo "✓ Success"
        ;;
    2)
        echo "✗ Input error - check input path"
        ;;
    5)
        echo "⚠ Some files failed - check audit.json"
        ;;
    *)
        echo "✗ Error code: $EXIT_CODE"
        ;;
esac

exit $EXIT_CODE
```

---

## 🔄 Batch Processing

### Directory Structure

Input directory structure is maintained in output:

**Input:**
```
documents/
├── 2024/
│   ├── january/
│   │   ├── report1.txt
│   │   └── report2.txt
│   └── february/
│       └── report3.txt
└── archives/
    └── old_report.txt
```

**Output (with `--formats text json`):**
```
output/
├── 2024/
│   ├── january/
│   │   ├── report1.txt
│   │   ├── report1.json
│   │   ├── report2.txt
│   │   └── report2.json
│   └── february/
│       ├── report3.txt
│       └── report3.json
└── archives/
    ├── old_report.txt
    └── old_report.json
```

### Supported File Types

- `.txt` - Plain text files
- `.text` - Text files
- `.md` - Markdown files
- `.json` - JSON files
- `.csv` - CSV files
- `.log` - Log files

**Note:** Binary files are skipped automatically.

---

## 🎯 Best Practices

### 1. Start with Dry Run

Test your configuration before processing:

```bash
redact input/ output/ \
  --dry-run \
  --verbose \
  --log test_report.json
```

Review `test_report.json` to verify detection accuracy.

### 2. Use Appropriate Confidence Thresholds

- **High Security (0.3-0.5)**: Catch more PII, more false positives
- **Balanced (0.5-0.7)**: Good accuracy, fewer false positives
- **High Precision (0.7-0.9)**: Miss some PII, very few false positives

### 3. Choose Right Output Format

- **Text**: For human review or simple archival
- **JSON**: For programmatic processing or analysis
- **HTML**: For reports and presentations
- **Markdown**: For documentation systems

### 4. Always Use Audit Logs in Production

```bash
redact production_data/ redacted/ \
  --policy company_policy.yaml \
  --log "logs/audit_$(date +%Y%m%d_%H%M%S).json" \
  --verbose
```

### 5. Monitor Exit Codes

```bash
redact input/ output/ --log audit.json
if [ $? -ne 0 ]; then
    # Send alert
    mail -s "Redaction Failed" admin@company.com < audit.json
fi
```

---

## 🐛 Troubleshooting

### Issue: "Input path does not exist"

**Solution:**
```bash
# Check path exists
ls input_dir/

# Use absolute path
redact /full/path/to/input/ output/
```

### Issue: "Cannot create output directory"

**Solution:**
```bash
# Check permissions
ls -ld output_parent_directory/

# Create manually
mkdir -p output/
redact input/ output/
```

### Issue: "No files found to process"

**Solution:**
```bash
# Check file extensions
ls input_dir/*.txt

# Supported extensions: .txt, .text, .md, .json, .csv, .log
```

### Issue: "Policy file does not exist"

**Solution:**
```bash
# Check policy path
ls policies/india_finance.yaml

# Use absolute path
redact input/ output/ --policy /full/path/to/policy.yaml
```

### Issue: "Some files failed to process"

**Solution:**
```bash
# Run with verbose to see errors
redact input/ output/ --verbose

# Check audit log for details
cat audit.json | python3 -m json.tool | grep error
```

---

## 🔗 Integration Examples

### Integration with CI/CD Pipeline

**GitHub Actions:**
```yaml
name: Data Redaction

on:
  push:
    paths:
      - 'sensitive_data/**'

jobs:
  redact:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Redact sensitive data
        run: |
          python3 redact_cli.py \
            sensitive_data/ \
            redacted_output/ \
            --policy policies/company.yaml \
            --formats text json \
            --log audit.json

      - name: Upload audit log
        uses: actions/upload-artifact@v3
        with:
          name: audit-log
          path: audit.json
```

### Integration with Cron Job

```bash
#!/bin/bash
# Daily redaction job

# Configuration
INPUT_DIR="/data/incoming"
OUTPUT_DIR="/data/redacted/$(date +%Y/%m/%d)"
POLICY="policies/gdpr_basic.yaml"
LOG_DIR="/var/log/redaction"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Run redaction
/usr/local/bin/redact \
  "$INPUT_DIR" \
  "$OUTPUT_DIR" \
  --policy "$POLICY" \
  --formats text json \
  --log "$LOG_DIR/audit_$(date +%Y%m%d).json" \
  --verbose

# Check exit code
if [ $? -eq 0 ]; then
    echo "$(date): Redaction successful" >> "$LOG_DIR/status.log"
    # Archive or delete input files
    mv "$INPUT_DIR"/* "$INPUT_DIR/processed/"
else
    echo "$(date): Redaction failed" >> "$LOG_DIR/status.log"
    # Send alert
fi
```

### Integration with Data Pipeline

```python
import subprocess
import json
import sys

def redact_batch(input_dir, output_dir, policy="default"):
    """Run CLI redaction as part of data pipeline."""

    cmd = [
        "python3", "redact_cli.py",
        input_dir,
        output_dir,
        "--policy", f"policies/{policy}.yaml",
        "--formats", "text", "json",
        "--log", "pipeline_audit.json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Load audit log
    with open("pipeline_audit.json") as f:
        audit = json.load(f)

    # Check results
    if result.returncode == 0:
        print(f"✓ Redacted {audit['summary']['processed_files']} files")
        return True
    else:
        print(f"✗ Redaction failed: {audit['errors']}")
        return False

# Use in pipeline
if redact_batch("raw_data/", "clean_data/", policy="india_finance"):
    # Continue pipeline
    process_clean_data()
else:
    # Handle failure
    sys.exit(1)
```

---

## 📚 Advanced Usage

### Custom Redaction Modes

```bash
# Full blocking (default)
redact input/ output/ --mode block

# Partial masking (show last 4 chars)
redact input/ output/ --mode partial_mask

# Label replacement (<PAN>, <EMAIL>)
redact input/ output/ --mode label

# Hashing (SHA256)
redact input/ output/ --mode hash

# Tokenization (reversible with key)
redact input/ output/ --mode tokenize
```

### Multi-Stage Processing

```bash
# Stage 1: High confidence only
redact input/ stage1/ \
  --confidence 0.8 \
  --log stage1_audit.json

# Stage 2: Review and process medium confidence
redact stage1/ stage2/ \
  --confidence 0.5 \
  --policy reviewed_policy.yaml \
  --log stage2_audit.json

# Stage 3: Final validation
redact stage2/ final/ \
  --dry-run \
  --verbose \
  --log validation.json
```

---

## 🎉 Summary

The CLI Interface provides:

✅ **Simple Command Line Syntax** - Easy to use and integrate
✅ **Batch Processing** - Handle multiple files automatically
✅ **Configurable Output Formats** - Text, JSON, HTML, Markdown
✅ **JSON Summary Output** - Comprehensive audit logs
✅ **Exit Codes** - Pipeline integration ready
✅ **Policy Support** - Use predefined or custom policies
✅ **Dry Run Mode** - Test before processing
✅ **Progress Tracking** - Detailed statistics

**System Status:** Production Ready 🚀

**Next Steps:**
1. Run `./setup_cli.sh` to install command globally
2. Try examples with test data
3. Create custom policies for your use case
4. Integrate into your automation pipeline

For more information, see:
- Policy creation: `EVALUATION_GUIDE.md`
- Testing: `TESTING_CICD_GUIDE.md`
- Full documentation: Project README
