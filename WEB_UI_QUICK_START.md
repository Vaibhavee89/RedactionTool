# Web UI - Quick Start Guide

## 🚀 Launch the Application

```bash
cd /Users/vaibhavee/project/redaction_tool/RedactionTool
streamlit run app/ui/streamlit_app_enhanced.py
```

The app will open in your browser at `http://localhost:8501`

---

## ✅ All 6 Requested Features - Status

### 1. ✅ Drag-and-Drop Upload
**Location**: Main page, file uploader widget
**How to use**:
- Drag file from desktop into upload area
- Or click to browse files
- Supports: PDF, DOCX, TXT, PNG, JPG, TIFF, MP4, AVI, MOV

### 2. ✅ Policy Selector
**Location**: Left sidebar → "📋 Policy Configuration"
**How to use**:
- Select from dropdown: None (Default), india_finance, gdpr_basic, hipaa_like
- Click expandable "📄 View Policy Details" to see rules
- Policy automatically applies to detected entities

### 3. ✅ Live Preview (Before/After)
**Location**: Main content area, two-column layout
**How to use**:
- Left column shows original content
- Right column shows redacted content (after clicking "Analyze & Redact")
- Scroll both columns independently

### 4. ⭐ Entity Highlight Overlays (NEW)
**Location**: Between original and redacted preview
**Section**: "🎯 Entity Detection Overlay"
**How to use**:
- Appears automatically after processing
- Color-coded highlights show detected PII
- Hover over highlighted text to see:
  - Entity type (PERSON, EMAIL, PHONE, etc.)
  - Confidence score (0.00 - 1.00)

**Color Legend:**
- 🟥 Pink = PERSON
- 🟦 Blue = EMAIL
- 🟩 Green = PHONE
- 🟨 Gold = PAN
- 🟧 Salmon = AADHAAR
- 🟪 Plum = CREDIT_CARD
- 🟫 Khaki = ADDRESS

### 5. ✅ Download Redacted Files
**Location**: Below preview, left button
**How to use**:
- Click "📥 Download Redacted Text" (primary blue button)
- File downloads as: `redacted_{original_filename}.txt`
- For batch mode: Downloads as ZIP file with all redacted files

### 6. ⭐ Download Audit Report (NEW)
**Location**: Below preview, right button
**How to use**:
- Click "📋 Download Audit Report" (secondary gray button)
- File downloads as: `audit_{original_filename}.json`
- Contains:
  - Metadata (filename, timestamp, tool version)
  - Statistics (total entities, breakdown by type)
  - All detected entities with details
  - Text snippets (original & redacted)

---

## 🎬 Step-by-Step Demo

### Demo 1: Simple Document Redaction

1. **Launch app**: `streamlit run app/ui/streamlit_app_enhanced.py`

2. **Select policy**: Sidebar → "Select Redaction Policy" → "india_finance"

3. **Upload file**: Drag `test_cli_data/sample1.txt` into upload area

4. **Click**: "🔍 Analyze & Redact" button

5. **View results**:
   - **Original**: Left column shows unredacted text
   - **Highlights**: Color-coded overlay shows detected PII
     - Hover over "Rajesh Kumar Sharma" → Pink (PERSON, 0.95)
     - Hover over "ABCDE1234F" → Gold (PAN, 0.98)
     - Hover over email → Blue (EMAIL, 0.99)
   - **Redacted**: Right column shows ████ blocks
   - **Statistics**: Shows 10 entities detected, 5 types, avg confidence 0.93

6. **Download**:
   - Click "📥 Download Redacted Text" → Gets `redacted_sample1.txt`
   - Click "📋 Download Audit Report" → Gets `audit_sample1.json`

### Demo 2: Batch Processing

1. **Select mode**: Sidebar → "Processing Mode" → "Batch Processing"

2. **Upload files**: Select multiple files (3-5 different types)

3. **View list**: Expand "📋 View File List" to see all files

4. **Click**: "🚀 Process All Files"

5. **Watch progress**: Progress bar shows current file being processed

6. **View results**:
   - Metrics: Total Files, Processed, Failed, Total PII Found
   - Breakdown: Text Documents, Images, Videos

7. **Download**: Click "📥 Download All Redacted Files (ZIP)" → Gets `redacted_batch.zip`

---

## 📸 What You'll See

### Sidebar Configuration Panel
```
⚙️ Configuration

Processing Mode:
○ Single File  ← selected
○ Batch Processing
○ Streaming (Large Files)

───────────────────────

📋 Policy Configuration
Select Redaction Policy: [india_finance ▼]
✅ Using policy: india_finance
▶ 📄 View Policy Details

───────────────────────

📋 Supported Features
▶ Text Inputs: ...
▶ Image Inputs: ...
▶ Video Inputs: ...
```

### Main Content Area (After Processing)
```
📄 Document Redaction

┌─────────────────────┬─────────────────────┐
│ Original Content    │ Redacted Content    │
│                     │                     │
│ Personal Info Form  │ Personal Info Form  │
│                     │                     │
│ Name: Rajesh Kumar  │ Name: ██████████    │
│ PAN: ABCDE1234F     │ PAN: ██████████     │
│ Email: raj@test.com │ Email: ████████     │
└─────────────────────┴─────────────────────┘

🎯 Entity Detection Overlay
┌─────────────────────────────────────────────┐
│ Personal Information Form                    │
│                                              │
│ Name: [Rajesh Kumar] ← Hover: PERSON (0.95) │
│ PAN: [ABCDE1234F]   ← Hover: PAN (0.98)     │
│ Email: [raj@test.com] ← Hover: EMAIL (0.99) │
└─────────────────────────────────────────────┘
💡 Hover over highlighted text to see details

📊 Detection Report
┌──────────────┬─────────────────┬────────────┬────────────┐
│ Type         │ Text            │ Source     │ Confidence │
├──────────────┼─────────────────┼────────────┼────────────┤
│ PERSON       │ Rajesh Kumar    │ presidio   │ 0.95       │
│ PAN          │ ABCDE1234F      │ regex      │ 0.98       │
│ EMAIL        │ raj@test.com    │ presidio   │ 0.99       │
└──────────────┴─────────────────┴────────────┴────────────┘

┌──────────────┬──────────────┬──────────────┐
│ Total PII    │ Entity Types │ Avg Conf     │
│     10       │      5       │    0.93      │
└──────────────┴──────────────┴──────────────┘

[📥 Download Redacted Text]  [📋 Download Audit Report]
```

---

## 🧪 Testing Checklist

Use this checklist to verify all features work:

### Feature 1: Drag-and-Drop Upload
- [ ] Drag file into upload area works
- [ ] Click to browse files works
- [ ] File name appears after upload
- [ ] All supported file types accepted
- [ ] Error message for unsupported types

### Feature 2: Policy Selector
- [ ] Dropdown shows all available policies
- [ ] "None (Default)" option works
- [ ] Policy name displays with ✅ when selected
- [ ] Policy details expand/collapse
- [ ] Policy JSON is valid and readable

### Feature 3: Live Preview
- [ ] Original content appears in left column
- [ ] Redacted content appears in right column
- [ ] Both columns are scrollable
- [ ] Text areas have 400px height
- [ ] Content matches expected format

### Feature 4: Entity Highlight Overlays
- [ ] Overlay section appears after processing
- [ ] All detected entities are highlighted
- [ ] Colors match entity types (pink=PERSON, blue=EMAIL, etc.)
- [ ] Hover shows tooltip with entity type
- [ ] Hover shows tooltip with confidence score
- [ ] Tooltip format: "PERSON (confidence: 0.95)"
- [ ] Caption appears: "💡 Hover over highlighted text..."

### Feature 5: Download Redacted Files
- [ ] Download button appears
- [ ] Button is styled as primary (blue)
- [ ] Click downloads file
- [ ] Filename format: `redacted_{original_name}.txt`
- [ ] File content matches redacted preview
- [ ] Batch mode downloads ZIP file

### Feature 6: Download Audit Report
- [ ] Download button appears
- [ ] Button is styled as secondary (gray)
- [ ] Click downloads JSON file
- [ ] Filename format: `audit_{original_name}.json`
- [ ] JSON is valid (use `python3 -m json.tool audit_*.json`)
- [ ] Contains metadata section
- [ ] Contains statistics section
- [ ] Contains detected_entities array
- [ ] Contains text_samples section
- [ ] Timestamp is ISO 8601 format

---

## 🎯 Sample Test Files

Use these files to test each feature:

### Text Document Testing
```bash
# Test with sample files from CLI testing
app/ui/streamlit_app_enhanced.py
↓
Upload: test_cli_data/sample1.txt
Expected entities: 10 (PERSON, PAN, AADHAAR, PHONE, EMAIL, ADDRESS)
```

### Image Testing
```bash
# Create test image with text (if available)
# Or use any image with visible text
Expected: OCR extracts text, PII detected from extracted text
```

### Policy Testing
```bash
# Test each policy file
1. Select "india_finance" → Should detect PAN, AADHAAR, PHONE
2. Select "gdpr_basic" → Should detect PERSON, EMAIL, PHONE
3. Select "hipaa_like" → Should detect medical entities
```

---

## 🐛 Troubleshooting

### Issue: No policies show in dropdown
**Solution**: Check `policies/` directory exists with `.yaml` files

### Issue: Entity highlights don't appear
**Solution**: Check browser allows `unsafe_allow_html` in st.markdown

### Issue: Download buttons don't work
**Solution**: Check browser doesn't block downloads, clear cache

### Issue: Colors don't match entity types
**Solution**: Check `highlight_entities_in_text()` function in code

### Issue: Audit report missing fields
**Solution**: Check `generate_audit_report()` function returns complete dict

---

## 📚 Related Documentation

- **Full Implementation Details**: `WEB_UI_IMPLEMENTATION_SUMMARY.md`
- **CLI Interface**: `CLI_INTERFACE_GUIDE.md`
- **Testing Guide**: `TESTING_CICD_GUIDE.md`
- **Evaluation**: `EVALUATION_GUIDE.md`

---

## 🎉 Success Criteria

Your implementation is successful if you can:

1. ✅ Upload a file via drag-and-drop
2. ✅ Select a policy from dropdown
3. ✅ See original and redacted content side-by-side
4. ✅ See color-coded entity highlights with hover tooltips
5. ✅ Download redacted file
6. ✅ Download audit report in JSON format

All 6 features = **🚀 Production Ready!**

---

**Last Updated**: 2026-02-05
**Version**: v2.0
**Status**: ✅ All Features Implemented
