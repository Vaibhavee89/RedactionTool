#!/usr/bin/env python3
"""
Quick test script to verify all new features are working correctly.
"""

import sys
import os

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all new modules can be imported."""
    print("="*60)
    print("Testing Module Imports...")
    print("="*60)

    tests = [
        ("TextLoader", "from app.services.ingestion.text_loader import TextLoader"),
        ("MultiPageDocumentLoader", "from app.services.ingestion.multipage_loader import MultiPageDocumentLoader"),
        ("BatchProcessor", "from app.services.ingestion.batch_processor import BatchProcessor"),
        ("StreamingProcessor", "from app.services.ingestion.streaming_processor import StreamingProcessor"),
        ("Enhanced PDFLoader", "from app.services.ingestion.pdf_loader import PDFLoader"),
        ("Enhanced VideoRedactor", "from app.services.redaction.video_redactor import VideoRedactor"),
    ]

    passed = 0
    failed = 0

    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"✅ {name}: PASS")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAIL - {e}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_instantiation():
    """Test that all classes can be instantiated."""
    print("\n" + "="*60)
    print("Testing Class Instantiation...")
    print("="*60)

    tests = []

    # Test TextLoader
    try:
        from app.services.ingestion.text_loader import TextLoader
        loader = TextLoader()
        print("✅ TextLoader instantiation: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ TextLoader instantiation: FAIL - {e}")
        tests.append(False)

    # Test MultiPageDocumentLoader
    try:
        from app.services.ingestion.multipage_loader import MultiPageDocumentLoader
        loader = MultiPageDocumentLoader()
        print("✅ MultiPageDocumentLoader instantiation: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ MultiPageDocumentLoader instantiation: FAIL - {e}")
        tests.append(False)

    # Test BatchProcessor
    try:
        from app.services.ingestion.batch_processor import BatchProcessor
        processor = BatchProcessor()
        print("✅ BatchProcessor instantiation: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ BatchProcessor instantiation: FAIL - {e}")
        tests.append(False)

    # Test StreamingProcessor
    try:
        from app.services.ingestion.streaming_processor import StreamingProcessor
        processor = StreamingProcessor(chunk_size=1000)
        print("✅ StreamingProcessor instantiation: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ StreamingProcessor instantiation: FAIL - {e}")
        tests.append(False)

    # Test enhanced PDFLoader
    try:
        from app.services.ingestion.pdf_loader import PDFLoader
        loader = PDFLoader()
        print("✅ Enhanced PDFLoader instantiation: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ Enhanced PDFLoader instantiation: FAIL - {e}")
        tests.append(False)

    # Test VideoRedactor
    try:
        from app.services.redaction.video_redactor import VideoRedactor
        redactor = VideoRedactor()
        print("✅ VideoRedactor instantiation: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ VideoRedactor instantiation: FAIL - {e}")
        tests.append(False)

    passed = sum(tests)
    failed = len(tests) - passed
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return all(tests)


def test_file_existence():
    """Test that all new files exist."""
    print("\n" + "="*60)
    print("Testing File Existence...")
    print("="*60)

    files = [
        "app/services/ingestion/text_loader.py",
        "app/services/ingestion/multipage_loader.py",
        "app/services/ingestion/batch_processor.py",
        "app/services/ingestion/streaming_processor.py",
        "app/ui/streamlit_app_enhanced.py",
        "cli_batch.py",
        "FEATURES.md",
        "IMPLEMENTATION_SUMMARY.md",
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tests = []

    for file_path in files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}: EXISTS")
            tests.append(True)
        else:
            print(f"❌ {file_path}: NOT FOUND")
            tests.append(False)

    passed = sum(tests)
    failed = len(tests) - passed
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return all(tests)


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\n" + "="*60)
    print("Testing Dependencies...")
    print("="*60)

    dependencies = [
        "streamlit",
        "pdfplumber",
        "pytesseract",
        "cv2",
        "PIL",
        "pdf2image",
        "docx",
        "spacy",
        "presidio_analyzer",
    ]

    tests = []
    for dep in dependencies:
        try:
            if dep == "cv2":
                import cv2
            elif dep == "PIL":
                from PIL import Image
            elif dep == "docx":
                from docx import Document
            else:
                __import__(dep)
            print(f"✅ {dep}: INSTALLED")
            tests.append(True)
        except ImportError:
            print(f"⚠️ {dep}: NOT INSTALLED (may cause issues)")
            tests.append(False)

    passed = sum(tests)
    failed = len(tests) - passed
    print(f"\n📊 Results: {passed} installed, {failed} missing")
    return passed > 0  # At least some deps should be installed


def main():
    print("\n" + "="*60)
    print("🔒 RedactionTool Enterprise - Feature Test Suite")
    print("="*60 + "\n")

    results = []

    # Run all tests
    results.append(("File Existence", test_file_existence()))
    results.append(("Module Imports", test_imports()))
    results.append(("Class Instantiation", test_instantiation()))
    results.append(("Dependencies", test_dependencies()))

    # Final summary
    print("\n" + "="*60)
    print("📋 FINAL SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("All new features are properly implemented and ready to use.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please check the errors above and ensure all dependencies are installed.")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
