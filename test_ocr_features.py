#!/usr/bin/env python3
"""
Test script for OCR and preprocessing features.
"""

import sys
import os
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_test_images():
    """Create test images for OCR testing."""
    print("Creating test images...")

    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)

    # 1. Create a simple text image
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)

    try:
        # Try to use a larger font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()

    draw.text((50, 50), "John Doe - john@email.com", fill='black', font=font)
    draw.text((50, 100), "Phone: 555-123-4567", fill='black', font=font)
    img.save('test_images/simple_text.png')
    print("✅ Created: test_images/simple_text.png")

    # 2. Create a skewed image
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Skewed Document Text", fill='black', font=font)
    draw.text((50, 100), "Email: alice@example.com", fill='black', font=font)

    # Rotate to simulate skew
    img_rotated = img.rotate(5, fillcolor='white', expand=True)
    img_rotated.save('test_images/skewed_text.png')
    print("✅ Created: test_images/skewed_text.png")

    # 3. Create a noisy image
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Noisy Image Test", fill='black', font=font)
    draw.text((50, 100), "SSN: 123-45-6789", fill='black', font=font)

    # Add noise
    img_array = np.array(img)
    noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
    noisy_img = cv2.add(img_array, noise)
    Image.fromarray(noisy_img).save('test_images/noisy_text.png')
    print("✅ Created: test_images/noisy_text.png")

    # 4. Create a low contrast image
    img = Image.new('RGB', (800, 200), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Low Contrast Document", fill=(100, 100, 100), font=font)
    draw.text((50, 100), "Phone: 555-999-8888", fill=(100, 100, 100), font=font)
    img.save('test_images/low_contrast.png')
    print("✅ Created: test_images/low_contrast.png")

    # 5. Create a simple table-like structure
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)

    # Draw table structure
    draw.rectangle([50, 50, 750, 100], outline='black', width=2)
    draw.rectangle([50, 100, 750, 150], outline='black', width=2)
    draw.rectangle([50, 150, 750, 200], outline='black', width=2)

    # Vertical lines
    draw.line([300, 50, 300, 200], fill='black', width=2)
    draw.line([550, 50, 550, 200], fill='black', width=2)

    # Add text
    draw.text((70, 60), "Name", fill='black')
    draw.text((320, 60), "Email", fill='black')
    draw.text((570, 60), "Phone", fill='black')

    draw.text((70, 110), "John Doe", fill='black')
    draw.text((320, 110), "john@test.com", fill='black')
    draw.text((570, 110), "555-1234", fill='black')

    draw.text((70, 160), "Jane Smith", fill='black')
    draw.text((320, 160), "jane@test.com", fill='black')
    draw.text((570, 160), "555-5678", fill='black')

    img.save('test_images/table_document.png')
    print("✅ Created: test_images/table_document.png")

    # 6. Create a multi-paragraph document
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((50, 30), "Confidential Document", fill='black', font=font)

    # Paragraph 1
    para1 = """This is the first paragraph containing
sensitive information about John Doe.
Email: john.doe@company.com"""

    y_offset = 100
    for line in para1.split('\n'):
        draw.text((50, y_offset), line.strip(), fill='black')
        y_offset += 30

    # Paragraph 2
    para2 = """Second paragraph with additional
personal information. Phone: 555-123-4567
and SSN: 123-45-6789."""

    y_offset += 50
    for line in para2.split('\n'):
        draw.text((50, y_offset), line.strip(), fill='black')
        y_offset += 30

    img.save('test_images/multi_paragraph.png')
    print("✅ Created: test_images/multi_paragraph.png")

    print("\n✅ Test images created successfully in test_images/")


def test_imports():
    """Test that OCR modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports...")
    print("="*60)

    tests = []

    try:
        from app.services.ocr.ocr_engine import OCREngine
        print("✅ OCREngine: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ OCREngine: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.ocr.image_preprocessor import ImagePreprocessor
        print("✅ ImagePreprocessor: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ ImagePreprocessor: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.ocr.layout_analyzer import LayoutAnalyzer
        print("✅ LayoutAnalyzer: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ LayoutAnalyzer: FAIL - {e}")
        tests.append(False)

    try:
        from app.services.ingestion.advanced_image_loader import AdvancedImageLoader
        print("✅ AdvancedImageLoader: PASS")
        tests.append(True)
    except Exception as e:
        print(f"❌ AdvancedImageLoader: FAIL - {e}")
        tests.append(False)

    passed = sum(tests)
    failed = len(tests) - passed
    print(f"\n📊 Results: {passed} passed, {failed} failed")

    return all(tests)


def test_preprocessing():
    """Test image preprocessing features."""
    print("\n" + "="*60)
    print("Testing Image Preprocessing...")
    print("="*60)

    try:
        from app.services.ocr.image_preprocessor import ImagePreprocessor

        preprocessor = ImagePreprocessor()

        # Test with skewed image
        if os.path.exists('test_images/skewed_text.png'):
            print("\n🔧 Testing preprocessing on skewed_text.png...")

            # Load and preprocess
            image = cv2.imread('test_images/skewed_text.png')

            # Test individual features
            print("  - Testing deskewing...")
            deskewed = preprocessor.deskew_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

            print("  - Testing denoising...")
            denoised = preprocessor.denoise_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

            print("  - Testing contrast enhancement...")
            enhanced = preprocessor.enhance_contrast(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

            print("  - Testing binarization...")
            binary = preprocessor.binarize_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

            print("  - Testing full preprocessing pipeline...")
            processed = preprocessor.preprocess(image)

            # Save processed image
            cv2.imwrite('test_images/processed_output.png', processed)

            print("\n✅ Preprocessing tests PASSED")
            print(f"   Saved result to: test_images/processed_output.png")
            return True

    except Exception as e:
        print(f"\n❌ Preprocessing tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_engine():
    """Test OCR engine with Tesseract."""
    print("\n" + "="*60)
    print("Testing OCR Engine...")
    print("="*60)

    try:
        from app.services.ocr.ocr_engine import OCREngine

        # Test with simple text image
        if os.path.exists('test_images/simple_text.png'):
            print("\n📖 Testing OCR on simple_text.png...")

            ocr = OCREngine(engine='tesseract', languages=['eng'], preprocess=False)

            # Extract text
            text = ocr.extract_text('test_images/simple_text.png')
            print(f"\nExtracted text:\n{text}")

            if 'John' in text or 'email' in text.lower():
                print("✅ OCR extraction successful")
                return True
            else:
                print("⚠️ OCR extraction may have issues")
                return False

    except Exception as e:
        print(f"\n❌ OCR engine test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layout_analysis():
    """Test layout-aware extraction."""
    print("\n" + "="*60)
    print("Testing Layout Analysis...")
    print("="*60)

    try:
        from app.services.ocr.layout_analyzer import LayoutAnalyzer

        if os.path.exists('test_images/multi_paragraph.png'):
            print("\n📑 Testing layout analysis on multi_paragraph.png...")

            analyzer = LayoutAnalyzer()
            result = analyzer.analyze_layout('test_images/multi_paragraph.png')

            print(f"\n📊 Layout Analysis Results:")
            print(f"   - Total blocks: {result['layout_metadata']['num_blocks']}")
            print(f"   - Paragraphs: {result['layout_metadata']['num_paragraphs']}")
            print(f"   - Headings: {result['layout_metadata']['num_headings']}")
            print(f"   - Tables: {result['layout_metadata']['num_tables']}")

            print(f"\n📝 Extracted text preview:")
            print(result['full_text'][:200] + "...")

            print("\n✅ Layout analysis PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Layout analysis test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_loader():
    """Test advanced image loader."""
    print("\n" + "="*60)
    print("Testing Advanced Image Loader...")
    print("="*60)

    try:
        from app.services.ingestion.advanced_image_loader import AdvancedImageLoader

        if os.path.exists('test_images/simple_text.png'):
            print("\n🚀 Testing AdvancedImageLoader...")

            # Test with preprocessing
            loader = AdvancedImageLoader(
                ocr_engine='tesseract',
                languages=['eng'],
                preprocess=True,
                layout_analysis=False
            )

            text = loader.load('test_images/simple_text.png')
            print(f"\nExtracted text:\n{text}")

            # Test with layout analysis
            loader_layout = AdvancedImageLoader(
                ocr_engine='tesseract',
                languages=['eng'],
                preprocess=True,
                layout_analysis=True
            )

            layout_result = loader_layout.load_with_layout('test_images/multi_paragraph.png')
            print(f"\n📊 Layout-aware extraction:")
            print(f"   - Blocks found: {len(layout_result['blocks'])}")

            print("\n✅ Advanced loader PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Advanced loader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("🔍 OCR & Preprocessing Features - Test Suite")
    print("="*60 + "\n")

    # Create test images
    create_test_images()

    # Run tests
    results = []
    results.append(("Module Imports", test_imports()))
    results.append(("Image Preprocessing", test_preprocessing()))
    results.append(("OCR Engine", test_ocr_engine()))
    results.append(("Layout Analysis", test_layout_analysis()))
    results.append(("Advanced Loader", test_advanced_loader()))

    # Summary
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
        print("OCR & Preprocessing features are working correctly.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Check the errors above for details.")
    print("="*60 + "\n")

    print("📁 Test images created in: test_images/")
    print("📁 Processed output saved to: test_images/processed_output.png")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
