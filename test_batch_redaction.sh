#!/bin/bash
# Script to create sample files for testing batch redaction

echo "Creating test environment for batch redaction..."

# Create test directories
mkdir -p test_input
mkdir -p test_output

# Create sample text file with PII
cat > test_input/sample1.txt << 'EOF'
Contact Information:
Name: John Doe
Email: john.doe@example.com
Phone: +1-555-123-4567
SSN: 123-45-6789
Address: 123 Main Street, Springfield, IL 62701

This document contains sensitive information that should be redacted.
EOF

# Create another sample text file
cat > test_input/sample2.txt << 'EOF'
Employee Record:
Employee: Jane Smith
Email: jane.smith@company.com
Phone: (555) 987-6543
DOB: 05/15/1985
Credit Card: 4532-1234-5678-9012

Confidential employee information.
EOF

# Create a sample with multiple PII types
cat > test_input/sample3.txt << 'EOF'
Customer Database Export:

Customer 1: Michael Johnson
Email: mjohnson@email.com
Phone: 555-111-2222
Address: 456 Oak Avenue, Boston, MA 02101

Customer 2: Sarah Williams
Email: sarah.w@example.org
Phone: 555-333-4444
SSN: 987-65-4321

Customer 3: Robert Brown
Phone: +1 (555) 555-5555
Email: rbrown@test.com
DOB: 12/25/1990
EOF

# Create a sample DOCX (we'll do this via Python)
python3 << 'PYTHON_EOF'
try:
    from docx import Document

    doc = Document()
    doc.add_heading('Confidential Report', 0)
    doc.add_paragraph('Name: Alice Cooper')
    doc.add_paragraph('Email: alice.cooper@company.com')
    doc.add_paragraph('Phone: 555-999-8888')
    doc.add_paragraph('SSN: 111-22-3333')

    doc.save('test_input/sample_doc.docx')
    print("✅ Created sample_doc.docx")
except ImportError:
    print("⚠️ python-docx not available, skipping DOCX creation")
except Exception as e:
    print(f"⚠️ Error creating DOCX: {e}")
PYTHON_EOF

# Create a simple PDF using reportlab if available
python3 << 'PYTHON_EOF'
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas('test_input/sample_pdf.pdf', pagesize=letter)
    c.drawString(100, 750, "Medical Record")
    c.drawString(100, 720, "Patient: Dr. Emily Davis")
    c.drawString(100, 690, "Email: emily.davis@hospital.com")
    c.drawString(100, 660, "Phone: 555-777-6666")
    c.drawString(100, 630, "SSN: 222-33-4444")
    c.save()
    print("✅ Created sample_pdf.pdf")
except ImportError:
    # Alternative: create a simple text-based PDF instruction
    print("⚠️ reportlab not available, create PDF manually or use existing PDFs")
except Exception as e:
    print(f"⚠️ Error creating PDF: {e}")
PYTHON_EOF

echo ""
echo "✅ Test environment created!"
echo ""
echo "📁 Test files created in: test_input/"
ls -lh test_input/
echo ""
echo "📂 Output will be saved to: test_output/"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Now you can test batch redaction using one of these methods:"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "METHOD 1: Web UI (Recommended)"
echo "  streamlit run app/ui/streamlit_app_enhanced.py"
echo "  Then: Select 'Batch Processing' mode and upload all files"
echo ""
echo "METHOD 2: Command Line"
echo "  python3 cli_batch.py batch -i test_input -o test_output"
echo ""
echo "METHOD 3: Recursive (including subdirectories)"
echo "  python3 cli_batch.py batch -i test_input -o test_output -r"
echo ""
echo "METHOD 4: Filter by file type"
echo "  python3 cli_batch.py batch -i test_input -o test_output --types .txt"
echo ""
