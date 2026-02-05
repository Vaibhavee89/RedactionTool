import streamlit as st
import sys
import os

# Fix path for Streamlit Cloud to ensure 'app' module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import io
import tempfile
import zipfile
from pathlib import Path
from app.services.ingestion.pdf_loader import PDFLoader
from app.services.ingestion.docx_loader import DocxLoader
from app.services.ingestion.image_loader import ImageLoader
from app.services.ingestion.text_loader import TextLoader
from app.services.ingestion.multipage_loader import MultiPageDocumentLoader
from app.services.ingestion.batch_processor import BatchProcessor
from app.services.ingestion.streaming_processor import StreamingProcessor
from app.services.pii.detector_engine import DetectorEngine
from app.services.redaction.redactor import Redactor
from app.services.redaction.image_redactor import ImageRedactor
from app.services.redaction.video_redactor import VideoRedactor
from app.core.config import Config

# Page Config
st.set_page_config(
    page_title="RedactionTool Enterprise v2.0",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔒 RedactionTool Enterprise v2.0")
    st.markdown("### Advanced Secure Document & Multimedia Redaction Platform")

    # Initialize Services
    Config.setup_paths()
    engine = DetectorEngine()
    text_redactor = Redactor()
    image_redactor = ImageRedactor()
    video_redactor = VideoRedactor()
    batch_processor = BatchProcessor()
    streaming_processor = StreamingProcessor()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Processing Mode
        processing_mode = st.radio(
            "Processing Mode",
            ["Single File", "Batch Processing", "Streaming (Large Files)"],
            help="Choose how you want to process your files"
        )

        st.divider()

        # Text Redaction Mode
        mode = st.radio(
            "Text Redaction Mode",
            ["Block (████)", "Mask (****1234)", "Label ([PERSON])"],
            help="Choose how detected PII should be redacted"
        )

        action_map = {
            "Block (████)": "block",
            "Mask (****1234)": "mask",
            "Label ([PERSON])": "label"
        }
        selected_action = action_map[mode]

        st.divider()

        # Feature Info
        with st.expander("📋 Supported Features"):
            st.markdown("""
            **Text Inputs:**
            - Plain text (.txt)
            - PDF (digital + scanned with OCR)
            - DOCX

            **Image Inputs:**
            - PNG / JPG / JPEG
            - Multi-page TIFF
            - Scanned documents

            **Video Inputs:**
            - MP4 / AVI / MOV
            - Frame-level face blurring

            **Batch Processing:**
            - Folder-level ingestion
            - Mixed file types
            - Progress tracking

            **Streaming Mode:**
            - Large file support
            - Memory-efficient processing
            - Chunked analysis
            """)

    # ========================
    # SINGLE FILE MODE
    # ========================
    if processing_mode == "Single File":
        st.header("📄 Single File Processing")

        uploaded_file = st.file_uploader(
            "Upload File",
            type=["txt", "pdf", "docx", "png", "jpg", "jpeg", "tiff", "tif", "mp4", "avi", "mov"],
            help="Supports: TXT, PDF, DOCX, images, and videos"
        )

        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name

            try:
                # VIDEO PROCESSING
                if file_ext in ['mp4', 'avi', 'mov']:
                    st.header("🎥 Video Redaction")
                    st.markdown("**Feature:** Automatically detects and blurs faces frame-by-frame.")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Video")
                        st.video(temp_path)

                    if st.button("Process Video", type="primary"):
                        output_path = temp_path.replace(f".{file_ext}", f"_redacted.{file_ext}")

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(p):
                            progress_bar.progress(p)
                            status_text.text(f"Processing: {int(p*100)}%")

                        with st.spinner("Blurring faces (this may take time)..."):
                            video_redactor.redact_faces(temp_path, output_path, update_progress)

                        status_text.text("✅ Processing Complete!")

                        with col2:
                            st.subheader("Redacted Video")
                            st.video(output_path)

                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Redacted Video",
                                data=f,
                                file_name=f"redacted_{uploaded_file.name}",
                                mime=f"video/{file_ext}"
                            )

                # IMAGE PROCESSING
                elif file_ext in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
                    st.header("🖼️ Image Redaction")

                    # Check if multi-page
                    if file_ext in ['tiff', 'tif']:
                        st.info("Multi-page TIFF detected. Processing all pages...")
                        loader = MultiPageDocumentLoader()
                    else:
                        loader = ImageLoader()

                    extracted_text = loader.load(temp_path)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(temp_path, use_container_width=True)
                        with st.expander("📝 See Extracted Text (OCR)"):
                            st.text(extracted_text)

                    col1_btn, col2_btn = st.columns(2)

                    with col1_btn:
                        if st.button("👤 Blur Faces", type="primary", use_container_width=True):
                            with st.spinner("Detecting faces & Blurring..."):
                                output_path = temp_path.replace(f".{file_ext}", f"_faces_redacted.{file_ext}")
                                num_faces = image_redactor.redact_faces(temp_path, output_path)

                                if num_faces > 0:
                                    st.success(f"✅ Blurred {num_faces} detected faces.")
                                else:
                                    st.warning("⚠️ No faces detected.")

                            with col2:
                                st.subheader("Redacted Image")
                                if os.path.exists(output_path):
                                    st.image(output_path, use_container_width=True)

                                    with open(output_path, "rb") as f:
                                        st.download_button(
                                            label="📥 Download",
                                            data=f,
                                            file_name=f"redacted_faces_{uploaded_file.name}",
                                            mime=f"image/{file_ext}",
                                            key="dl_face"
                                        )

                    with col2_btn:
                        if st.button("🔍 Blur Sensitive Text", type="primary", use_container_width=True):
                            with st.spinner("Detecting PII & Blurring..."):
                                findings = engine.detect(extracted_text)
                                output_path = temp_path.replace(f".{file_ext}", f"_text_redacted.{file_ext}")
                                image_redactor.redact_image(temp_path, findings, output_path)

                                if findings:
                                    st.success(f"✅ Blurred {len(findings)} detected entities.")
                                else:
                                    st.warning("⚠️ No PII detected to blur.")

                            with col2:
                                st.subheader("Redacted Image")
                                if os.path.exists(output_path):
                                    st.image(output_path, use_container_width=True)

                                    with open(output_path, "rb") as f:
                                        st.download_button(
                                            label="📥 Download",
                                            data=f,
                                            file_name=f"redacted_text_{uploaded_file.name}",
                                            mime=f"image/{file_ext}",
                                            key="dl_text"
                                        )

                # TEXT DOCUMENT PROCESSING
                else:  # txt, pdf, docx
                    st.header("📄 Document Redaction")

                    # Select appropriate loader
                    if file_ext == "txt":
                        loader = TextLoader()
                    elif file_ext == "pdf":
                        loader = PDFLoader()
                        # Check for scanned PDF option
                        force_ocr = st.checkbox(
                            "Force OCR (for scanned PDFs)",
                            help="Enable this if the PDF is scanned and text extraction fails"
                        )
                    elif file_ext == "docx":
                        loader = DocxLoader()
                        force_ocr = False

                    # Load text
                    if file_ext == "pdf" and force_ocr:
                        text = loader.load(temp_path, force_ocr=True)
                    else:
                        text = loader.load(temp_path)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Content")
                        st.text_area("Extracted Text", text, height=400)

                    if st.button("🔍 Analyze & Redact", type="primary"):
                        with st.spinner("Detecting & Redacting..."):
                            findings = engine.detect(text)
                            policy = {f['entity_type']: selected_action for f in findings}
                            redacted_text = text_redactor.redact_text(text, findings, policy)

                        with col2:
                            st.subheader("Redacted Content")
                            st.text_area("Redacted Text", redacted_text, height=400)

                        st.subheader("📊 Detection Report")
                        if findings:
                            data = [{"Type": f['entity_type'], "Text": f['text'], "Source": f['source']} for f in findings]
                            st.dataframe(data, use_container_width=True)
                            st.metric("Total PII Found", len(findings))
                        else:
                            st.info("No PII detected in document.")

                        st.download_button(
                            label="📥 Download Redacted Text",
                            data=redacted_text,
                            file_name=f"redacted_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )

            except Exception as e:
                st.error(f"❌ Error occurred: {e}")
                st.exception(e)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    # ========================
    # BATCH PROCESSING MODE
    # ========================
    elif processing_mode == "Batch Processing":
        st.header("📦 Batch Processing")
        st.markdown("Process multiple files at once - supports mixed file types!")

        # File upload (multiple)
        uploaded_files = st.file_uploader(
            "Upload Multiple Files",
            type=["txt", "pdf", "docx", "png", "jpg", "jpeg", "mp4", "avi"],
            accept_multiple_files=True,
            help="Select multiple files to process in batch"
        )

        if uploaded_files and len(uploaded_files) > 0:
            st.info(f"📁 {len(uploaded_files)} files uploaded")

            # Show file list
            with st.expander("📋 View File List"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"{i+1}. {file.name} ({file.type})")

            if st.button("🚀 Process All Files", type="primary"):
                # Create temp directory for input and output
                with tempfile.TemporaryDirectory() as temp_dir:
                    input_dir = os.path.join(temp_dir, "input")
                    output_dir = os.path.join(temp_dir, "output")
                    os.makedirs(input_dir)

                    # Save uploaded files
                    file_paths = []
                    for file in uploaded_files:
                        file_path = os.path.join(input_dir, file.name)
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                        file_paths.append(file_path)

                    # Process with progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(filename, progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {filename}")

                    with st.spinner("Processing batch..."):
                        results = batch_processor.process_file_list(
                            file_paths,
                            output_dir,
                            progress_callback=update_progress
                        )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Batch processing complete!")

                    # Display results
                    st.subheader("📊 Processing Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Files", results['total_files'])
                    with col2:
                        st.metric("Processed", len(results['processed']))
                    with col3:
                        st.metric("Failed", len(results['failed']))
                    with col4:
                        st.metric("Total PII Found", results['stats']['total_pii_found'])

                    # Stats breakdown
                    st.subheader("📈 File Type Breakdown")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("📄 Text Documents", results['stats']['text_documents'])
                    with stat_col2:
                        st.metric("🖼️ Images", results['stats']['images'])
                    with stat_col3:
                        st.metric("🎥 Videos", results['stats']['videos'])

                    # Show failed files if any
                    if results['failed']:
                        st.warning("⚠️ Some files failed to process:")
                        for failed in results['failed']:
                            st.error(f"❌ {failed['file']}: {failed['error']}")

                    # Create ZIP file for download
                    if results['processed']:
                        zip_path = os.path.join(temp_dir, "redacted_files.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for item in results['processed']:
                                if item.get('output_path') and os.path.exists(item['output_path']):
                                    zipf.write(
                                        item['output_path'],
                                        arcname=os.path.basename(item['output_path'])
                                    )

                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download All Redacted Files (ZIP)",
                                data=f,
                                file_name="redacted_batch.zip",
                                mime="application/zip",
                                type="primary"
                            )

    # ========================
    # STREAMING MODE
    # ========================
    elif processing_mode == "Streaming (Large Files)":
        st.header("🌊 Streaming Mode - Large File Processing")
        st.markdown("Memory-efficient processing for very large documents")

        uploaded_file = st.file_uploader(
            "Upload Large File",
            type=["txt", "pdf", "docx"],
            help="Best for files > 10MB"
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📊 File size: {file_size_mb:.2f} MB")

            # Chunk size configuration
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=5000,
                max_value=50000,
                value=10000,
                step=5000,
                help="Larger chunks = faster processing but more memory usage"
            )

            streaming_proc = StreamingProcessor(chunk_size=chunk_size)

            file_ext = uploaded_file.name.split('.')[-1].lower()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name

            # Estimate processing time
            estimate = streaming_proc.estimate_processing_time(temp_path)
            st.info(f"⏱️ Estimated processing time: {estimate['estimated_minutes']:.1f} minutes")

            if st.button("🚀 Process Large File", type="primary"):
                output_path = temp_path.replace(f".{file_ext}", f"_redacted.txt")

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {int(progress*100)}%")

                with st.spinner("Processing in streaming mode..."):
                    try:
                        result = streaming_proc.process_large_text_file(
                            temp_path,
                            output_path,
                            overlap=500,
                            progress_callback=update_progress
                        )

                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")

                        # Show results
                        st.success("File processed successfully!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total PII Found", result['total_pii_found'])
                        with col2:
                            if 'total_pages' in result:
                                st.metric("Pages Processed", result['total_pages'])

                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Redacted File",
                                data=f,
                                file_name=f"redacted_{uploaded_file.name}.txt",
                                mime="text/plain",
                                type="primary"
                            )

                    except Exception as e:
                        st.error(f"❌ Error during streaming processing: {e}")
                        st.exception(e)

                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🔒 RedactionTool Enterprise v2.0 | Secure PII Redaction Platform</p>
        <p>Powered by spaCy, Presidio, OpenCV & Tesseract OCR</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
