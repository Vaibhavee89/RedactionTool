import streamlit as st
import sys
import os

# Fix path for Streamlit Cloud to ensure 'app' module is found
# Should point to the project root (2 levels up from app/ui/streamlit_app.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import io
import os
import tempfile
from app.services.ingestion.pdf_loader import PDFLoader
from app.services.ingestion.docx_loader import DocxLoader
from app.services.ingestion.image_loader import ImageLoader
from app.services.pii.detector_engine import DetectorEngine
from app.services.redaction.redactor import Redactor
from app.services.redaction.image_redactor import ImageRedactor
from app.services.redaction.video_redactor import VideoRedactor
from app.core.config import Config

# Page Config
st.set_page_config(page_title="RedactionTool Ent.", page_icon="🔒", layout="wide")

def main():
    st.title("🔒 RedactionTool Enterprise")
    st.markdown("### Secure Document & Multimedia Redaction Platform")

    # Initialize Services
    Config.setup_paths()
    engine = DetectorEngine()
    text_redactor = Redactor()
    image_redactor = ImageRedactor()
    video_redactor = VideoRedactor()

    # Sidebar Controls
    with st.sidebar:
        st.header("Configuration")
        mode = st.radio("Text Redaction Mode", ["Block (████)", "Mask (****1234)", "Label ([PERSON])"])
        
        # Map UI choice to policy action
        action_map = {
            "Block (████)": "block",
            "Mask (****1234)": "mask",
            "Label ([PERSON])": "label"
        }
        selected_action = action_map[mode]

    # File Upload
    uploaded_file = st.file_uploader("Upload Document (PDF, DOCX) or Media (PNG, JPG, MP4)", type=["pdf", "docx", "png", "jpg", "jpeg", "mp4"])

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        try:
            # ---------------------------
            # VIDEO PROCESSING
            # ---------------------------
            if file_ext == 'mp4':
                st.header("🎥 Video Redaction")
                st.markdown("**Feature:** Automatically detects and blurs faces.")
                
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
                    
                    status_text.text("Processing Complete!")
                    
                    with col2:
                        st.subheader("Redacted Video")
                        st.video(output_path)
                        
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Redacted Video",
                            data=f,
                            file_name=f"redacted_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                    
                    # Cleanup output
                    # os.unlink(output_path) # Keep for download usually, or rely on OS cleanup

            # ---------------------------
            # IMAGE PROCESSING
            # ---------------------------
            elif file_ext in ['png', 'jpg', 'jpeg']:
                st.header("🖼️ Image Redaction")
                
                # Load text for analysis first
                loader = ImageLoader()
                extracted_text = loader.load(temp_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(temp_path, use_container_width=True)
                    with st.expander("See Extracted Text"):
                        st.text(extracted_text)

                if st.button("Blur Faces", type="primary", key="blur_faces"):
                    with st.spinner("Detecting faces & Blurring..."):
                         output_path = temp_path.replace(f".{file_ext}", f"_redacted.{file_ext}")
                         num_faces = image_redactor.redact_faces(temp_path, output_path)
                         
                         if num_faces > 0:
                             st.success(f"Blurred {num_faces} detected faces.")
                         else:
                             st.warning("No faces detected.")

                    with col2:
                        st.subheader("Redacted Image")
                        if os.path.exists(output_path):
                            st.image(output_path, use_container_width=True)
                            
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="Download Redacted Image",
                                    data=f,
                                    file_name=f"redacted_{uploaded_file.name}",
                                    mime=f"image/{file_ext}",
                                    key="dl_face"
                                )

                if st.button("Blur Sensitive Text", type="primary", key="blur_text"):
                    with st.spinner("Detecting PII & Blurring..."):
                        # 1. Detect
                        findings = engine.detect(extracted_text)
                        
                        # 2. Redact Visual
                        output_path = temp_path.replace(f".{file_ext}", f"_redacted.{file_ext}")
                        image_redactor.redact_image(temp_path, findings, output_path)
                        
                        if findings:
                            st.success(f"Blurred {len(findings)} detected entities.")
                        else:
                            st.warning("No PII detected to blur.")

                    with col2:
                        st.subheader("Redacted Image")
                        if os.path.exists(output_path):
                            st.image(output_path, use_container_width=True)
                            
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="Download Redacted Image",
                                    data=f,
                                    file_name=f"redacted_{uploaded_file.name}",
                                    mime=f"image/{file_ext}"
                                )

            # ---------------------------
            # DOCUMENT PROCESSING
            # ---------------------------
            else: # PDF, DOCX
                st.header("📄 Document Redaction")
                text = ""
                if file_ext == "pdf":
                    text = PDFLoader().load(temp_path)
                elif file_ext == "docx":
                    text = DocxLoader().load(temp_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Content")
                    st.text_area("Extracted Text", text, height=400)

                if st.button("Analyze & Redact", type="primary"):
                    with st.spinner("Detecting & Redacting..."):
                        findings = engine.detect(text)
                        policy = {f['entity_type']: selected_action for f in findings}
                        redacted_text = text_redactor.redact_text(text, findings, policy)

                    with col2:
                        st.subheader("Redacted Content")
                        st.text_area("Redacted Text", redacted_text, height=400)
                    
                    st.subheader("Detection Report")
                    if findings:
                        data = [{"Type": f['entity_type'], "Text": f['text'], "Source": f['source']} for f in findings]
                        st.dataframe(data, use_container_width=True)
                    
                    st.download_button(
                        label="Download Redacted Text",
                        data=redacted_text,
                        file_name=f"redacted_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Error occurred: {e}")
        finally:
            # Optional cleanup of upload
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    main()
