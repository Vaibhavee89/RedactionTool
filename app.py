import streamlit as st
import spacy
import cv2
import numpy as np
import pytesseract
import pdfplumber
from docx import Document
from PIL import Image, ImageDraw
import tempfile
import os
from faker import Faker
import re
from langdetect import detect
import io
import subprocess
from io import BytesIO


# Configure Tesseract path (update this path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Vaibh\tesseract.exe'  # Windows
# For macOS/Linux, it might be: pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Initialize Faker for generating replacement text
fake = Faker()

# Load spaCy models
@st.cache_resource
def load_spacy_models():
    try:
        nlp_en = spacy.load("en_core_web_trf")
        nlp_multi = spacy.load("xx_ent_wiki_sm")
        return nlp_en, nlp_multi
    except OSError:
        st.warning("Downloading required spaCy models...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_trf"])
        subprocess.run(["python", "-m", "spacy", "download", "xx_ent_wiki_sm"])
        nlp_en = spacy.load("en_core_web_trf")
        nlp_multi = spacy.load("xx_ent_wiki_sm")
        return nlp_en, nlp_multi


def detect_language(text):
    """Detect the language of the text"""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English

def redact_text(text, entities_to_redact, nlp_model):
    """Redact sensitive information from text using NER"""
    doc = nlp_model(text)
    redacted_text = text
    
    # Sort entities by start position in reverse order to avoid index shifting
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents 
                if ent.label_ in entities_to_redact]
    entities.sort(reverse=True)
    
    for start, end, label in entities:
        # Generate replacement text based on entity type
        if label == "PERSON":
            replacement = f"[REDACTED_PERSON_{fake.first_name()}]"
        elif label == "ORG":
            replacement = f"[REDACTED_ORG_{fake.company()}]"
        elif label == "DATE":
            replacement = "[REDACTED_DATE]"
        elif label == "GPE":  # Geopolitical entity
            replacement = f"[REDACTED_LOCATION_{fake.city()}]"
        elif label == "MONEY":
            replacement = "[REDACTED_AMOUNT]"
        elif label == "TIME":
            replacement = "[REDACTED_TIME]"
        else:
            replacement = f"[REDACTED_{label}]"
        
        redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
    
    return redacted_text

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    return pytesseract.image_to_string(image)

def detect_faces(image):
    """Detect faces in an image using OpenCV"""
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces, opencv_image

def blur_faces(image, faces):
    """Blur detected faces in the image"""
    for (x, y, w, h) in faces:
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        # Replace face region with blurred version
        image[y:y+h, x:x+w] = blurred_face
    return image

def process_video_frames(video_path):
    """Process video frames to blur faces"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Blur faces
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face
        
        out.write(frame)
        
        # Update progress
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    progress_bar.empty()
    
    return temp_output.name

def main():
    st.title("🔒 Multimedia Redaction Tool")
    st.markdown("**Redact sensitive information from text, images, and videos**")
    
    # Load spaCy models
    nlp_en, nlp_multi = load_spacy_models()
    if nlp_en is None or nlp_multi is None:
        st.stop()
    
    # Sidebar for options
    st.sidebar.header("Redaction Options")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a file to redact",
        type=['pdf', 'docx', 'jpg', 'jpeg', 'png', 'mp4'],
        help="Supported formats: PDF, DOCX, JPG, PNG, MP4"
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        st.success(f"Uploaded: {file_name}")
        
        # Text-based files (PDF, DOCX)
        if file_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            st.header("📄 Text Redaction")
            
            # Entity selection
            entities_to_redact = st.sidebar.multiselect(
                "Select entities to redact:",
                options=['PERSON', 'ORG', 'DATE', 'GPE', 'MONEY', 'TIME'],
                default=['PERSON', 'ORG'],
                help="Choose which types of sensitive information to redact"
            )
            
            if st.button("🔍 Extract and Redact Text", type="primary"):
                with st.spinner("Processing text..."):
                    # Extract text based on file type
                    if file_type == 'application/pdf':
                        text = extract_text_from_pdf(uploaded_file)
                    else:  # DOCX
                        text = extract_text_from_docx(uploaded_file)
                    
                    if text.strip():
                        # Detect language
                        language = detect_language(text)
                        st.info(f"Detected language: {language}")
                        
                        # Choose appropriate model
                        nlp_model = nlp_en if language == 'en' else nlp_multi
                        
                        # Redact text
                        redacted_text = redact_text(text, entities_to_redact, nlp_model)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original Text")
                            st.text_area("", text, height=400, key="original")
                        
                        with col2:
                            st.subheader("Redacted Text")
                            st.text_area("", redacted_text, height=400, key="redacted")
                        
                        # Download redacted text
                        st.download_button(
                            label="📥 Download Redacted Text",
                            data=redacted_text,
                            file_name=f"redacted_{file_name}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("No text found in the uploaded file.")
        
        # Image files
        elif file_type in ['image/jpeg', 'image/jpg', 'image/png']:
            st.header("🖼️ Image Redaction")
            
            # Load image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # OCR option
            if st.sidebar.checkbox("Extract and redact text from image (OCR)"):
                entities_to_redact = st.sidebar.multiselect(
                    "Select entities to redact:",
                    options=['PERSON', 'ORG', 'DATE', 'GPE', 'MONEY', 'TIME'],
                    default=['PERSON', 'ORG'],
                    key="image_entities"
                )
                
                if st.button("🔍 Extract Text from Image", type="secondary"):
                    with st.spinner("Extracting text..."):
                        extracted_text = extract_text_from_image(image)
                        if extracted_text.strip():
                            language = detect_language(extracted_text)
                            nlp_model = nlp_en if language == 'en' else nlp_multi
                            redacted_text = redact_text(extracted_text, entities_to_redact, nlp_model)
                            
                            st.subheader("Extracted Text")
                            st.text_area("Original", extracted_text, height=150)
                            st.text_area("Redacted", redacted_text, height=150)
                        else:
                            st.warning("No text found in the image.")
            
            # Face redaction
            if st.button("😷 Redact Faces", type="primary"):
                with st.spinner("Detecting and blurring faces..."):
                    faces, opencv_image = detect_faces(image)
                    
                    if len(faces) > 0:
                        st.success(f"Found {len(faces)} face(s)")
                        blurred_image = blur_faces(opencv_image.copy(), faces)
                        
                        # Convert back to PIL Image for display
                        blurred_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
                        
                        with col2:
                            st.subheader("Redacted Image")
                            st.image(blurred_pil, use_container_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        blurred_pil.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Redacted Image",
                            data=byte_im,
                            file_name=f"redacted_{file_name}",
                            mime="image/png"
                        )
                    else:
                        st.warning("No faces detected in the image.")
        
        # Video files
        elif file_type == 'video/mp4':
            st.header("🎥 Video Redaction")
            
            # Display original video
            st.subheader("Original Video")
            st.video(uploaded_file)
            
            if st.button("😷 Redact Faces in Video", type="primary"):
                with st.spinner("Processing video... This may take a while."):
                    # Save uploaded video to temporary file
                    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    temp_input.write(uploaded_file.read())
                    temp_input.close()
                    
                    try:
                        # Process video
                        output_path = process_video_frames(temp_input.name)
                        
                        # Display processed video
                        st.subheader("Redacted Video")
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Redacted Video",
                                data=f.read(),
                                file_name=f"redacted_{file_name}",
                                mime="video/mp4"
                            )
                        
                        # Cleanup
                        os.unlink(output_path)
                    
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                    
                    finally:
                        # Cleanup input file
                        os.unlink(temp_input.name)
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload** a file (PDF, DOCX, image, or video)
    2. **Select** redaction options from the sidebar
    3. **Process** the file using the appropriate button
    4. **Download** the redacted result
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Supported Entities")
    st.sidebar.markdown("""
    - **PERSON**: Names of people
    - **ORG**: Organizations, companies
    - **DATE**: Dates and time periods
    - **GPE**: Countries, cities, states
    - **MONEY**: Monetary values
    - **TIME**: Time expressions
    """)

if __name__ == "__main__":
    main()

