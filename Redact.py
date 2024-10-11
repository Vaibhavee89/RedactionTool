import streamlit as st
from docx import Document
import pdfplumber
from faker import Faker
import spacy
from langdetect import detect
from PIL import Image
import pytesseract
import cv2
import tempfile
import os
import numpy as np


# Load English NER model
nlp_en = spacy.load("en_core_web_trf")

# Load Multilingual NER model (supports Hindi)
nlp_hi = spacy.load("xx_ent_wiki_sm")

# Fake data generators
fake_en = Faker('en')
fake_hi = Faker('hi')

# Initialize Tesseract OCR for image text extraction
# pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract-ocr-w64-setup-5.4.0.20240606.exe'

# Function to redact multilingual text
def redact_multilingual(text, entity_types, redaction_level):
    lang = detect(text)
    if lang == 'en':
        nlp = nlp_en
    elif lang == 'hi':
        nlp = nlp_hi
    else:
        raise ValueError("Unsupported language")
    
    doc = nlp(text)
    redacted_text = text
    fake = fake_en if lang == 'en' else fake_hi

    for ent in doc.ents:
        if ent.label_ in entity_types:
            synthetic = fake.sentence(nb_words=len(ent.text.split()))
            redacted_text = redacted_text.replace(ent.text, f"<<{synthetic}>>")
    
    return redacted_text

# Function to extract text from images using OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to redact faces in images
def redact_faces_in_image(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y:y+h, x:x+w] = blurred_face

    return image

# Function to process video (OCR and face redaction on each frame)
def process_video(video_file, output_file):
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Redact faces by blurring them
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face

        out.write(frame)

    cap.release()
    out.release()

# Streamlit Dashboard UI
st.set_page_config(page_title="Multimedia Redaction Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option:", ["Text Redaction", "Image Redaction", "Video Redaction"])

# Upload File
st.sidebar.title("File Upload")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'docx', 'jpg', 'png', 'mp4'])

# Main UI based on user selection
if uploaded_file is not None:
    file_name = uploaded_file.name

    if option == "Text Redaction" and file_name.endswith(('.pdf', '.docx')):
        st.title("Text Redaction")
        # Extract text from PDF or Word document
        if file_name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                text = ''.join(page.extract_text() for page in pdf.pages)
        elif file_name.endswith('.docx'):
            doc = Document(uploaded_file)
            text = '\n'.join([para.text for para in doc.paragraphs])

        st.write("Extracted Text:")
        st.text_area("Document Content", value=text, height=300)

        entity_types = st.multiselect("Choose entity types to redact", ["PERSON", "ORG", "DATE", "GPE"], default=["PERSON"])
        redaction_level = st.radio("Redaction Level", ["full", "partial"])

        if st.button("Redact"):
            redacted_text = redact_multilingual(text, entity_types, redaction_level)
            st.write("Redacted Text:")
            st.text_area("Redacted Content", value=redacted_text, height=300)

    elif option == "Image Redaction" and file_name.endswith(('.jpg', '.png')):
        st.title("Image Redaction")
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        text = extract_text_from_image(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Redact Faces"):
            redacted_image = redact_faces_in_image(image_cv)
            st.image(redacted_image, caption="Redacted Image", use_column_width=True)

    elif option == "Video Redaction" and file_name.endswith('.mp4'):
        st.title("Video Redaction")
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, 'wb') as temp_video:
            temp_video.write(uploaded_file.read())

        # Output redacted video file
        redacted_video_path = 'redacted_video.mp4'
        if st.button("Process Video"):
            with st.spinner('Processing video...'):
                process_video(temp_video_path, redacted_video_path)
            st.success("Video processing complete")
            st.video(redacted_video_path)

else:
    st.sidebar.warning("Please upload a file to continue.")

