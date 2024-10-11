import streamlit as st
import pandas as pd
import spacy
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import cv2
import numpy as np
import easyocr
from PIL import Image

# Initialize spaCy and pre-trained models for synthetic data generation
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Initialize EasyOCR without GPU for performance efficiency
reader = easyocr.Reader(['en'], gpu=False)

# Function to resize the image for faster processing
def resize_image(image, max_width=1000):
    """Resize image while maintaining aspect ratio to a reasonable width for faster processing."""
    h, w = image.shape[:2]
    if w > max_width:
        scale_ratio = max_width / float(w)
        image = cv2.resize(image, (max_width, int(h * scale_ratio)))
    return image

# Function to define a region of interest (ROI)
def define_roi(image):
    """Define a specific region of interest to limit OCR processing area."""
    height, width = image.shape[:2]
    roi = image[0:int(height/2), 0:width]  # Example: top half of the image
    return roi

# Function to preprocess the image (convert to grayscale and thresholding)
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    return thresholded

# Function to extract text and bounding boxes using EasyOCR
def extract_text_and_boxes(image_path):
    image = cv2.imread(image_path)
    image = resize_image(image)  # Resize the image for faster processing
    image = define_roi(image)    # Limit OCR to a specific ROI (optional)
    image = preprocess_image(image)  # Preprocess to reduce noise

    # Convert to PIL format for EasyOCR processing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = reader.readtext(np.array(pil_image))  # Extract text and boxes using EasyOCR
    return image, results

# Function to draw black boxes over the redacted text in the image
def redact_text_in_image(image_path, level):
    image, results = extract_text_and_boxes(image_path)
    for result in results:
        text, bbox, _ = result
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        # Redact based on the specified level
        if level == 1:  # Level 1: Mask text with black blocks
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    output_path = image_path.replace(".png", "_redacted.png").replace(".jpg", "_redacted.jpg")
    cv2.imwrite(output_path, image)
    return output_path

# Function to detect and blur faces in an image
def blur_faces(image_path):
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Blur the face region
        face_region = image[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        image[y:y + h, x:x + w] = blurred_face
    
    output_path = image_path.replace(".png", "_faces_redacted.png").replace(".jpg", "_faces_redacted.jpg")
    cv2.imwrite(output_path, image)
    return output_path

# Function to generate synthetic data using GPT-2
def generate_synthetic_data(entity_text, entity_label):
    input_ids = tokenizer.encode(f"Generate a synthetic example for {entity_label}: {entity_text}", return_tensors='pt')
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    synthetic_data = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return synthetic_data

# Function to redact text based on levels
def redact_text(text, level=1):
    doc = nlp(text)
    redacted_text = text

    for ent in doc.ents:
        if level == 1:  # Low level: mask entities
            redacted_text = redacted_text.replace(ent.text, "[REDACTED]")
        elif level == 2:  # Medium level: remove entities and redact symbols
            redacted_text = redacted_text.replace(ent.text, "")
            redacted_text = ''.join(['[SYMBOL]' if not c.isalnum() and not c.isspace() else c for c in redacted_text])
        elif level == 3:  # High level: replace with synthetic data
            synthetic_data = generate_synthetic_data(ent.text, ent.label_)
            redacted_text = redacted_text.replace(ent.text, synthetic_data)

    return redacted_text

# Process the uploaded file based on its type
def process_file(file_path, file_type, level):
    if file_type == "text":
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        redacted_content = redact_text(content, level)
        output_file = file_path.replace(".txt", f"_redacted_l{level}.txt")
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(redacted_content)
        st.write(f"Redacted text file saved to {output_file}")
    
    elif file_type == "csv":
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
        
        for col in df.columns:
            df[col] = df[col].apply(lambda x: redact_text(str(x), level) if isinstance(x, str) else x)
        output_file = file_path.replace(".csv", f"_redacted_l{level}.csv")
        df.to_csv(output_file, index=False)
        st.write(f"Redacted CSV file saved to {output_file}")

    elif file_type == "image":
        if level == 3:
            face_redacted_image = blur_faces(file_path)
            st.write(f"Redacted image with faces blurred saved to {face_redacted_image}")
        else:
            redacted_image = redact_text_in_image(file_path, level)
            st.write(f"Redacted image saved to {redacted_image}")

# Main function for file upload and processing
def main():
    st.title("RE-DACT: Redaction and Anonymization Tool")

    # Use Streamlit's file uploader
    uploaded_file = st.file_uploader("Choose a file (text, CSV, or image)", type=["txt", "csv", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_type = st.selectbox("Is this a text file, CSV file, or an image file?", ["text", "csv", "image"])
        
        # Choose redaction level
        level = st.slider("Select redaction level", 1, 3)
        
        # Save the uploaded file temporarily
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the file based on type and redaction level
        process_file(file_path, file_type, level)

if __name__ == "__main__":
    main()
