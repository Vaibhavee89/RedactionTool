
# Redaction Tool

This is a multimedia redaction tool built with Streamlit, capable of processing text, images, and video files. It supports both English and Hindi for text redaction using Named Entity Recognition (NER) and also allows for the redaction of faces in images and videos using facial detection techniques.

## Features

- **Text Redaction**: Automatically redacts sensitive information like names, organizations, dates, and locations from PDF, DOCX, and images (via OCR).
- **Face Redaction**: Detects and blurs faces in images and videos.
- **Multilingual Support**: Supports both English and Hindi for text processing using spaCy NER models.
- **File Types Supported**:
  - PDFs (`.pdf`)
  - Word Documents (`.docx`)
  - Images (`.jpg`, `.png`)
  - Videos (`.mp4`)

## Technologies Used

- **Streamlit**: For building the web-based UI.
- **spaCy**: For Named Entity Recognition (NER) to identify sensitive information in text.
- **Faker**: To generate synthetic data for redaction.
- **Tesseract OCR**: For extracting text from images.
- **OpenCV**: For face detection and blurring in images and videos.
- **pdfplumber**: For extracting text from PDF files.
- **Python-docx**: For extracting text from Word documents.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/multimedia-redaction-tool.git
   cd multimedia-redaction-tool
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy models**:

   ```bash
   python -m spacy download en_core_web_trf
   python -m spacy download xx_ent_wiki_sm
   ```

4. **Install Tesseract OCR**:
   - Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` path in the code to point to your Tesseract executable.

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Upload Files**: Choose a PDF, DOCX, image, or video file to redact sensitive information.
3. **Redaction Options**: For text files, choose the entity types to redact (e.g., names, organizations). For images and videos, face redaction will be applied automatically.
4. **View Results**: The redacted text, image, or video will be displayed within the app.

## Example

1. **Redacting text in a PDF**:
   - Upload a PDF document.
   - Choose the types of sensitive information to redact (e.g., PERSON, ORG, DATE).
   - View the redacted text.

2. **Redacting faces in an image**:
   - Upload an image (`.jpg` or `.png`).
   - Faces in the image will be automatically detected and blurred.

3. **Redacting faces in a video**:
   - Upload a video file (`.mp4`).
   - The tool will process the video and blur faces in each frame.

## Project Structure

```bash
.
├── redaction.py                 # Main Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Python package dependencies
```



