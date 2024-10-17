
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
#- **Faker**: To generate synthetic data for redaction.
- **Tesseract OCR**: For extracting text from images.
- **OpenCV**: For face detection and blurring in images and videos.
- **pdfplumber**: For extracting text from PDF files.
- **python-docx**: For extracting text from Word documents.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Visual Studio Code (VS Code)**
- **Tesseract OCR**:
  - Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).
  - Ensure that you update the `pytesseract.pytesseract.tesseract_cmd` path in `app.py` to point to your Tesseract executable.

### Steps

1. **Clone the Repository**

   Open a terminal and run:

   ```bash
   git clone https://github.com/yourusername/multimedia-redaction-tool.git
   cd multimedia-redaction-tool
   ```

2. **Create a Virtual Environment**

   - **On Windows**:

     ```bash
     python -m venv venv
     ```

   - **On macOS/Linux**:

     ```bash
     python3 -m venv venv
     ```

3. **Activate the Virtual Environment**

   - **On Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - **On macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

4. **Install the Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy Models**

   ```bash
   python -m spacy download en_core_web_trf
   python -m spacy download xx_ent_wiki_sm
   ```

6. **Configure Tesseract OCR Path**

   - Open `app.py` in a text editor or VS Code.
   - Update the `pytesseract.pytesseract.tesseract_cmd` variable with the path to your Tesseract executable. For example:

     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```

## Running the Project in Visual Studio Code (VS Code)

### Opening the Project in VS Code

1. **Launch VS Code**.

2. **Open the Project Folder**:

   - Go to **File** > **Open Folder** (or **Open** on macOS).
   - Navigate to the `multimedia-redaction-tool` directory and open it.

### Setting Up the Virtual Environment in VS Code

1. **Select Python Interpreter**:

   - Click on the Python version displayed in the status bar at the bottom-left corner of VS Code.
   - If it's not visible, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette, type `Python: Select Interpreter`, and select it.
   - Choose the interpreter from your virtual environment:
     - **Windows**: `.\venv\Scripts\python.exe`
     - **macOS/Linux**: `./venv/bin/python`

2. **Install the Python Extension (if not already installed)**:

   - Go to the Extensions tab (or press `Ctrl+Shift+X`), search for `Python`, and install the official Python extension by Microsoft.

### Running the Streamlit App

1. **Activate the Virtual Environment in the Terminal**:

   - Open a new terminal in VS Code: **Terminal** > **New Terminal**.
   - Activate the virtual environment:

     - **On Windows**:

       ```bash
       venv\Scripts\activate
       ```

     - **On macOS/Linux**:

       ```bash
       source venv/bin/activate
       ```

2. **Run the Streamlit Application**:

   ```bash
   streamlit run app.py
   ```

   - The Streamlit app will start, and a new browser tab should open automatically pointing to `http://localhost:8501`.
   - If it doesn't open automatically, you can manually navigate to this URL in your web browser.

### Debugging in VS Code

If you wish to debug your application:

1. **Create a Launch Configuration**:

   - Go to the **Run and Debug** tab on the left sidebar or press `Ctrl+Shift+D`.
   - Click on **create a launch.json file** if you don't have one already.
   - Choose **Python** when prompted for the environment.

2. **Edit `launch.json`**:

   Replace the content with the following configuration:

   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Streamlit",
               "type": "python",
               "request": "launch",
               "module": "streamlit",
               "args": [
                   "run",
                   "${workspaceFolder}/app.py"
               ],
               "console": "integratedTerminal"
           }
       ]
   }
   ```

3. **Start Debugging**:

   - Set breakpoints in your `app.py` or other Python files.
   - Press `F5` to start debugging.
   - The Streamlit app will run in debug mode, allowing you to inspect variables and step through code.

## Usage

1. **Upload Files**: In the Streamlit app, upload a PDF, DOCX, image, or video file that you want to redact.

2. **Redaction Options**:

   - **For Text Files**: Select the entity types you wish to redact (e.g., PERSON, ORG, DATE, GPE).
   - **For Images and Videos**: Face redaction is applied automatically.

3. **View Results**:

   - **Text**: The extracted and redacted text will be displayed in text areas.
   - **Images**: The original and redacted images will be shown.
   - **Videos**: The redacted video will be available for playback within the app.

## Example

1. **Redacting Text in a PDF**:

   - **Upload**: Click on the file uploader and select a PDF document.
   - **Select Entities**: Choose the entities to redact (e.g., PERSON, ORG).
   - **Redact**: Click the "Redact" button to process the text.
   - **Result**: The redacted text will be displayed below.

2. **Redacting Faces in an Image**:

   - **Upload**: Select an image file (`.jpg` or `.png`).
   - **Process**: The app will automatically detect and blur faces.
   - **Result**: Both the original and redacted images will be displayed.

3. **Redacting Faces in a Video**:

   - **Upload**: Choose a video file (`.mp4`).
   - **Process**: The app will process the video to blur faces frame by frame.
   - **Result**: The redacted video will be available for playback.

## Project Structure

```bash
.
├── app.py                 # Main Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Python package dependencies
├── venv/                  # Virtual environment directory
```

## Dependencies

- **streamlit==1.25.0**
- **docx==0.2.4**
- **pdfplumber==0.7.6**
- **faker==19.2.0**
- **spacy==3.7.0**
- **langdetect==1.0.9**
- **Pillow==10.0.0**
- **pytesseract==0.3.10**
- **opencv-python==4.8.0.76**
- **opencv-python-headless==4.8.0.76**
- **numpy==1.23.5**

Install these using:

```bash
pip install -r requirements.txt
```



