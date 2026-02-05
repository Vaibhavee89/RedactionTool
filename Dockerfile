# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=tesseract

# Install system dependencies
# tesseract-ocr: for OCR
# libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6: for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy models
# Note: we use the smaller model for the container to keep size reasonable
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download xx_ent_wiki_sm

# Copy application code
COPY . .

# Create directory for uploads if it doesn't exist
RUN mkdir -p uploads output

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
