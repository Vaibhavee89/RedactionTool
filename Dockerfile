# Use official Python slim image
FROM python:3.10-slim

# Install system dependencies (Tesseract, Poppler for PDFs, OpenCV dependencies, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary spaCy models (in one layer)
RUN python -m spacy download en_core_web_trf && \
    python -m spacy download xx_ent_wiki_sm

# Copy all remaining application code
COPY . .

# Expose Streamlit’s default port
EXPOSE 8501

# Set Streamlit as the default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
