# Deployment Guide

To allow others to access the RedactionTool, you can deploy it as a Docker container. This ensures that all dependencies (Tesseract, OpenCV, etc.) are installed correctly.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your machine or server.

## Quick Start (Docker Compose)

1. **Build and Run**
   Run this command in the project root:
   ```bash
   docker-compose up --build -d
   ```

2. **Access the App**
   Open your browser and go to:
   - Local: `http://localhost:8501`
   - Network: `http://<YOUR_IP_ADDRESS>:8501`

   Share `<YOUR_IP_ADDRESS>:8501` with others on your network.

3. **Stop the App**
   ```bash
   docker-compose down
   ```

## Manual Docker Run

If you prefer not to use Compose:

1. **Build Image**
   ```bash
   docker build -t redaction-tool .
   ```

2. **Run Container**
   ```bash
   docker run -d -p 8501:8501 --name my-redaction-app redaction-tool
   ```

## Cloud Deployment (AWS/GCP/Azure)

Since this is a Dockerized app, you can easily deploy it to:
- **AWS App Runner** (Easiest)
- **Google Cloud Run**
- **Azure Container Apps**

Just push your code to a repository and link it to these services, checking that `Dockerfile` is used for the build.
