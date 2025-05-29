#!/usr/bin/env bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt
