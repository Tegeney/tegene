import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pytesseract
import json
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Tesseract path for Render
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
ANSWER_KEY_PATH = "answer_keys.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MIN_IMAGE_DIM = 200  # Reduced from 500 to 200 to accommodate narrow answer sheet

# Initialize answer key file if it doesn't exist
if not os.path.exists(ANSWER_KEY_PATH):
    with open(ANSWER_KEY_PATH, "w") as f:
        json.dump({}, f)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_answer_keys():
    try:
        with open(ANSWER_KEY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading answer keys: {str(e)}")
        return {}

def save_answer_keys(keys):
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(keys, f)
        return True
    except IOError as e:
        logger.error(f"Error saving answer keys: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit_key", methods=["POST"])
def submit_key():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        subject = data.get("subject", "").strip()
        answers = data.get("answers", [])
        
        if not subject:
            return jsonify({"error": "Subject is required"}), 400
        
        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
            return jsonify({"error": "Answers must be an array of strings"}), 400
        
        # Validate each answer is A-D
        valid_answers = []
        for a in answers:
            a_clean = a.strip().upper()
            if a_clean in ['A', 'B', 'C', 'D']:
                valid_answers.append(a_clean)
            else:
                return jsonify({"error": f"Invalid answer: {a}. Must be A, B, C, or D"}), 400
        
        keys = load_answer_keys()
        keys[subject] = valid_answers
        
        if not save_answer_keys(keys):
            return jsonify({"error": "Failed to save answer keys"}), 500
            
        logger.info(f"Answer key saved for {subject} at {datetime.now()}")
        return jsonify({
            "message": f"Answer key for {subject} saved successfully",
            "subject": subject,
            "answers": valid_answers
        })
        
    except Exception as e:
        logger.error(f"Error in submit_key: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def detect_answers(image):
    try:
        if image is None:
            return None
        
        # Preprocessing: Enhance contrast and apply Gaussian blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Adaptive thresholding with higher sensitivity
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 5)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if 10 < w < 50 and 10 < h < 50:  # Adjusted range for robustness
                bubbles.append((x, y, w, h))

        # Sort bubbles top to bottom, then left to right
        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))
        
        answers = []
        options = ["A", "B", "C", "D"]

        # Process bubbles in groups of 4
        for i in range(0, len(bubbles), 4):
            group = sorted(bubbles[i:i+4], key=lambda b: b[0])
            max_black = 0
            selected = None

            for j, (x, y, w, h) in enumerate(group):
                bubble_region = thresh[y:y+h, x:x+w]
                black_pixels = np.sum(bubble_region == 255)
                fill_ratio = black_pixels / (w * h)

                # Threshold to differentiate filled bubbles
                if fill_ratio > max_black and fill_ratio > 0.5:
                    max_black = fill_ratio
                    selected = options[j]

            answers.append(selected if selected else "N/A")

        return answers
    
    except Exception as e:
        logger.error(f"Error in detect_answers: {str(e)}")
        return None

@app.route("/process_sheet", methods=["POST"])
def process_sheet():
    try:
        # Check file in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        subject = request.form.get("subject", "").strip()
        
        if not subject:
            return jsonify({"error": "Subject is required"}), 400
            
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400
            
        # Check file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        
        if file_length > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB"}), 400
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        # Check minimum dimensions
        height, width = image.shape[:2]
        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            return jsonify({"error": f"Image too small. Minimum dimension: {MIN_IMAGE_DIM}px"}), 400
            
        # Detect answers
        student_answers = detect_answers(image)
        if student_answers is None:
            return jsonify({"error": "Failed to process answer sheet"}), 500
            
        # Get answer key
        keys = load_answer_keys()
        if subject not in keys:
            return jsonify({"error": f"No answer key found for {subject}"}), 404
            
        correct_answers = keys[subject]
        min_length = min(len(student_answers), len(correct_answers))
        
        # Calculate score
        score = 0
        detailed_results = []
        
        for i in range(min_length):
            is_correct = student_answers[i] == correct_answers[i]
            if is_correct:
                score += 1
            detailed_results.append({
                "question": i+1,
                "student_answer": student_answers[i],
                "correct_answer": correct_answers[i],
                "is_correct": is_correct
            })
        
        percentage = (score / min_length * 100) if min_length > 0 else 0
        
        logger.info(f"Processed answer sheet for {subject} at {datetime.now()} with score {score}/{min_length}")
        return jsonify({
            "subject": subject,
            "score": score,
            "total": min_length,
            "percentage": round(percentage, 2),
            "detailed_results": detailed_results,
            "student_answers": student_answers,
            "correct_answers": correct_answers
        })
        
    except Exception as e:
        logger.error(f"Error in process_sheet: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
