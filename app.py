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
import sys
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

# Configuration
ANSWER_KEY_PATH = os.getenv("ANSWER_KEY_PATH", "answer_keys.json")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024))  # Default 5MB
MIN_IMAGE_DIM = int(os.getenv("MIN_IMAGE_DIM", 500))
MIN_QUESTIONS_PER_SUBJECT = int(os.getenv("MIN_QUESTIONS_PER_SUBJECT", 40))

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")

# Initialize answer key file
if not os.path.exists(ANSWER_KEY_PATH):
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump({}, f)
        logger.info(f"Created answer key file at {ANSWER_KEY_PATH}")
    except IOError as e:
        logger.error(f"Failed to create answer key file: {str(e)}")
        sys.exit(1)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_answer_keys() -> Dict:
    try:
        with open(ANSWER_KEY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading answer keys: {str(e)}")
        return {}

def save_answer_keys(keys: Dict) -> bool:
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(keys, f, indent=2)
        return True
    except IOError as e:
        logger.error(f"Error saving answer keys: {str(e)}")
        return False

def detect_subject_areas(image: np.ndarray) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
    """Detect subject areas using color detection"""
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for subject areas
        color_ranges = {
            'subject1': {
                'lower': np.array([0, 100, 100]),  # Red range
                'upper': np.array([10, 255, 255])
            },
            'subject2': {
                'lower': np.array([100, 100, 100]),  # Blue range
                'upper': np.array([140, 255, 255])
            }
        }
        
        subject_areas = {}
        
        for subject, colors in color_ranges.items():
            mask = cv2.inRange(hsv, colors['lower'], colors['upper'])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    subject_areas[subject] = (x, y, w, h)
        
        return subject_areas if subject_areas else None
    
    except Exception as e:
        logger.error(f"Error in detect_subject_areas: {str(e)}")
        return None

def detect_answers(image: np.ndarray, subject_area: Optional[Tuple[int, int, int, int]] = None) -> Optional[List[str]]:
    try:
        if image is None:
            return None
            
        # Crop to subject area if specified
        if subject_area:
            x, y, w, h = subject_area
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                logger.error("Invalid subject area coordinates")
                return None
            image = image[y:y+h, x:x+w]
        
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 5)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if 15 < w < 60 and 15 < h < 60:
                bubbles.append((x, y, w, h))

        # Sort bubbles top-to-bottom, then left-to-right
        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))
        
        answers = []
        options = ["A", "B", "C", "D"]

        # Process in groups of 4 (A-D options)
        for i in range(0, len(bubbles), 4):
            group = bubbles[i:i+4]
            if len(group) != 4:
                answers.append("N/A")
                continue
                
            group = sorted(group, key=lambda b: b[0])  # Sort left to right
            
            max_black = 0
            selected = None

            for j, (x, y, w, h) in enumerate(group):
                bubble_region = thresh[y:y+h, x:x+w]
                black_pixels = np.sum(bubble_region == 255)
                fill_ratio = black_pixels / (w * h) if w * h > 0 else 0

                if fill_ratio > max_black and fill_ratio > 0.5:
                    max_black = fill_ratio
                    selected = options[j]

            answers.append(selected if selected else "N/A")

        return answers
    
    except Exception as e:
        logger.error(f"Error in detect_answers: {str(e)}")
        return None

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return jsonify({"error": "Failed to load index page"}), 500

@app.route("/submit_key", methods=["POST"])
def submit_key():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data received in submit_key")
            return jsonify({"error": "No JSON data received"}), 400
            
        subject = data.get("subject", "").strip()
        subject_code = data.get("subject_code", "").strip()
        answers = data.get("answers", [])
        
        if not subject or not subject_code:
            logger.warning(f"Missing subject or subject_code: subject={subject}, subject_code={subject_code}")
            return jsonify({"error": "Subject name and code are required"}), 400
        
        if not isinstance(answers, list) or len(answers) < MIN_QUESTIONS_PER_SUBJECT:
            logger.warning(f"Invalid answers array: length={len(answers)}")
            return jsonify({"error": f"Answers must be an array of at least {MIN_QUESTIONS_PER_SUBJECT} strings"}), 400
        
        # Validate answers
        valid_answers = []
        for i, a in enumerate(answers):
            a_clean = a.strip().upper() if isinstance(a, str) else ""
            if a_clean not in ['A', 'B', 'C', 'D', '']:
                logger.warning(f"Invalid answer at position {i+1}: {a}")
                return jsonify({"error": f"Invalid answer at position {i+1}: {a}. Must be A, B, C, D, or empty"}), 400
            valid_answers.append(a_clean if a_clean else "N/A")
        
        keys = load_answer_keys()
        key_id = f"{subject_code}_{subject}"
        keys[key_id] = {
            "subject": subject,
            "code": subject_code,
            "answers": valid_answers
        }
        
        if not save_answer_keys(keys):
            logger.error("Failed to save answer keys")
            return jsonify({"error": "Failed to save answer keys"}), 500
            
        logger.info(f"Answer key saved for {subject} ({subject_code}) with {len(valid_answers)} questions")
        return jsonify({
            "message": f"Answer key for {subject} saved successfully",
            "subject": subject,
            "code": subject_code,
            "total_questions": len(valid_answers),
            "answers": valid_answers
        })
        
    except Exception as e:
        logger.error(f"Error in submit_key: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/process_sheet", methods=["POST"])
def process_sheet():
    try:
        if 'file' not in request.files:
            logger.warning("No file uploaded in process_sheet")
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("No selected file in process_sheet")
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400
            
        # Check file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        
        if file_length > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_length} bytes")
            return jsonify({"error": f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB"}), 400
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            return jsonify({"error": "Failed to decode image"}), 400
            
        if image.shape[0] < MIN_IMAGE_DIM or image.shape[1] < MIN_IMAGE_DIM:
            logger.warning(f"Image too small: {image.shape}")
            return jsonify({"error": f"Image too small. Minimum dimension: {MIN_IMAGE_DIM}px"}), 400
            
        # Detect subject areas
        subject_areas = detect_subject_areas(image)
        if not subject_areas:
            logger.error("Could not detect subject areas")
            return jsonify({"error": "Could not detect subject areas on answer sheet"}), 400
            
        # Process each subject area
        results = {}
        keys = load_answer_keys()
        
        for subject_id, area in subject_areas.items():
            answers = detect_answers(image, area)
            if not answers:
                logger.error(f"Failed to detect answers for {subject_id}")
                results[subject_id] = {"error": "Failed to detect answers"}
                continue
            
            best_match = None
            best_score = 0
            
            for key_id, key_data in keys.items():
                correct_answers = key_data['answers']
                match_count = sum(1 for i in range(min(len(answers), len(correct_answers))) 
                                if answers[i] == correct_answers[i])
                
                match_percentage = (match_count / len(correct_answers)) * 100 if correct_answers else 0
                
                if match_percentage > best_score:
                    best_score = match_percentage
                    best_match = key_data
            
            if best_match:
                correct_answers = best_match['answers']
                total_questions = len(correct_answers)
                score = 0
                detailed = []
                
                for i in range(total_questions):
                    if i >= len(answers):
                        break
                    is_correct = answers[i] == correct_answers[i]
                    if is_correct:
                        score += 1
                    detailed.append({
                        "question": i+1,
                        "student_answer": answers[i],
                        "correct_answer": correct_answers[i],
                        "is_correct": is_correct
                    })
                
                percentage = (score / total_questions) * 100 if total_questions > 0 else 0
                
                results[subject_id] = {
                    "subject": best_match['subject'],
                    "code": best_match['code'],
                    "score": score,
                    "total": total_questions,
                    "percentage": round(percentage, 2),
                    "detailed_results": detailed,
                    "student_answers": answers[:total_questions],
                    "correct_answers": correct_answers
                }
            else:
                results[subject_id] = {
                    "error": "No matching answer key found",
                    "detected_answers": answers
                }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in process_sheet: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    try:
        return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()}), 200
    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}")
        return jsonify({"error": "Health check failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_ENV") == "development")
