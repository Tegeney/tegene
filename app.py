import os
import cv2
import json
import logging
import numpy as np
import pytesseract
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging with timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tesseract path (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__)
CORS(app)

# Constants
ANSWER_KEY_PATH = "answer_keys.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MIN_IMAGE_DIM = 500

# Ensure answer key storage exists
if not os.path.exists(ANSWER_KEY_PATH):
    with open(ANSWER_KEY_PATH, "w") as f:
        json.dump({}, f)

# --- Helper functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_answer_keys():
    try:
        with open(ANSWER_KEY_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading keys: {e}")
        return {}

def save_answer_keys(keys):
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(keys, f)
        return True
    except Exception as e:
        logger.error(f"Error saving keys: {e}")
        return False

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_key', methods=['POST'])
def submit_key():
    try:
        data = request.get_json()
        subject = data.get("subject", "").strip()
        answers = data.get("answers", [])

        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
            return jsonify({"error": "Answers must be a list of strings"}), 400

        valid_answers = []
        for a in answers:
            clean = a.strip().upper()
            if clean in ['A', 'B', 'C', 'D']:
                valid_answers.append(clean)
            else:
                return jsonify({"error": f"Invalid answer: {a}"}), 400

        keys = load_answer_keys()
        keys[subject] = valid_answers

        if not save_answer_keys(keys):
            return jsonify({"error": "Could not save answer keys"}), 500

        logger.info(f"Answer key saved for {subject}")
        return jsonify({"message": "Saved", "subject": subject, "answers": valid_answers})

    except Exception as e:
        logger.error(f"Error in submit_key: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/process_sheet', methods=['POST'])
def process_sheet():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        subject = request.form.get("subject", "").strip()

        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400
        file.seek(0)

        # Read image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        h, w = image.shape[:2]
        if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
            return jsonify({"error": "Image too small"}), 400

        student_answers = detect_answers(image)
        if student_answers is None:
            return jsonify({"error": "Could not detect answers"}), 500

        keys = load_answer_keys()
        if subject not in keys:
            return jsonify({"error": f"No answer key for {subject}"}), 404

        correct_answers = keys[subject]
        n = min(len(correct_answers), len(student_answers))
        score = 0
        detailed = []

        for i in range(n):
            correct = student_answers[i] == correct_answers[i]
            if correct:
                score += 1
            detailed.append({
                "question": i + 1,
                "student_answer": student_answers[i],
                "correct_answer": correct_answers[i],
                "is_correct": correct
            })

        percent = round((score / n * 100) if n > 0 else 0, 2)

        logger.info(f"Sheet processed for {subject}: {score}/{n}")
        return jsonify({
            "subject": subject,
            "score": score,
            "total": n,
            "percentage": percent,
            "detailed_results": detailed,
            "student_answers": student_answers,
            "correct_answers": correct_answers
        })

    except Exception as e:
        logger.error(f"Error in process_sheet: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

# --- Core Logic ---
def process_options(image, options, labels):
    max_fill = 0
    selected = None
    for i, (x, y, w, h) in enumerate(options):
        bubble = image[y:y+h, x:x+w]
        fill = np.sum(bubble == 255) / (w * h)
        if fill > max_fill and fill > 0.55:
            max_fill = fill
            selected = labels[i]
    return selected or "N/A"

def detect_answers(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Remove lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=3)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, v_kernel, iterations=3)

        contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / float(h)
            if (25 < w < 70 and 25 < h < 70 and 0.7 < aspect < 1.3 and 400 < area < 3500):
                bubbles.append((x, y, w, h))

        bubbles.sort(key=lambda b: (b[1], b[0]))
        rows = []
        row = []
        last_y = None
        y_threshold = 30

        for b in bubbles:
            if last_y is None or abs(b[1] - last_y) <= y_threshold:
                row.append(b)
            else:
                if row:
                    rows.append(sorted(row, key=lambda b: b[0]))
                row = [b]
            last_y = b[1]
        if row:
            rows.append(sorted(row, key=lambda b: b[0]))

        answers = []
        labels = ["A", "B", "C", "D"]
        for r in rows:
            if len(r) >= 8:
                answers.append(process_options(clean, r[:4], labels))
                answers.append(process_options(clean, r[4:8], labels))
            elif len(r) >= 4:
                answers.append(process_options(clean, r[:4], labels))

        return answers

    except Exception as e:
        logger.error(f"Error detecting answers: {e}")
        return None

# --- Entry point ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
