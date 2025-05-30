import os
import cv2
import numpy as np
import pytesseract
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configs
ANSWER_KEY_PATH = "answer_keys.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MIN_IMAGE_DIM = 500

# Init answer key file
if not os.path.exists(ANSWER_KEY_PATH):
    with open(ANSWER_KEY_PATH, "w") as f:
        json.dump({}, f)

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
            return jsonify({"error": "Answers must be an array of strings"}), 400

        valid_answers = []
        for a in answers:
            a = a.strip().upper()
            if a in ["A", "B", "C", "D"]:
                valid_answers.append(a)
            else:
                return jsonify({"error": f"Invalid answer: {a}. Must be A, B, C, or D"}), 400

        keys = load_answer_keys()
        keys[subject] = valid_answers
        if not save_answer_keys(keys):
            return jsonify({"error": "Could not save answer key"}), 500

        logger.info(f"Saved key for {subject}")
        return jsonify({"message": f"Answer key for {subject} saved successfully"}), 200

    except Exception as e:
        logger.error(f"Error in submit_key: {str(e)}")
        return jsonify({"error": "Server error"}), 500

def process_options(thresh, options, labels):
    max_fill = 0
    selected = None
    for i, (x, y, w, h) in enumerate(options):
        roi = thresh[y:y+h, x:x+w]
        total = cv2.countNonZero(roi)
        fill_ratio = total / float(w * h)
        if fill_ratio > 0.4 and fill_ratio > max_fill:
            max_fill = fill_ratio
            selected = labels[i]
    return selected if selected else "N/A"

def detect_answers(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = w * h
            if 20 < w < 80 and 20 < h < 80 and 0.8 < aspect_ratio < 1.2 and area > 500:
                bubbles.append((x, y, w, h))

        if len(bubbles) < 4:
            return None

        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))  # sort top to bottom, then left to right

        rows = []
        current_row = []
        last_y = bubbles[0][1]

        for b in bubbles:
            if abs(b[1] - last_y) > 25:
                if current_row:
                    rows.append(sorted(current_row, key=lambda r: r[0]))
                    current_row = []
            current_row.append(b)
            last_y = b[1]

        if current_row:
            rows.append(sorted(current_row, key=lambda r: r[0]))

        answers = []
        labels = ['A', 'B', 'C', 'D']

        for row in rows:
            if len(row) >= 4:
                row = sorted(row, key=lambda r: r[0])
                selected = None
                max_fill = 0
                for i, (x, y, w, h) in enumerate(row[:4]):
                    roi = thresh[y:y+h, x:x+w]
                    fill = cv2.countNonZero(roi)
                    fill_ratio = fill / float(w * h)
                    if fill_ratio > 0.25 and fill_ratio > max_fill:
                        max_fill = fill_ratio
                        selected = labels[i]
                answers.append(selected if selected else "N/A")

        return answers if answers else None

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return None

@app.route("/process_sheet", methods=["POST"])
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

        # File size check
        file.seek(0, os.SEEK_END)
        if file.tell() > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400
        file.seek(0)

        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        if img.shape[0] < MIN_IMAGE_DIM or img.shape[1] < MIN_IMAGE_DIM:
            return jsonify({"error": "Image too small"}), 400

        student_answers = detect_answers(img)
        if student_answers is None:
            return jsonify({"error": "Could not detect answers"}), 500

        answer_keys = load_answer_keys()
        if subject not in answer_keys:
            return jsonify({"error": f"No answer key for subject '{subject}'"}), 404

        correct_answers = answer_keys[subject]
        min_len = min(len(student_answers), len(correct_answers))
        score = 0
        results = []

        for i in range(min_len):
            correct = correct_answers[i]
            student = student_answers[i]
            is_correct = student == correct
            results.append({
                "question": i + 1,
                "student_answer": student,
                "correct_answer": correct,
                "is_correct": is_correct
            })
            if is_correct:
                score += 1

        percent = (score / min_len * 100) if min_len > 0 else 0

        logger.info(f"{subject}: {score}/{min_len}")
        return jsonify({
            "subject": subject,
            "score": score,
            "total": min_len,
            "percentage": round(percent, 2),
            "detailed_results": results,
            "student_answers": student_answers,
            "correct_answers": correct_answers
        })

    except Exception as e:
        logger.error(f"Error in process_sheet: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "OK", "time": datetime.now().isoformat()}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
