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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__)
CORS(app)

# Config
ANSWER_KEY_PATH = "answer_keys.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MIN_IMAGE_DIM = 500

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
        logger.error(f"Load key error: {e}")
        return {}

def save_answer_keys(keys):
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(keys, f)
        return True
    except Exception as e:
        logger.error(f"Save key error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit_key", methods=["POST"])
def submit_key():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    subject = data.get("subject", "").strip()
    answers = data.get("answers", [])

    if not subject:
        return jsonify({"error": "Subject is required"}), 400

    if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
        return jsonify({"error": "Answers must be an array of strings"}), 400

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

    logger.info(f"Answer key saved for {subject}")
    return jsonify({"message": f"Answer key for {subject} saved", "subject": subject, "answers": valid_answers})

def process_options(image, options, option_labels):
    max_fill = 0
    selected = None
    for i, (x, y, w, h) in enumerate(options):
        bubble = image[y:y+h, x:x+w]
        fill = np.sum(bubble == 255) / (w * h)
        if fill > max_fill and fill > 0.55:
            max_fill = fill
            selected = option_labels[i]
    return selected if selected else "N/A"

def detect_answers(image):
    try:
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)

        # remove lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

        contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / float(h)
            if (15 < w < 80 and 15 < h < 80 and 0.5 < aspect < 1.5 and 300 < area < 4000):
                bubbles.append((x, y, w, h))

        logger.info(f"Detected {len(bubbles)} bubbles")

        if not bubbles:
            return []

        # group into rows
        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))
        rows = []
        current_row = []
        y_threshold = 30
        last_y = bubbles[0][1]

        for b in bubbles:
            x, y, w, h = b
            if abs(y - last_y) > y_threshold:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b[0]))
                    current_row = []
            current_row.append(b)
            last_y = y
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        logger.info(f"Grouped into {len(rows)} rows")

        options = ["A", "B", "C", "D"]
        answers = []

        for row in rows:
            if len(row) >= 8:
                left = sorted(row[:4], key=lambda b: b[0])
                right = sorted(row[4:8], key=lambda b: b[0])
                answers.append(process_options(clean, left, options))
                answers.append(process_options(clean, right, options))
            elif len(row) >= 4:
                left = sorted(row[:4], key=lambda b: b[0])
                answers.append(process_options(clean, left, options))

        logger.info(f"Detected answers: {answers}")
        return answers

    except Exception as e:
        logger.error(f"detect_answers error: {e}")
        return []

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

        file.seek(0, os.SEEK_END)
        if file.tell() > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400
        file.seek(0)

        image_np = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        h, w = image.shape[:2]
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return jsonify({"error": "Image too small"}), 400

        student_answers = detect_answers(image)
        keys = load_answer_keys()
        correct_answers = keys.get(subject, [])

        if not correct_answers:
            return jsonify({"error": f"No answer key for subject {subject}"}), 404

        n = min(len(student_answers), len(correct_answers))
        if n == 0:
            return jsonify({"error": "No answers detected"}), 500

        score = 0
        detailed = []
        for i in range(n):
            sa = student_answers[i]
            ca = correct_answers[i]
            correct = sa == ca
            if correct:
                score += 1
            detailed.append({
                "question": i + 1,
                "student_answer": sa,
                "correct_answer": ca,
                "is_correct": correct
            })

        percent = round((score / n) * 100, 2)

        return jsonify({
            "subject": subject,
            "score": score,
            "total": n,
            "percentage": percent,
            "detailed_results": detailed,
            "student_answers": ", ".join(student_answers),
            "correct_answers": ", ".join(correct_answers)
        })

    except Exception as e:
        logger.error(f"process_sheet error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
