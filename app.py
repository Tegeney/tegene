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

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__)
CORS(app)

ANSWER_KEY_PATH = "answer_keys.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024
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
        logger.error(f"Load answer keys failed: {e}")
        return {}

def save_answer_keys(keys):
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(keys, f)
        return True
    except Exception as e:
        logger.error(f"Save answer keys failed: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit_key", methods=["POST"])
def submit_key():
    try:
        data = request.get_json()
        subject = data.get("subject", "").strip()
        answers = data.get("answers", [])

        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        valid_answers = []
        for a in answers:
            a_clean = a.strip().upper()
            if a_clean in ['A', 'B', 'C', 'D']:
                valid_answers.append(a_clean)
            else:
                return jsonify({"error": f"Invalid answer: {a}. Must be A-D"}), 400

        keys = load_answer_keys()
        keys[subject] = valid_answers

        if not save_answer_keys(keys):
            return jsonify({"error": "Failed to save keys"}), 500

        return jsonify({
            "message": f"Key for {subject} saved",
            "subject": subject,
            "answers": valid_answers
        })
    except Exception as e:
        logger.error(f"submit_key error: {e}")
        return jsonify({"error": f"Server error: {e}"}), 500

def process_options(image, options, labels):
    max_fill = 0
    selected = None
    for i, (x, y, w, h) in enumerate(options):
        roi = image[y:y+h, x:x+w]
        fill_ratio = np.sum(roi == 255) / float(w * h)
        if fill_ratio > max_fill and fill_ratio > 0.35:
            max_fill = fill_ratio
            selected = labels[i]
    return selected if selected else "N/A"

def detect_answers(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / float(h)
            if (10 < w < 100 and 10 < h < 100 and 0.4 < aspect < 2.0 and 200 < area < 6000):
                bubbles.append((x, y, w, h))

        logger.info(f"Bubbles detected: {len(bubbles)}")

        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

        rows = []
        row = []
        y_threshold = 30
        last_y = None

        for b in bubbles:
            if last_y is None or abs(b[1] - last_y) <= y_threshold:
                row.append(b)
            else:
                rows.append(sorted(row, key=lambda b: b[0]))
                row = [b]
            last_y = b[1]
        if row:
            rows.append(sorted(row, key=lambda b: b[0]))

        logger.info(f"Rows formed: {len(rows)}")

        answers = []
        labels = ['A', 'B', 'C', 'D']

        for row in rows:
            if len(row) >= 8:
                left = sorted(row[:4], key=lambda b: b[0])
                right = sorted(row[4:8], key=lambda b: b[0])
                answers.append(process_options(clean, left, labels))
                answers.append(process_options(clean, right, labels))
            elif len(row) >= 4:
                options = sorted(row[:4], key=lambda b: b[0])
                answers.append(process_options(clean, options, labels))

        logger.info(f"Answers detected: {answers}")
        return answers

    except Exception as e:
        logger.error(f"detect_answers error: {e}")
        return None

@app.route("/process_sheet", methods=["POST"])
def process_sheet():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        subject = request.form.get("subject", "").strip()

        if not subject:
            return jsonify({"error": "Subject required"}), 400
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400
        file.seek(0)

        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Image decode failed"}), 400

        height, width = image.shape[:2]
        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            return jsonify({"error": "Image too small"}), 400

        student_answers = detect_answers(image)
        if student_answers is None:
            return jsonify({"error": "No answers detected"}), 500

        keys = load_answer_keys()
        correct_answers = keys.get(subject)
        if not correct_answers:
            return jsonify({"error": f"No answer key for {subject}"}), 404

        score = 0
        results = []
        length = min(len(student_answers), len(correct_answers))

        for i in range(length):
            is_correct = student_answers[i] == correct_answers[i]
            score += is_correct
            results.append({
                "question": i+1,
                "student_answer": student_answers[i],
                "correct_answer": correct_answers[i],
                "is_correct": is_correct
            })

        percentage = (score / length * 100) if length else 0

        return jsonify({
            "subject": subject,
            "score": score,
            "total": length,
            "percentage": round(percentage, 2),
            "detailed_results": results,
            "student_answers": student_answers,
            "correct_answers": correct_answers
        })
    except Exception as e:
        logger.error(f"process_sheet error: {e}")
        return jsonify({"error": f"Server error: {e}"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
