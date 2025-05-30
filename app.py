import os from flask import Flask, request, jsonify, render_template import cv2 import numpy as np import pytesseract import json from flask_cors import CORS from werkzeug.utils import secure_filename import logging from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') logger = logging.getLogger(name)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(name) CORS(app)

ANSWER_KEY_PATH = "answer_keys.json" ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} MAX_FILE_SIZE = 5 * 1024 * 1024 MIN_IMAGE_DIM = 500

if not os.path.exists(ANSWER_KEY_PATH): with open(ANSWER_KEY_PATH, "w") as f: json.dump({}, f)

def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_answer_keys(): try: with open(ANSWER_KEY_PATH, "r") as f: return json.load(f) except (json.JSONDecodeError, IOError) as e: logger.error(f"Error loading answer keys: {str(e)}") return {}

def save_answer_keys(keys): try: with open(ANSWER_KEY_PATH, "w") as f: json.dump(keys, f) return True except IOError as e: logger.error(f"Error saving answer keys: {str(e)}") return False

@app.route('/') def index(): return render_template('index.html')

@app.route("/submit_key", methods=["POST"]) def submit_key(): try: data = request.get_json() if not data: return jsonify({"error": "No JSON data received"}), 400

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
    return jsonify({
        "message": f"Answer key for {subject} saved successfully",
        "subject": subject,
        "answers": valid_answers
    })

except Exception as e:
    logger.error(f"Error in submit_key: {str(e)}")
    return jsonify({"error": f"Server error: {str(e)}"}), 500

def process_options(image, options, option_labels): max_fill = 0 selected = None for i, (x, y, w, h) in enumerate(options): bubble = image[y:y+h, x:x+w] total = w * h filled = cv2.countNonZero(bubble) fill_ratio = filled / float(total) logger.debug(f"Option {option_labels[i]} fill ratio: {fill_ratio:.2f}") if fill_ratio > max_fill and fill_ratio > 0.35: max_fill = fill_ratio selected = option_labels[i] return selected if selected else "N/A"

def detect_answers(image): try: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) blurred = cv2.GaussianBlur(gray, (5, 5), 0) thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / float(h)
        if (10 < w < 100 and 10 < h < 100 and 0.4 < aspect < 2.0 and 200 < area < 6000):
            bubbles.append((x, y, w, h))

    bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

    rows = []
    current_row = []
    y_thresh = 25
    last_y = -100

    for b in bubbles:
        if abs(b[1] - last_y) > y_thresh and current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = []
        current_row.append(b)
        last_y = b[1]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    answers = []
    options = ["A", "B", "C", "D"]

    for row in rows:
        if len(row) >= 4:
            chunk = sorted(row[:4], key=lambda b: b[0])
            answers.append(process_options(thresh, chunk, options))
            if len(row) >= 8:
                chunk2 = sorted(row[4:8], key=lambda b: b[0])
                answers.append(process_options(thresh, chunk2, options))

    logger.info(f"Detected {len(answers)} answers")
    return answers

except Exception as e:
    logger.error(f"Error in detect_answers: {str(e)}")
    return []

@app.route("/process_sheet", methods=["POST"]) def process_sheet(): try: if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400

file = request.files['file']
    subject = request.form.get("subject", "").strip()

    if not subject:
        return jsonify({"error": "Subject is required"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    if file_length > MAX_FILE_SIZE:
        return jsonify({"error": f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB"}), 400

    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    h, w = image.shape[:2]
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        return jsonify({"error": f"Image too small. Minimum dimension: {MIN_IMAGE_DIM}px"}), 400

    student_answers = detect_answers(image)
    if not student_answers:
        return jsonify({"error": "No answers detected"}), 500

    keys = load_answer_keys()
    if subject not in keys:
        return jsonify({"error": f"No answer key found for {subject}"}), 404

    correct_answers = keys[subject]
    min_len = min(len(student_answers), len(correct_answers))
    score = 0
    details = []

    for i in range(min_len):
        correct = student_answers[i] == correct_answers[i]
        if correct:
            score += 1
        details.append({
            "question": i+1,
            "student_answer": student_answers[i],
            "correct_answer": correct_answers[i],
            "is_correct": correct
        })

    percent = (score / min_len * 100) if min_len > 0 else 0

    return jsonify({
        "subject": subject,
        "score": score,
        "total": min_len,
        "percentage": round(percent, 2),
        "detailed_results": details,
        "student_answers": student_answers,
        "correct_answers": correct_answers
    })

except Exception as e:
    logger.error(f"Error in process_sheet: {str(e)}")
    return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health') def health_check(): return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

if name == "main": port = int(os.environ.get("PORT", 5000)) app.run(host='0.0.0.0', port=port)

