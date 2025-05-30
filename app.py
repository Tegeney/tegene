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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for the logger name

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = Flask(__name__) # Use __name__ for Flask app name
CORS(app)

ANSWER_KEY_PATH = "answer_keys.json"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024
MIN_IMAGE_DIM = 500

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

def process_options(image, options, option_labels):
    max_fill = 0
    selected = None
    for i, (x, y, w, h) in enumerate(options):
        bubble = image[y:y+h, x:x+w]
        total = w * h
        filled = cv2.countNonZero(bubble)
        fill_ratio = filled / float(total)
        logger.debug(f"Option {option_labels[i]} fill ratio: {fill_ratio:.2f}")
        # A threshold of 0.35 (35%) seems reasonable for a filled bubble.
        # You might need to tune this based on your image quality.
        if fill_ratio > max_fill and fill_ratio > 0.35:
            max_fill = fill_ratio
            selected = option_labels[i]
    return selected if selected else "N/A" # Return "N/A" if no option is clearly selected

def detect_answers(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Using adaptive thresholding for varying lighting conditions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / float(h)

            # Filter out contours that are not likely to be answer bubbles
            # These values (10 < w < 100, 10 < h < 100, 0.4 < aspect < 2.0, 200 < area < 6000)
            # are empirical and may need adjustment based on your scanned forms.
            if (10 < w < 100 and 10 < h < 100 and 0.4 < aspect < 2.0 and 200 < area < 6000):
                bubbles.append((x, y, w, h))

        # Sort bubbles first by Y-coordinate, then by X-coordinate
        # This helps in grouping bubbles into rows
        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

        rows = []
        current_row = []
        # y_thresh determines how close bubbles need to be vertically to be considered in the same row.
        # This value might need tuning.
        y_thresh = 25
        last_y = -100 # Initialize with a value far from any possible y-coordinate

        for b in bubbles:
            if not current_row or abs(b[1] - last_y) <= y_thresh:
                current_row.append(b)
            else:
                # New row detected, process the previous row
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [b]
            last_y = b[1]

        if current_row: # Add the last row if it's not empty
            rows.append(sorted(current_row, key=lambda b: b[0]))

        answers = []
        option_labels = ["A", "B", "C", "D"] # Standard multiple-choice options

        for row in rows:
            # Assuming each question has 4 options (A, B, C, D)
            # We take the first 4 bubbles in the row
            if len(row) >= 4:
                chunk = sorted(row[:4], key=lambda b: b[0]) # Ensure chunk is sorted by X-coordinate
                selected_option = process_options(image, chunk, option_labels)
                answers.append(selected_option)

        logger.info(f"Detected answers: {answers}")
        return answers

    except Exception as e:
        logger.error(f"Error in detect_answers: {str(e)}")
        return []

# Continue with the rest of your Flask application logic for /score_sheet
@app.route("/score_sheet", methods=["POST"])
def score_sheet():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        subject = request.form.get('subject', '').strip()

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        if file and allowed_file(file.filename):
            if len(file.read()) > MAX_FILE_SIZE:
                return jsonify({"error": "File size exceeds limit"}), 413
            file.seek(0) # Reset file pointer after reading size

            filename = secure_filename(file.filename)
            filepath = os.path.join("/tmp", filename) # Save to a temporary location
            file.save(filepath)

            img = cv2.imread(filepath)
            if img is None:
                return jsonify({"error": "Could not read image file"}), 400

            h, w = img.shape[:2]
            if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
                return jsonify({"error": f"Image dimensions too small. Min: {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}"}), 400

            student_answers = detect_answers(img)
            answer_keys = load_answer_keys()

            if subject not in answer_keys:
                return jsonify({"error": f"Answer key for subject '{subject}' not found"}), 404

            correct_answers = answer_keys[subject]
            score = 0
            feedback = []

            # Ensure both lists are of comparable length for fair scoring
            min_len = min(len(student_answers), len(correct_answers))

            for i in range(min_len):
                question_num = i + 1
                student_ans = student_answers[i]
                correct_ans = correct_answers[i]

                if student_ans == correct_ans:
                    score += 1
                    feedback.append(f"Question {question_num}: Correct (Your: {student_ans}, Key: {correct_ans})")
                else:
                    feedback.append(f"Question {question_num}: Incorrect (Your: {student_ans}, Key: {correct_ans})")

            # Handle cases where student answers or key have more questions
            if len(student_answers) > len(correct_answers):
                for i in range(len(correct_answers), len(student_answers)):
                    feedback.append(f"Question {i+1}: No answer key for this question (Your: {student_answers[i]})")
            elif len(correct_answers) > len(student_answers):
                for i in range(len(student_answers), len(correct_answers)):
                    feedback.append(f"Question {i+1}: Not answered by student (Key: {correct_answers[i]})")

            os.remove(filepath) # Clean up the temporary file

            return jsonify({
                "message": "Sheet scored successfully",
                "subject": subject,
                "score": score,
                "total_questions_scored": min_len,
                "student_answers": student_answers,
                "correct_answers": correct_answers,
                "feedback": feedback
            })
        else:
            return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
        logger.error(f"Error in score_sheet: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

