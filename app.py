from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pytesseract
import os
import json

app = Flask(__name__)

# Directory to store answer keys
ANSWER_KEY_PATH = "answer_keys.json"

# Initialize answer key file if it doesn't exist
if not os.path.exists(ANSWER_KEY_PATH):
    with open(ANSWER_KEY_PATH, "w") as f:
        json.dump({}, f)

# Load answer keys from file
def load_answer_keys():
    try:
        with open(ANSWER_KEY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {}

# Save answer keys to file
def save_answer_keys(keys):
    try:
        with open(ANSWER_KEY_PATH, "w") as f:
            json.dump(keys, f)
    except IOError as e:
        raise Exception(f"Failed to save answer keys: {str(e)}")

# Serve the front-end
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to submit answer key
@app.route("/submit_key", methods=["POST"])
def submit_key():
    data = request.json
    subject = data.get("subject")
    answers = data.get("answers")  # e.g., ["A", "C", "B", ...]

    if not subject or not answers:
        return jsonify({"error": "Missing subject or answers"}), 400

    keys = load_answer_keys()
    keys[subject] = answers
    save_answer_keys(keys)

    return jsonify({"message": f"Answer key for {subject} saved."})

# Util: Extract bubbles (simple detection for demo)
def detect_answers(image):
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # Filter for bubble-like contours
        if 10 < w < 50 and 10 < h < 50:  # Relaxed size constraints
            bubbles.append((x, y, w, h))

    # Sort bubbles top to bottom, then left to right
    bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))
    answers = []
    options = ["A", "B", "C", "D"]

    # Process bubbles in groups of 4
    for i in range(0, len(bubbles), 4):
        group = sorted(bubbles[i:i+4], key=lambda b: b[0])  # Sort left to right
        max_black = 0
        selected = None

        # Check each bubble in the group
        for j, (x, y, w, h) in enumerate(group):
            bubble_region = thresh[y:y+h, x:x+w]
            black_pixels = np.sum(bubble_region == 255)
            total_pixels = w * h
            fill_ratio = black_pixels / total_pixels

            if fill_ratio > max_black:
                max_black = fill_ratio
                selected = options[j]

        if max_black > 0.3:  # Lowered threshold for better detection
            answers.append(selected)
        else:
            answers.append(None)

    return answers

# Endpoint to process scanned answer sheet
@app.route("/process_sheet", methods=["POST"])
def process_sheet():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    subject = request.form.get("subject")

    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid file type. Use PNG or JPG."}), 400

    if not subject:
        return jsonify({"error": "Missing subject"}), 400

    # Read image file
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # Detect answers
    student_answers = detect_answers(image)
    if student_answers is None:
        return jsonify({"error": "Failed to process image"}), 500

    # Load answer key
    keys = load_answer_keys()
    if subject not in keys:
        return jsonify({"error": f"No answer key found for {subject}"}), 404

    correct_answers = keys[subject]
    score = 0
    total = min(len(student_answers), len(correct_answers))

    # Compare answers
    for student, correct in zip(student_answers, correct_answers):
        if student == correct:
            score += 1

    return jsonify({
        "student_answers": student_answers,
        "correct_answers": correct_answers,
        "score": score,
        "total": total,
        "percentage": (score / total * 100) if total > 0 else 0
    })

if __name__ == "__main__":
    app.run(debug=True)