from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store answer keys
answer_keys = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_key', methods=['POST'])
def submit_key():
    subject = request.form.get('subject')
    answers = request.form.get('answers')
    
    if not subject or not answers:
        return jsonify({'error': 'Subject and answers are required'}), 400
    
    # Process answers string (e.g., "A,B,C,D,A,...")
    answer_list = [ans.strip().upper() for ans in answers.split(',')]
    
    if len(answer_list) != 60:
        return jsonify({'error': 'Exactly 60 answers are required'}), 400
    
    # Store in memory (in production, use a database)
    answer_keys[subject] = {i+1: answer_list[i] for i in range(60)}
    
    return jsonify({
        'message': f'Answer key for {subject} submitted successfully',
        'count': len(answer_list)
    })

@app.route('/process_sheet', methods=['POST'])
def process_sheet():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    subject = request.form.get('subject')
    file = request.files['file']
    
    if not subject:
        return jsonify({'error': 'Subject is required'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if answer key exists for this subject
        if subject not in answer_keys:
            return jsonify({'error': f'No answer key found for {subject}'}), 400
        
        # Process the answer sheet
        try:
            results = grade_answer_sheet(filepath, answer_keys[subject])
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def grade_answer_sheet(image_path, answer_key):
    # This is a simplified version - implement your actual image processing here
    # For demonstration, we'll simulate processing
    
    # In a real implementation, you would:
    # 1. Load and preprocess the image
    # 2. Detect the answer sheet structure
    # 3. Extract answers for each question
    # 4. Compare with answer_key
    
    # Simulated processing (replace with actual implementation)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Example: simple mock results
    score = 0
    answers = {}
    for q_num in range(1, 61):
        # Simulate random answers for demo
        student_answer = np.random.choice(['A', 'B', 'C', 'D', None])
        answers[q_num] = student_answer
        if student_answer == answer_key.get(q_num):
            score += 1
    
    percentage = (score / 60) * 100
    
    return {
        'subject': 'Math',  # Would get from request in real implementation
        'score': score,
        'percentage': round(percentage, 2),
        'total_questions': 60,
        'answers': answers,
        'answer_key': answer_key
    }

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
