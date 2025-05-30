from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import pytesseract

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store answer keys
answer_keys = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AnswerSheetGrader:
    def __init__(self, answer_key):
        self.answer_key = answer_key
        
    def process_image(self, image_path):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detect contours (bubbles)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        # For this specific answer sheet format, we'll use a grid-based approach
        # since the layout is consistent across all sample images
        
        # Initialize answers dictionary
        answers = {}
        
        # Process each question (1-60)
        for q_num in range(1, 61):
            # Determine row and column position
            row = (q_num - 1) % 30
            col = 0 if q_num <= 30 else 1
            
            # Calculate bubble positions (this needs calibration for your specific sheet)
            y_start = 100 + row * 30  # Adjust these values based on your image
            x_start = 100 + col * 200  # Adjust for left/right column
            
            # Check each option (A, B, C, D)
            option_positions = {
                'A': (x_start + 20, y_start + 10),
                'B': (x_start + 50, y_start + 10),
                'C': (x_start + 80, y_start + 10),
                'D': (x_start + 110, y_start + 10)
            }
            
            marked_options = []
            for option, (x, y) in option_positions.items():
                # Check a small region around the expected bubble position
                bubble_region = thresh[y-5:y+5, x-5:x+5]
                if np.sum(bubble_region) > 1000:  # Threshold for marked bubble
                    marked_options.append(option)
            
            # Determine answer (handle multiple/no marks)
            if len(marked_options) == 1:
                answers[q_num] = marked_options[0]
            else:
                answers[q_num] = None  # Invalid/multiple marks
        
        return answers
    
    def grade_sheet(self, image_path):
        answers = self.process_image(image_path)
        score = 0
        
        for q_num, student_answer in answers.items():
            if student_answer == self.answer_key.get(q_num):
                score += 1
                
        percentage = (score / 60) * 100
        return {
            'score': score,
            'percentage': percentage,
            'answers': answers,
            'total_questions': 60,
            'answer_key': self.answer_key
        }

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
            grader = AnswerSheetGrader(answer_keys[subject])
            results = grader.grade_sheet(filepath)
            results['subject'] = subject
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
