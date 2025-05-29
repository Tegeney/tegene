document.addEventListener('DOMContentLoaded', function() {
    // Handle Answer Key Form Submission
    const keyForm = document.getElementById('keyForm');
    if (keyForm) {
        keyForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const subject = document.getElementById('subject').value.trim();
            const answersInput = document.getElementById('answers').value.trim();
            const resultElement = document.getElementById('result');
            
            // Clear previous results
            resultElement.textContent = '';
            
            // Validate inputs
            if (!subject) {
                resultElement.textContent = "Error: Please enter a subject name";
                return;
            }
            
            if (!answersInput) {
                resultElement.textContent = "Error: Please enter answers";
                return;
            }
            
            // Process and validate answers
            const answers = answersInput.split(',').map(a => a.trim().toUpperCase());
            
            if (answers.some(a => !/^[A-D]$/.test(a))) {
                resultElement.textContent = "Error: Answers must be comma-separated letters (A, B, C, or D)";
                return;
            }
            
            try {
                const response = await fetch('/submit_key', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        subject: subject,
                        answers: answers
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to submit answer key');
                }
                
                resultElement.textContent = `✅ Success: ${data.message}\n\nSubject: ${subject}\nAnswers: ${answers.join(', ')}`;
                keyForm.reset();
            } catch (error) {
                console.error('Submission error:', error);
                resultElement.textContent = `❌ Error: ${error.message}`;
            }
        });
    }
    
    // Handle Answer Sheet Form Submission
    const sheetForm = document.getElementById('sheetForm');
    if (sheetForm) {
        sheetForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(sheetForm);
            const loadingElement = document.getElementById('loading');
            const resultElement = document.getElementById('result');
            
            // Validate subject
            if (!formData.get('subject').trim()) {
                resultElement.textContent = "Error: Please enter a subject name";
                return;
            }
            
            // Validate file
            const file = formData.get('file');
            if (!file || file.size === 0) {
                resultElement.textContent = "Error: Please select an image file";
                return;
            }
            
            // Show loading and clear previous results
            loadingElement.style.display = 'block';
            resultElement.textContent = '';
            
            try {
                const response = await fetch('/process_sheet', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to process answer sheet');
                }
                
                // Format the results
                let resultText = `📝 Subject: ${formData.get('subject')}\n`;
                resultText += `✅ Score: ${data.score}/${data.total} (${data.percentage.toFixed(1)}%)\n\n`;
                resultText += "Question-by-Question Results:\n";
                
                for (let i = 0; i < Math.min(data.student_answers.length, data.correct_answers.length); i++) {
                    const studentAns = data.student_answers[i] || 'Empty';
                    const correctAns = data.correct_answers[i];
                    const isCorrect = studentAns === correctAns;
                    
                    resultText += `Q${i+1}: ${isCorrect ? '✔' : '✖'} `;
                    resultText += `Your: ${studentAns}, Correct: ${correctAns}\n`;
                }
                
                resultElement.textContent = resultText;
                sheetForm.reset();
            } catch (error) {
                console.error('Processing error:', error);
                resultElement.textContent = `❌ Error: ${error.message}`;
            } finally {
                loadingElement.style.display = 'none';
            }
        });
    }
});