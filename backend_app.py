from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import resume_matcher
# Ensure resume_matcher.py is in the same directory
try:
    from resume_matcher import TFIDFMatcher, SAMPLE_JOBS, extract_keywords
    logger.info("Successfully imported resume_matcher")
except ImportError as e:
    logger.error(f"Failed to import resume_matcher: {e}")
    raise

import PyPDF2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Matcher
try:
    matcher = TFIDFMatcher().fit(SAMPLE_JOBS)
    logger.info("TFIDFMatcher initialized and fitted")
except Exception as e:
    logger.error(f"Error initializing matcher: {e}")
    matcher = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not matcher:
        return jsonify({"error": "Matcher backend not initialized"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = file.filename
    logger.info(f"Received file: {filename}")
    
    text = ""
    try:
        if filename.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif filename.lower().endswith('.json'):
            content = json.load(file)
            text = content.get('text', '')
        else:
            # Assume text/markdown
            text = file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400
        
    if not text.strip():
        return jsonify({"error": "Could not extract text from file"}), 400
        
    try:
        # Perform matching
        matches = matcher.match(text)
        
        # Extract keywords
        detected_keywords = extract_keywords(text)
        
        # Mocking some data that isn't directly returned by matcher yet
        # In a real scenario, we'd analyze which keywords are missing based on the top job match
        top_match = matches[0] if matches else None
        
        recommended_keywords = []
        if top_match:
            job_text = top_match['job_meta'].get('text', '')
            job_keywords = extract_keywords(job_text)
            # Find keywords in job but not in resume
            recommended_keywords = [kw for kw in job_keywords if kw not in detected_keywords][:8]

        response = {
            "score": int(top_match['similarity'] * 100) if top_match else 0,
            "summary": f"Based on your resume, the best fit is '{top_match['job_title']}' with a match score of {int(top_match['similarity']*100)}%. " + 
                       (f"Matched context: {top_match['matched_context'][:200]}..." if top_match else ""),
            "detected_keywords": detected_keywords[:15],
            "recommended_keywords": recommended_keywords,
            "formatting_issues": ["Check date formats", "Add LinkedIn URL"] # Placeholder
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
