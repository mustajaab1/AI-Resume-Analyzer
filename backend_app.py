from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import PyPDF2

app = Flask(__name__)
CORS(app)

# Initialize Matcher

# Initialize Matchers Global
tfidf_matcher = None
deep_matcher = None

# Initialize Auxiliary Analyzers
skill_gap_analyzer = None
career_advisor = None
resume_feedback = None

try:
    from resume_matcher import (
        DeepAnalyzerMatcher, TFIDFMatcher, SAMPLE_JOBS, extract_keywords,
        SkillGapAnalyzer, CareerAdvisor, ResumeFeedback, validate_resume_content,
        DEEP_ANALYZER_AVAILABLE
    )
    
    # 1. Initialize TF-IDF Matcher (Always available/fallback)
    try:
        tfidf_matcher = TFIDFMatcher().fit(SAMPLE_JOBS)
        logger.info("TFIDFMatcher initialized and fitted")
    except Exception as e:
         logger.error(f"Failed to init TFIDFMatcher: {e}")

    # 2. Initialize Deep Matcher (If available)
    if DEEP_ANALYZER_AVAILABLE:
        try:
            deep_matcher = DeepAnalyzerMatcher(models_dir="models")
            deep_matcher.fit(SAMPLE_JOBS)
            logger.info("DeepAnalyzerMatcher initialized and fitted")
        except Exception as e:
            logger.warning(f"DeepAnalyzerMatcher failed to load ({e}).")
    else:
        logger.warning("Deep Analyzer dependencies not met.")

    # Initialize Auxiliary
    skill_gap_analyzer = SkillGapAnalyzer()
    career_advisor = CareerAdvisor()
    resume_feedback = ResumeFeedback()

except ImportError as e:
    logger.error(f"Failed to import from resume_matcher: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "tfidf_active": tfidf_matcher is not None,
        "deep_active": deep_matcher is not None
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Determine requested analysis type
    analysis_type = request.form.get('type', 'general') # Default to general (TF-IDF)
    
    matcher = None
    if analysis_type == 'deep':
        if deep_matcher:
            matcher = deep_matcher
            logger.info("Using DeepAnalyzerMatcher")
        else:
            logger.warning("DeepAnalyzerMatcher requested but not active. Falling back to TFIDF.")
            matcher = tfidf_matcher
    else:
        matcher = tfidf_matcher
        logger.info("Using TFIDFMatcher")

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

    # Validate content first
    validation = validate_resume_content(text) if 'validate_resume_content' in globals() else {"is_valid": True}
    
    try:
        # Perform matching
        matches = matcher.match(text)
        top_match = matches[0] if matches else None
        
        # Extract keywords
        detected_keywords = extract_keywords(text)
        
        # Skill Gap Analysis
        missing_skills = []
        matching_skills = []
        
        if top_match and skill_gap_analyzer:
            job_text = top_match['job_meta'].get('text', '') + " " + top_match['job_meta'].get('title', '')
            gap_analysis = skill_gap_analyzer.analyze(text, job_text)
            missing_skills = gap_analysis.get('missing_skills', [])
            matching_skills = gap_analysis.get('matching_skills', [])
        
        # Career Advice
        learning_resources = {}
        if career_advisor and missing_skills:
            learning_resources = career_advisor.suggest_resources(missing_skills)
            
        # Detailed Feedback
        feedback = []
        if resume_feedback:
            feedback = resume_feedback.generate_feedback(text)
            if not validation.get('is_valid', True):
                feedback.insert(0, "⚠️ **Validation Warning**: " + "; ".join(validation.get('issues', [])))

        # Fallback for recommended keywords (if skill gap analyzer missed something or wasn't used)
        if not missing_skills and top_match:
             job_text = top_match['job_meta'].get('text', '')
             job_keywords = extract_keywords(job_text)
             missing_skills = [kw for kw in job_keywords if kw not in detected_keywords][:8]

        response = {
            "score": int(top_match['similarity'] * 100) if top_match else 0,
            "job_title": top_match['job_title'] if top_match else "Unknown Role",
            "summary": f"Best fit: '{top_match['job_title']}' ({int(top_match['similarity']*100)}%). " + 
                       (f"Context: {top_match['matched_context'][:200]}..." if top_match else ""),
            
            "detected_keywords": detected_keywords[:15],
            "missing_skills": missing_skills,
            "learning_resources": learning_resources,
            "detailed_feedback": feedback,
            "validation": validation
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
