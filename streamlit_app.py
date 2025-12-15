"""
streamlit_app.py

A minimal Streamlit app to demonstrate resume -> job matching.
Supports Deep Analyzer (BERT+SBERT) and TF-IDF matchers.

Run:
    streamlit run streamlit_app.py
"""
import streamlit as st
import json
import pandas as pd
import os

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from resume_matcher import (
    TFIDFMatcher, DeepAnalyzerMatcher,
    SAMPLE_JOBS, DEEP_ANALYZER_AVAILABLE,
    validate_resume_content,
    SkillGapAnalyzer, CareerAdvisor, ResumeFeedback
)

st.set_page_config(page_title="Resume Matcher", layout="centered")

st.title("Resume & Career Advisor")
st.write("Upload a resume text or paste it below, choose the matching backend, and see top job matches.")

uploaded_file = st.file_uploader("Upload resume file (txt, json, or pdf)", type=["txt","json","pdf"])
resume_text = st.text_area("Or paste resume text here", height=200)

if uploaded_file is not None and not resume_text:
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            if PyPDF2 is None:
                st.error("PyPDF2 is required for PDF support. Install it with: pip install PyPDF2")
                resume_text = ""
            else:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                resume_text = ""
                for page in pdf_reader.pages:
                    resume_text += page.extract_text() + "\n"
                if not resume_text.strip():
                    st.warning("No text could be extracted from the PDF. The file might be image-based or corrupted.")
        else:
            content = uploaded_file.read().decode("utf-8")
            parsed = None
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "text" in parsed:
                resume_text = parsed["text"]
            else:
                resume_text = content
    except Exception as e:
        st.error(f"Error reading file: {e}")
        resume_text = ""

# --- RESUME VALIDATION CHECK ---
is_valid_resume = True
if resume_text and len(resume_text.strip()) > 0:
    validation = validate_resume_content(resume_text)
    if not validation['is_valid']:
        is_valid_resume = False
        st.error(f"⚠️ **Invalid Resume Detected** (Confidence: {validation['confidence']*100:.0f}%)")
        
        st.warning("Matching is disabled because this file does not appear to be a resume.")
    else:
        st.success("✅ Valid resume detected")

# Check for local models availability for Deep Analyzer
LOCAL_MODELS_EXIST = False
if DEEP_ANALYZER_AVAILABLE:
    bert_path = os.path.join("models", "bert_classifier")
    sbert_path = os.path.join("models", "sbert_matcher")
    LOCAL_MODELS_EXIST = os.path.exists(bert_path) and os.path.exists(sbert_path)

backend_options = []
if LOCAL_MODELS_EXIST:
    backend_options.append("Deep Analyzer (BERT + SBERT)")

backend_options.append("TF-IDF (Keyword-based)")

backend_type = st.selectbox(
    "Matching method",
    backend_options,
    help="Deep Analyzer: Semantic understanding + Category detection. TF-IDF: Keyword matching."
)

k = st.slider("Top k matches", min_value=1, max_value=10, value=5)

# Initialize results empty by default
results = []

# Only show run button if valid or overridden
if is_valid_resume:
    if st.button("Run match") and resume_text.strip():
        with st.spinner("Analyzing & Generating Advice..."):
            results = []
            try:
                if backend_type.startswith("Deep Analyzer"):
                    matcher = DeepAnalyzerMatcher()
                    matcher.fit(SAMPLE_JOBS)
                    results = matcher.match(resume_text, k=k)
                else:  # TF-IDF
                    matcher = TFIDFMatcher()
                    matcher.fit(SAMPLE_JOBS)
                    results = matcher.match(resume_text, k=k)
            except Exception as e:
                st.error(f"Matching Error: {e}")
                results = []
    
    # --- DISPLAY RESULTS ---
    
    # Filter out weak matches
    filtered_results = [r for r in results if r['similarity'] >= 0.15]
    
    if filtered_results:
        # Summary Table
        df = pd.DataFrame([{
            "Job ID": r["job_id"], 
            "Title": r["job_title"], 
            "Match Score": f"{r['similarity']*100:.2f}%"
        } for r in filtered_results])
        st.table(df)
        
        st.divider()
        st.subheader("Detailed Career Analysis")
        
        # Instantiate Advisors
        gap_analyzer = SkillGapAnalyzer()
        career_advisor = CareerAdvisor()
        feed_provider = ResumeFeedback()
        
        # 1. Resume General Feedback
        with st.expander("📝 Resume Quality Feedback", expanded=True):
            tips = feed_provider.generate_feedback(resume_text)
            for tip in tips:
                st.write(tip)

        st.markdown("---")
        
        for r in filtered_results:
            match_pct = r['similarity']*100
            score_color = "green" if match_pct > 70 else "orange" if match_pct > 40 else "red"
            
            with st.expander(f"**{r['job_id']} - {r['job_title']}** (Score: :{score_color}[{match_pct:.2f}%])", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                job_desc = r['job_meta'].get('text', '')
                
                with col1:
                    st.markdown("**📋 Job Description:**")
                    st.write(job_desc[:300] + "..." if len(job_desc) > 300 else job_desc)
                
                with col2:
                    st.markdown("**✨ Skill Gap Analysis:**")
                    gap_data = gap_analyzer.analyze(resume_text, job_desc)
                    
                    missing = gap_data['missing_skills']
                    matching = gap_data['matching_skills']
                    
                    if matching:
                        st.success(f"Matched Skills: {', '.join(matching)}")
                    if missing:
                        st.error(f"Missing Skills: {', '.join(missing)}")
                        
                        st.markdown("**🎓 Recommended Learning Paths:**")
                        suggestions = career_advisor.suggest_resources(missing)
                        for skill, res in suggestions.items():
                            st.caption(f"- **{skill.title()}**: {res}")
                    else:
                        st.info("No clear skill gaps found! Good fit.")
                        
                # If we have extra info like detected category
                if "matched_context" in r:
                    st.caption(f"Context: {r['matched_context']}")
    
                st.markdown("---")
    elif results: # Results existed but filtered out
        st.warning("No matches found above the similarity threshold (15%). The input might be irrelevant or too short.")
