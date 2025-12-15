"""
streamlit_app.py

A minimal Streamlit app to demonstrate resume -> job matching.
Supports TF-IDF matcher out of the box, and embedding matcher if sentence-transformers is installed.

Run:
    streamlit run streamlit_app.py
"""
import streamlit as st
import json
import pandas as pd
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
from resume_matcher import (
    TFIDFMatcher, EmbeddingMatcher, MLMatcher, TransformerMatcher, 
    SAMPLE_JOBS, TRANSFORMERS_AVAILABLE
)

st.set_page_config(page_title="Resume Matcher", layout="centered")

st.title("Resume & Job Matcher")
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

# Always show Transformer option - will show error if not available
backend_options = [
    "Transformer (BERT-based)", 
    "ML Models (classification/regression)", 
    "Embeddings (contextual)", 
    "TF-IDF (keyword-based)"
]

backend_type = st.selectbox(
    "Matching method",
    backend_options,
    help="Transformer: Advanced BERT-based matching with interpretability. ML Models: TF-IDF + embeddings with scikit-learn. Embeddings: Semantic matching. TF-IDF: Fast keyword-based."
)

ml_options = None
transformer_options = None

if backend_type.startswith("Transformer"):
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not installed. Install with: pip install transformers torch")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            trans_model = st.selectbox(
                "Model",
                ["distilbert-base-uncased", "roberta-base", "bert-base-uncased"],
                help="Pre-trained transformer model (DistilBERT is fastest, RoBERTa is most accurate)"
            )
        with col2:
            trans_task = st.selectbox(
                "Task type",
                ["Classification", "Regression"],
                help="Classification: match/no match. Regression: continuous score prediction"
            )
        with col3:
            trans_batch = st.slider(
                "Batch size",
                min_value=1,
                max_value=16,
                value=4,
                help="Larger batches are faster but use more memory"
            )
        
        show_interpretability = st.checkbox(
            "Enable interpretability (attention weights)",
            value=True,
            help="Show which tokens contribute most to the match (slower but more insightful)"
        )
        
        transformer_options = {
            "model": trans_model,
            "task": trans_task.lower(),
            "batch_size": trans_batch,
            "interpretability": show_interpretability
        }

if backend_type.startswith("ML Models"):
    col1, col2 = st.columns(2)
    with col1:
        ml_task = st.selectbox(
            "Task type",
            ["Classification", "Regression"],
            help="Classification: match/no match. Regression: continuous score prediction"
        )
    with col2:
        ml_model = st.selectbox(
            "ML Model",
            ["LogisticRegression", "LinearSVC", "RandomForest"],
            help="Choose the machine learning algorithm"
        )
    ml_options = {
        "task": ml_task.lower(),
        "model": ml_model
    }

k = st.slider("Top k matches", min_value=1, max_value=10, value=5)

# Cache matchers to avoid re-fitting on every request
@st.cache_resource
def get_ml_matcher(model_type, model_name):
    """Cache the ML matcher to improve performance."""
    try:
        matcher = MLMatcher(
            model_type=model_type,
            model_name=model_name,
            embedding_model="all-MiniLM-L6-v2",
            use_faiss=False
        )
        matcher.fit(SAMPLE_JOBS)
        return matcher, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_embedding_matcher():
    """Cache the embedding matcher to improve performance."""
    try:
        matcher = EmbeddingMatcher(model_name="all-MiniLM-L6-v2", use_faiss=False)
        matcher.fit(SAMPLE_JOBS)
        return matcher, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_tfidf_matcher():
    """Cache the TF-IDF matcher to improve performance."""
    matcher = TFIDFMatcher().fit(SAMPLE_JOBS)
    return matcher

@st.cache_resource
def get_transformer_matcher(model_name, model_type, batch_size, enable_interpretability):
    """Cache the transformer matcher to improve performance."""
    try:
        matcher = TransformerMatcher(
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            enable_interpretability=enable_interpretability,
            use_gpu=False  # Set to True if GPU available
        )
        matcher.fit(SAMPLE_JOBS)
        return matcher, None
    except Exception as e:
        return None, str(e)

if st.button("Run match") and resume_text.strip():
    with st.spinner("Matching..."):
        if backend_type.startswith("Transformer"):
            if transformer_options:
                matcher, error = get_transformer_matcher(
                    transformer_options["model"],
                    transformer_options["task"],
                    transformer_options["batch_size"],
                    transformer_options["interpretability"]
                )
                if error:
                    st.error(f"Transformer matcher not available: {error}")
                    results = []
                else:
                    results = matcher.match(
                        resume_text, 
                        k=k, 
                        return_interpretability=transformer_options["interpretability"]
                    )
            else:
                results = []
        elif backend_type.startswith("ML Models"):
            if ml_options:
                matcher, error = get_ml_matcher(ml_options["task"], ml_options["model"])
                if error:
                    st.error(f"ML matcher not available: {error}")
                    results = []
                else:
                    results = matcher.match(resume_text, k=k)
            else:
                results = []
        elif backend_type.startswith("Embeddings"):
            matcher, error = get_embedding_matcher()
            if error:
                st.error(f"Embedding backend not available: {error}")
                results = []
            else:
                results = matcher.match(resume_text, k=k)
        else:  # TF-IDF
            matcher = get_tfidf_matcher()
            results = matcher.match(resume_text, k=k)
    if results:
        df = pd.DataFrame([{"Job ID": r["job_id"], "Title": r["job_title"], "Match Score": f"{r['similarity']*100:.2f}%"} for r in results])
        st.table(df)
        
        st.divider()
        st.subheader("Detailed Matches")
        
        for r in results:
            with st.expander(f"**{r['job_id']} - {r['job_title']}** (Match: {r['similarity']*100:.2f}%)", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**📋 Job Description:**")
                    st.write(r['job_meta'].get('text', 'No description available'))
                
                with col2:
                    st.markdown("**✨ Matched Resume Context:**")
                    matched_context = r.get('matched_context', '')
                    
                    # Check if we have valid matched context
                    if matched_context and len(matched_context.strip()) > 0:
                        # Highlight the matched context
                        st.success(f"*{matched_context}*")
                        
                        # Show matching keywords if available
                        from resume_matcher import extract_keywords
                        job_text = r['job_meta'].get('text', '') + ' ' + r['job_meta'].get('title', '')
                        keywords = extract_keywords(job_text, max_keywords=5)
                        if keywords:
                            st.caption(f"**Key matching terms:** {', '.join(keywords[:5])}")
                    else:
                        # Show a portion of the resume as fallback
                        resume_preview = resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                        st.info(f"*{resume_preview}*")
                        st.caption("Showing resume preview (context matching in progress)")
                
                # Show interpretability data if available (Transformer matcher)
                if 'attention_data' in r and r['attention_data']:
                    st.markdown("**🔍 Interpretability (Attention Analysis):**")
                    attn_data = r['attention_data']
                    
                    # Check if using keyword-based interpretability (sentence-transformers mode)
                    if attn_data.get('matching_keywords') is not None or attn_data.get('note'):
                        # Keyword-based interpretability (sentence-transformers)
                        if attn_data.get('matching_keywords'):
                            st.write("**🔑 Matching Keywords:**")
                            matching_kw = attn_data['matching_keywords']
                            if matching_kw:
                                # Display matching keywords as badges
                                kw_text = " ".join([f"`{kw}`" for kw in matching_kw[:10]])
                                st.markdown(kw_text)
                            else:
                                st.caption("No matching keywords found")
                        
                        # Show resume and job keywords
                        col1, col2 = st.columns(2)
                        with col1:
                            if attn_data.get('resume_keywords'):
                                st.caption(f"**Resume Keywords:** {', '.join(attn_data['resume_keywords'][:8])}")
                        with col2:
                            if attn_data.get('job_keywords'):
                                st.caption(f"**Job Keywords:** {', '.join(attn_data['job_keywords'][:8])}")
                        
                        if attn_data.get('note'):
                            st.caption(f"*{attn_data['note']}*")
                    
                    # Check if using attention-based interpretability (raw transformer mode)
                    elif attn_data.get('resume_tokens') and attn_data.get('job_tokens'):
                        # Attention-based interpretability (raw transformers)
                        st.caption(f"Analyzed {len(attn_data['resume_tokens'])} resume tokens vs {len(attn_data['job_tokens'])} job tokens")
                        
                        # Show feature importance with detailed matches
                        if backend_type.startswith("Transformer") and transformer_options and transformer_options["interpretability"]:
                            try:
                                # Get matcher again for feature importance (cached, so fast)
                                trans_matcher, _ = get_transformer_matcher(
                                    transformer_options["model"],
                                    transformer_options["task"],
                                    transformer_options["batch_size"],
                                    transformer_options["interpretability"]
                                )
                                if trans_matcher and not trans_matcher.use_sentence_transformer:
                                    # Only use get_feature_importance for raw transformers
                                    job_text = r['job_meta'].get('text', '') + ' ' + r['job_meta'].get('title', '')
                                    feat_importance = trans_matcher.get_feature_importance(resume_text, job_text)
                                    if feat_importance.get('matches'):
                                        st.write("**Top token matches (by attention):**")
                                        for match in feat_importance['matches'][:5]:
                                            st.caption(f"Resume: `{match['resume_token']}` ↔ Job: `{match['job_token']}` (attention: {match['attention_score']:.3f})")
                                        
                                        with st.expander("View full feature importance analysis"):
                                            st.json(feat_importance)
                            except Exception as e:
                                st.caption(f"Feature importance unavailable: {str(e)}")
                    
                    # Fallback: show any available data
                    elif attn_data:
                        with st.expander("View interpretability data"):
                            st.json(attn_data)
                
                st.markdown("---")
    else:
        st.info("No results. Check backend availability or increase k.")

st.sidebar.header("Sample jobs")
for j in SAMPLE_JOBS:
    st.sidebar.write(f"**{j['id']}** {j['title']} - {j['text'][:80]}...")
