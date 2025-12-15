"""
resume_matcher.py

Provides:
- TF-IDF based search (working out of the box with scikit-learn)
- Optional semantic-embedding search using sentence-transformers (model downloads required)
- Optional FAISS index creation for ANN search (if faiss is installed)
- Machine Learning matchers (Logistic Regression, Linear SVC, Random Forest)
- Advanced Transformer-based matcher with BERT/RoBERTa (interpretability features)
- Simple preprocessing pipeline
- CLI demo and programmatic API

Usage:
    python resume_matcher.py demo        # runs a small internal demo with synthetic data
    Import functions and use in your own code

Notes:
- For embedding-based search, install sentence-transformers and optionally faiss-cpu:
    pip install sentence-transformers faiss-cpu
- For transformer-based matching, install transformers and torch:
    pip install transformers torch
- The code is written to be robust if optional libs are missing; it will fall back to TF-IDF.
"""

from typing import List, Dict, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# optional libs
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, pipeline
    )
    import torch
    from torch.nn import functional as F
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None
    F = None

def clean_text(text: str) -> str:
    """Clean text for TF-IDF matching (aggressive cleaning, removes punctuation and special chars)."""
    text = text or ""
    # Remove all punctuation and special characters except spaces
    text = re.sub(r"[^\w\s]", " ", text)  # Keep only word characters and spaces
    # Remove standalone numbers (like dates, phone numbers)
    text = re.sub(r'\b\d+\b', ' ', text)
    # Remove single characters (likely punctuation artifacts)
    text = re.sub(r'\b\w\b', ' ', text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def prepare_text_for_embeddings(text: str) -> str:
    """Prepare text for embeddings (remove punctuation but preserve semantic meaning)."""
    text = text or ""
    # Remove standalone punctuation marks and special characters
    text = re.sub(r'\s*[,\/#@$%^&*()_+=\[\]{}|\\:";\'<>?`~]\s*', ' ', text)
    # Remove standalone numbers (dates, phone numbers, etc.)
    text = re.sub(r'\b\d{4,}\b', ' ', text)  # Years, phone numbers
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_matched_context(resume_text: str, job_text: str, max_chunks: int = 5) -> str:
    """
    Extract the most relevant chunks from resume that match the job description.
    Uses both TF-IDF and embeddings for better accuracy.
    
    Args:
        resume_text: Full resume text
        job_text: Job description text
        max_chunks: Maximum number of text chunks to return
    
    Returns:
        String containing the most relevant resume chunks with matched keywords
    """
    if not resume_text or not job_text:
        return resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
    
    # Better text chunking: split by sentences, bullet points, and newlines
    # Split by multiple delimiters
    chunks = re.split(r'[.!?\n•\-\*]+', resume_text)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]  # Lowered threshold
    
    # Also try splitting by common resume patterns (lines, bullet points)
    if len(chunks) < 3:
        # Try line-based splitting
        line_chunks = resume_text.split('\n')
        line_chunks = [chunk.strip() for chunk in line_chunks if len(chunk.strip()) > 10]
        if len(line_chunks) > len(chunks):
            chunks = line_chunks
    
    # If still not enough chunks, try splitting by commas or semicolons
    if len(chunks) < 3:
        comma_chunks = re.split(r'[,;]+', resume_text)
        comma_chunks = [chunk.strip() for chunk in comma_chunks if len(chunk.strip()) > 20]
        if len(comma_chunks) > len(chunks):
            chunks = comma_chunks
    
    # Final fallback: create chunks of fixed length
    if not chunks or len(chunks) < 2:
        # Split into chunks of ~100 characters
        chunk_size = 100
        chunks = [resume_text[i:i+chunk_size].strip() 
                 for i in range(0, len(resume_text), chunk_size)]
        chunks = [chunk for chunk in chunks if len(chunk) > 10]
    
    if not chunks:
        return resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
    
    try:
        # Use both TF-IDF and embeddings for better matching
        tfidf_scores = []
        embedding_scores = []
        
        # TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=5000)
            all_texts = [clean_text(job_text)] + [clean_text(chunk) for chunk in chunks]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            job_vector = tfidf_matrix[0:1]
            chunk_vectors = tfidf_matrix[1:]
            tfidf_scores = cosine_similarity(job_vector, chunk_vectors)[0]
        except Exception:
            tfidf_scores = [0.0] * len(chunks)
        
        # Embedding similarity (if available)
        # Cache the model to avoid reloading
        if SentenceTransformer is not None:
            try:
                # Use a lightweight model for faster processing
                # Cache the model in a module-level variable to avoid reloading
                if not hasattr(extract_matched_context, '_cached_model'):
                    extract_matched_context._cached_model = SentenceTransformer("all-MiniLM-L6-v2")
                model = extract_matched_context._cached_model
                
                job_emb = model.encode([prepare_text_for_embeddings(job_text)], convert_to_numpy=True, show_progress_bar=False)
                chunk_embs = model.encode([prepare_text_for_embeddings(chunk) for chunk in chunks], 
                                         show_progress_bar=False, convert_to_numpy=True)
                embedding_scores = cosine_similarity(job_emb, chunk_embs)[0]
            except Exception:
                embedding_scores = [0.0] * len(chunks)
        else:
            embedding_scores = [0.0] * len(chunks)
        
        # Combine scores (weighted: 40% TF-IDF, 60% embeddings)
        if embedding_scores and any(s > 0 for s in embedding_scores):
            combined_scores = 0.4 * np.array(tfidf_scores) + 0.6 * np.array(embedding_scores)
        else:
            combined_scores = np.array(tfidf_scores)
        
        # Get top matching chunks
        top_indices = np.argsort(-combined_scores)[:max_chunks]
        
        # Use a very low threshold to ensure we always get results
        # Only filter if we have very high confidence matches
        min_threshold = 0.15 if any(combined_scores > 0.2) else 0.0
        matched_chunks = []
        seen_chunks = set()
        
        # Always get top chunks, even if scores are low
        for idx in top_indices:
            chunk = chunks[idx]
            # Avoid duplicates
            chunk_lower = chunk.lower()[:50]  # Use first 50 chars as key
            if chunk_lower not in seen_chunks:
                # Only apply threshold if we have high-scoring matches
                if combined_scores[idx] >= min_threshold or min_threshold == 0.0:
                    matched_chunks.append(chunk)
                    seen_chunks.add(chunk_lower)
                    if len(matched_chunks) >= max_chunks:
                        break
        
        # If we still don't have chunks, take top ones regardless of score
        if not matched_chunks:
            for idx in top_indices[:max_chunks]:
                chunk = chunks[idx]
                chunk_lower = chunk.lower()[:50]
                if chunk_lower not in seen_chunks:
                    matched_chunks.append(chunk)
                    seen_chunks.add(chunk_lower)
                    if len(matched_chunks) >= max_chunks:
                        break
        
        if matched_chunks:
            result = " ... ".join(matched_chunks)
            
            # Truncate if too long
            if len(result) > 500:
                result = result[:500] + "..."
            
            return result
        else:
            # Final fallback: return first few chunks
            fallback_chunks = chunks[:max_chunks] if chunks else []
            if fallback_chunks:
                result = " ... ".join(fallback_chunks)
                if len(result) > 500:
                    result = result[:500] + "..."
                return result
            return resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
            
    except Exception as e:
        # Fallback: return first few chunks
        return " ... ".join(chunks[:max_chunks]) if chunks else resume_text[:300] + "..."

def is_likely_name(word: str) -> bool:
    """Check if a word is likely a person's name."""
    # Common first names (common in resumes)
    common_names = {
        'john', 'jane', 'michael', 'sarah', 'david', 'emily', 'james', 'jennifer',
        'robert', 'lisa', 'william', 'maria', 'richard', 'susan', 'joseph', 'karen',
        'thomas', 'nancy', 'charles', 'betty', 'christopher', 'helen', 'daniel', 'sandra',
        'matthew', 'donna', 'anthony', 'carol', 'mark', 'ruth', 'donald', 'sharon',
        'steven', 'michelle', 'paul', 'laura', 'andrew', 'kimberly', 'joshua', 'deborah',
        'kenneth', 'jessica', 'kevin', 'cynthia', 'brian', 'angela', 'george', 'melissa',
        'edward', 'brenda', 'ronald', 'amy', 'timothy', 'emma', 'jason', 'olivia',
        'jeffrey', 'cynthia', 'ryan', 'marie', 'jacob', 'janet', 'gary', 'catherine',
        'nicholas', 'frances', 'eric', 'ann', 'jonathan', 'samantha', 'stephen', 'debra',
        'larry', 'rachel', 'justin', 'carolyn', 'scott', 'janet', 'brandon', 'virginia',
        'benjamin', 'maria', 'samuel', 'heather', 'frank', 'diane', 'gregory', 'julie',
        'raymond', 'joyce', 'alexander', 'victoria', 'patrick', 'kelly', 'jack', 'christina',
        'dennis', 'joan', 'jerry', 'evelyn', 'tyler', 'judith', 'aaron', 'megan',
        'jose', 'cheryl', 'henry', 'andrea', 'adam', 'hannah', 'douglas', 'jacqueline',
        'nathan', 'martha', 'zachary', 'gloria', 'kyle', 'teresa', 'noah', 'sara',
        'alan', 'janice', 'juan', 'marie', 'wayne', 'julia', 'roy', 'grace', 'ralph', 'judy'
    }
    
    # Check if it's a common name
    if word.lower() in common_names:
        return True
    
    # Check if it's capitalized (likely a name) and not a common word
    if word[0].isupper() and len(word) > 2:
        # Exclude common capitalized words (months, places, etc.)
        common_caps = {
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'usa', 'uk', 'nyc', 'la', 'sf', 'ny', 'ca', 'tx', 'fl', 'il'
        }
        if word.lower() not in common_caps:
            return True
    
    return False

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text, filtering out names, punctuation, and irrelevant tokens."""
    try:
        # Clean text first
        cleaned = clean_text(text)
        # Extract words (minimum 3 characters, alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned)
        if not words:
            return []
        
        # Simple frequency-based keyword extraction
        from collections import Counter
        word_freq = Counter(words)
        
        # Extended stop words list
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 
            'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 
            'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use', 'she', 'with', 'this', 
            'that', 'from', 'have', 'been', 'will', 'would', 'could', 'should', 'about', 
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 
            'among', 'their', 'there', 'these', 'those', 'them', 'they', 'than', 'then',
            'when', 'where', 'what', 'which', 'while', 'until', 'since', 'until', 'upon',
            'under', 'over', 'other', 'only', 'once', 'more', 'most', 'much', 'many', 'made',
            'make', 'made', 'like', 'just', 'know', 'keep', 'keep', 'into', 'here', 'have',
            'good', 'give', 'from', 'find', 'even', 'each', 'does', 'does', 'come', 'call',
            'both', 'been', 'back', 'also', 'after', 'again', 'about', 'above', 'across',
            'against', 'along', 'among', 'around', 'because', 'before', 'behind', 'below',
            'beneath', 'beside', 'between', 'beyond', 'during', 'except', 'inside', 'outside',
            'through', 'throughout', 'toward', 'under', 'underneath', 'until', 'within', 'without'
        }
        
        # Filter out stop words, names, and short words
        filtered_keywords = []
        for word, count in word_freq.most_common(max_keywords * 3):
            word_lower = word.lower()
            # Skip if it's a stop word
            if word_lower in stop_words:
                continue
            # Skip if it's likely a name
            if is_likely_name(word):
                continue
            # Skip if too short
            if len(word) < 3:
                continue
            # Skip if it's mostly numbers
            if word.isdigit():
                continue
            
            filtered_keywords.append(word_lower)
            if len(filtered_keywords) >= max_keywords:
                break
        
        return filtered_keywords
    except Exception:
        return []

class TFIDFMatcher:
    def __init__(self, ngram_range=(1,2), min_df=1):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        self.job_texts = []
        self.job_meta = []
        self.job_matrix = None

    def fit(self, job_postings: List[Dict]):
        self.job_meta = job_postings
        self.job_texts = [clean_text(j.get("text","") + " " + j.get("title","")) for j in job_postings]
        self.job_matrix = self.vectorizer.fit_transform(self.job_texts)
        return self

    def match(self, resume_text: str, k=5) -> List[Dict]:
        if self.job_matrix is None:
            raise ValueError("Matcher not fitted. Call fit(job_postings) first.")
        q = clean_text(resume_text)
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.job_matrix)[0]
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            job_text = self.job_meta[i].get("text", "") + " " + self.job_meta[i].get("title", "")
            matched_context = extract_matched_context(resume_text, job_text)
            results.append({
                "job_id": self.job_meta[i].get("id"),
                "job_title": self.job_meta[i].get("title"),
                "similarity": float(sims[i]),
                "job_meta": self.job_meta[i],
                "matched_context": matched_context
            })
        return results

class EmbeddingMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_faiss: bool = False):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.job_meta = []
        self.job_texts = []
        self.job_embeddings = None
        self.use_faiss = use_faiss and (faiss is not None)
        self.index = None

    def fit(self, job_postings: List[Dict]):
        self.job_meta = job_postings
        # Use less aggressive cleaning for embeddings to preserve semantic meaning
        self.job_texts = [prepare_text_for_embeddings(j.get("text","") + " " + j.get("title","")) for j in job_postings]
        self.job_embeddings = np.vstack(self.model.encode(self.job_texts, show_progress_bar=False, convert_to_numpy=True))
        if self.use_faiss:
            dim = self.job_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.job_embeddings)
            index.add(self.job_embeddings.astype('float32'))
            self.index = index
        return self

    def match(self, resume_text: str, k=5):
        # Use less aggressive cleaning for embeddings to preserve semantic meaning
        q = prepare_text_for_embeddings(resume_text)
        q_emb = self.model.encode([q], convert_to_numpy=True)
        if self.use_faiss and self.index is not None:
            qn = q_emb.copy()
            faiss.normalize_L2(qn)
            D, I = self.index.search(qn.astype('float32'), k)
            sims = D[0].tolist()
            idxs = I[0].tolist()
        else:
            sims = (cosine_similarity(q_emb, self.job_embeddings)[0]).tolist()
            idxs = list(np.argsort(-np.array(sims))[:k])
            sims = [sims[i] for i in idxs]
        results = []
        for i, s in zip(idxs, sims):
            job_text = self.job_meta[i].get("text", "") + " " + self.job_meta[i].get("title", "")
            matched_context = extract_matched_context(resume_text, job_text)
            results.append({
                "job_id": self.job_meta[i].get("id"),
                "job_title": self.job_meta[i].get("title"),
                "similarity": float(s),
                "job_meta": self.job_meta[i],
                "matched_context": matched_context
            })
        return results

class MLMatcher:
    """
    Machine Learning-based matcher using both TF-IDF and embeddings as features.
    Supports both classification (match/no match) and regression (score prediction).
    Available models: Logistic Regression, Linear SVC, Random Forest
    """
    def __init__(self, model_type: str = "classification", model_name: str = "LogisticRegression", 
                 embedding_model: str = "all-MiniLM-L6-v2", use_faiss: bool = False):
        """
        Initialize ML matcher.
        
        Args:
            model_type: "classification" or "regression"
            model_name: "LogisticRegression", "LinearSVC", or "RandomForest"
            embedding_model: Name of sentence transformer model
            use_faiss: Whether to use FAISS for faster embedding search
        """
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        
        self.model_type = model_type
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.use_faiss = use_faiss and (faiss is not None)
        
        # Initialize feature extractors
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.scaler = StandardScaler()
        
        # Initialize ML model
        if model_type == "classification":
            if model_name == "LogisticRegression":
                self.ml_model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == "LinearSVC":
                self.ml_model = LinearSVC(max_iter=1000, random_state=42)
            elif model_name == "RandomForest":
                self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown classification model: {model_name}")
        else:  # regression
            if model_name == "LogisticRegression":
                # Use Ridge regression (linear model similar to LogisticRegression)
                self.ml_model = Ridge(alpha=1.0, random_state=42)
            elif model_name == "LinearSVC":
                # Use SVR with linear kernel (similar to LinearSVC)
                self.ml_model = SVR(kernel='linear', C=1.0)
            elif model_name == "RandomForest":
                self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown regression model: {model_name}")
        
        self.job_meta = []
        self.job_texts = []
        self.tfidf_features = None
        self.embedding_features = None
        self.is_fitted = False
    
    def _extract_features(self, texts: List[str], is_training: bool = False):
        """Extract both TF-IDF and embedding features."""
        # TF-IDF features
        if is_training:
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Embedding features
        embedding_features = self.embedding_model.encode(
            [prepare_text_for_embeddings(t) for t in texts],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Combine features
        combined_features = np.hstack([tfidf_features, embedding_features])
        return combined_features
    
    def fit(self, job_postings: List[Dict], resume_texts: Optional[List[str]] = None):
        """
        Fit the ML model using job postings.
        If resume_texts are provided, uses them to generate training data.
        Otherwise, generates synthetic training data from job postings.
        """
        self.job_meta = job_postings
        self.job_texts = [prepare_text_for_embeddings(j.get("text", "") + " " + j.get("title", "")) 
                          for j in job_postings]
        
        # Generate training data
        if resume_texts is None:
            # Generate synthetic resume texts from job postings (simplified versions)
            resume_texts = [f"{j.get('title', '')} {j.get('text', '')[:200]}" for j in job_postings]
        
        # Extract features for jobs
        job_features = self._extract_features(self.job_texts, is_training=True)
        
        # Extract features for resumes
        resume_features = self._extract_features(resume_texts, is_training=False)
        
        # Generate labels using cosine similarity between TF-IDF and embeddings
        # This creates pseudo-labels for training
        tfidf_sims = cosine_similarity(
            self.tfidf_vectorizer.transform([clean_text(rt) for rt in resume_texts]),
            self.tfidf_vectorizer.transform([clean_text(jt) for jt in self.job_texts])
        )
        
        embedding_sims = cosine_similarity(
            self.embedding_model.encode([prepare_text_for_embeddings(rt) for rt in resume_texts], 
                                       show_progress_bar=False, convert_to_numpy=True),
            self.embedding_model.encode([prepare_text_for_embeddings(jt) for jt in self.job_texts],
                                       show_progress_bar=False, convert_to_numpy=True)
        )
        
        # Combine similarities (weighted average)
        combined_sims = 0.4 * tfidf_sims + 0.6 * embedding_sims
        
        # Prepare training data: for each resume, create features with each job
        X_train = []
        y_train = []
        
        for i, resume_feat in enumerate(resume_features):
            for j, job_feat in enumerate(job_features):
                # Combine resume and job features (concatenate)
                combined_feat = np.hstack([resume_feat, job_feat])
                X_train.append(combined_feat)
                
                # Create label
                sim_score = combined_sims[i, j]
                if self.model_type == "classification":
                    # Binary classification: >0.5 = match (1), <=0.5 = no match (0)
                    y_train.append(1 if sim_score > 0.5 else 0)
                else:  # regression
                    # Use similarity score directly as target
                    y_train.append(float(sim_score))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Store job features for prediction
        self.job_features = job_features
        self.is_fitted = True
        
        return self
    
    def match(self, resume_text: str, k=5) -> List[Dict]:
        """Match resume against jobs using ML model."""
        if not self.is_fitted:
            raise ValueError("Matcher not fitted. Call fit(job_postings) first.")
        
        # Extract features for resume
        resume_features = self._extract_features([resume_text], is_training=False)[0]
        
        # Prepare features for each job
        scores = []
        for job_feat in self.job_features:
            combined_feat = np.hstack([resume_features, job_feat]).reshape(1, -1)
            combined_feat_scaled = self.scaler.transform(combined_feat)
            
            if self.model_type == "classification":
                # Get probability of positive class
                if hasattr(self.ml_model, "predict_proba"):
                    score = self.ml_model.predict_proba(combined_feat_scaled)[0][1]
                else:
                    # For LinearSVC, use decision function
                    score = (self.ml_model.decision_function(combined_feat_scaled)[0] + 1) / 2
                    score = max(0, min(1, score))  # Clip to [0, 1]
            else:  # regression
                score = self.ml_model.predict(combined_feat_scaled)[0]
                score = max(0, min(1, score))  # Clip to [0, 1]
            
            scores.append(float(score))
        
        # Get top k matches
        idx = np.argsort(-np.array(scores))[:k]
        results = []
        for i in idx:
            job_text = self.job_meta[i].get("text", "") + " " + self.job_meta[i].get("title", "")
            matched_context = extract_matched_context(resume_text, job_text)
            results.append({
                "job_id": self.job_meta[i].get("id"),
                "job_title": self.job_meta[i].get("title"),
                "similarity": float(scores[i]),
                "job_meta": self.job_meta[i],
                "matched_context": matched_context
            })
        return results

class TransformerMatcher:
    """
    Advanced Transformer-based matcher using pre-trained BERT/RoBERTa models.
    Supports both classification and regression tasks with interpretability features.
    
    Features:
    - Uses sequence-pair classification for resume-job matching
    - Attention weight extraction for interpretability
    - Batch processing for performance optimization
    - Model caching and quantization support
    - Compatible with HuggingFace transformers library
    """
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 model_type: str = "classification",
                 task_type: str = "sequence-pair",
                 max_length: int = 512,
                 batch_size: int = 8,
                 use_gpu: bool = False,
                 enable_interpretability: bool = True,
                 use_sentence_transformer: bool = True):
        """
        Initialize Transformer matcher.
        
        Args:
            model_name: HuggingFace model name (e.g., "distilbert-base-uncased", 
                       "roberta-base", "bert-base-uncased")
            model_type: "classification" or "regression"
            task_type: "sequence-pair" (recommended) or "sequence-classification"
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing (larger = faster but more memory)
            use_gpu: Whether to use GPU if available
            enable_interpretability: Whether to extract attention weights for interpretability
            use_sentence_transformer: If True and sentence-transformers available, uses 
                                    fine-tuned similarity models instead of raw transformers
                                    (recommended for better accuracy)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers library not installed. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.model_type = model_type
        self.task_type = task_type
        self.max_length = max_length
        self.batch_size = batch_size
        self.enable_interpretability = enable_interpretability
        self.use_sentence_transformer = use_sentence_transformer and (SentenceTransformer is not None)
        
        # Device setup
        self.device = "cuda" if use_gpu and torch and torch.cuda.is_available() else "cpu"
        
        # Use sentence-transformers for better similarity if available (they're fine-tuned for this)
        if self.use_sentence_transformer:
            try:
                # Use sentence-transformers for similarity (much better than raw DistilBERT)
                # Map common model names to sentence-transformer equivalents
                st_model_map = {
                    "distilbert-base-uncased": "all-MiniLM-L6-v2",  # Fast and accurate
                    "bert-base-uncased": "all-mpnet-base-v2",  # More accurate
                    "roberta-base": "all-roberta-large-v1"  # Most accurate but slower
                }
                st_model_name = st_model_map.get(model_name, "all-MiniLM-L6-v2")
                self.sentence_model = SentenceTransformer(st_model_name)
                self.model = None
                self.tokenizer = None
                self.classifier = None
            except Exception as e:
                # Fallback to regular transformer if sentence-transformers fails
                self.use_sentence_transformer = False
                print(f"Warning: Could not load sentence-transformer, falling back to {model_name}: {e}")
        
        # Load regular transformer model if not using sentence-transformers
        if not self.use_sentence_transformer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Add padding token if not present (must be done before model loading)
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                # Use sequence pair classification for better resume-job matching
                try:
                    # Try to load with sequence classification head
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=2 if model_type == "classification" else 1,
                        problem_type="single_label_classification" if model_type == "classification" else "regression"
                    )
                    self.classifier = None  # Model already has classifier head
                except Exception:
                    # Fallback: use base model and add custom classifier
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(model_name)
                    hidden_size = self.model.config.hidden_size
                    if model_type == "classification":
                        self.classifier = torch.nn.Linear(hidden_size, 2).to(self.device)
                    else:
                        self.classifier = torch.nn.Linear(hidden_size, 1).to(self.device)
                
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load transformer model {model_name}: {e}")
        
        self.job_meta = []
        self.job_texts = []
        self.is_fitted = False
    
    def _prepare_sequence_pair(self, text1: str, text2: str) -> Dict:
        """Prepare a sequence pair for the transformer model."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available. This method requires a raw transformer model, not sentence-transformers.")
        
        # Truncate texts if needed to fit max_length
        # Reserve space for special tokens [CLS] text1 [SEP] text2 [SEP]
        max_text_len = (self.max_length - 3) // 2
        
        text1 = text1[:max_text_len] if len(text1) > max_text_len else text1
        text2 = text2[:max_text_len] if len(text2) > max_text_len else text2
        
        return self.tokenizer(
            text1,
            text2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def _boost_keyword_matches(self, base_score: float, resume_text: str, job_text: str) -> float:
        """Boost score when important technical keywords match (filters out punctuation and names)."""
        # Important technical keywords (can be expanded)
        important_keywords = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'machine learning',
            'deep learning', 'tensorflow', 'pytorch', 'aws', 'docker', 'kubernetes',
            'data science', 'backend', 'frontend', 'devops', 'api', 'database',
            'pandas', 'scikit-learn', 'numpy', 'html', 'css', 'express', 'mongodb',
            'postgresql', 'mysql', 'redis', 'git', 'ci/cd', 'terraform', 'linux',
            'typescript', 'angular', 'vue', 'django', 'flask', 'spring', 'kotlin',
            'swift', 'go', 'rust', 'c++', 'c#', 'php', 'ruby', 'scala', 'r',
            'kubernetes', 'jenkins', 'ansible', 'chef', 'puppet', 'vagrant',
            'elasticsearch', 'kibana', 'grafana', 'prometheus', 'splunk',
            'microservices', 'restful', 'graphql', 'websocket', 'grpc',
            'agile', 'scrum', 'kanban', 'ci/cd', 'tdd', 'bdd', 'devops'
        ]
        
        # Clean texts to remove punctuation and special characters
        resume_cleaned = clean_text(resume_text)
        job_cleaned = clean_text(job_text)
        
        # Extract meaningful keywords from both texts
        resume_keywords = set(extract_keywords(resume_text, max_keywords=50))
        job_keywords = set(extract_keywords(job_text, max_keywords=50))
        
        # Count matches in important keywords
        important_matches = sum(1 for kw in important_keywords 
                              if kw in resume_cleaned and kw in job_cleaned)
        
        # Count matches in extracted keywords (excluding names and stop words)
        keyword_matches = len(resume_keywords & job_keywords)
        
        # Combined boost: important keywords get more weight
        boost = min(0.15, important_matches * 0.04 + min(keyword_matches * 0.01, 0.05))
        
        return min(1.0, base_score + boost)
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector."""
        if self.use_sentence_transformer:
            # Use sentence-transformers (much better for similarity tasks)
            embedding = self.sentence_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embedding
        else:
            # Use regular transformer (less reliable for similarity)
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'last_hidden_state'):
                    # Use mean pooling of all non-padding tokens
                    attention_mask = inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        # Mask out padding tokens
                        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                        masked_embeddings = outputs.last_hidden_state * mask
                        sum_embeddings = torch.sum(masked_embeddings, dim=1)
                        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                        mean_pooled = sum_embeddings / sum_mask
                    else:
                        # Fallback: mean of all tokens
                        mean_pooled = outputs.last_hidden_state.mean(dim=1)
                    
                    return mean_pooled.cpu().numpy()
                elif hasattr(outputs, 'pooler_output'):
                    return outputs.pooler_output.cpu().numpy()
                else:
                    # Ultimate fallback
                    return np.zeros((1, 768))  # Default size
    
    def _compute_similarity(self, resume_text: str, job_text: str, return_attention: bool = False) -> float:
        """Compute similarity score using separate encoding (uses sentence-transformers if available)."""
        # Encode resume and job separately
        resume_emb = self._encode_text(resume_text)
        job_emb = self._encode_text(job_text)
        
        # Ensure embeddings are numpy arrays
        if isinstance(resume_emb, torch.Tensor):
            resume_emb = resume_emb.cpu().numpy()
        if isinstance(job_emb, torch.Tensor):
            job_emb = job_emb.cpu().numpy()
        
        # Flatten if needed
        if resume_emb.ndim > 1:
            resume_emb = resume_emb.flatten()
        if job_emb.ndim > 1:
            job_emb = job_emb.flatten()
        
        # Compute cosine similarity
        resume_norm = resume_emb / (np.linalg.norm(resume_emb) + 1e-9)
        job_norm = job_emb / (np.linalg.norm(job_emb) + 1e-9)
        similarity = np.dot(resume_norm, job_norm)
        
        # Convert [-1, 1] to [0, 1] range
        base_score = (similarity + 1) / 2
        
        # Apply keyword boosting for technical skill matches
        boosted_score = self._boost_keyword_matches(base_score, resume_text, job_text)
        
        # Apply score normalization with better discrimination
        # Use a more aggressive normalization to spread scores
        # Map [0, 1] to a wider range for better differentiation
        if boosted_score < 0.3:
            # Low scores: compress them further
            normalized_score = boosted_score * 0.3
        elif boosted_score < 0.6:
            # Medium scores: expand them
            normalized_score = 0.3 + (boosted_score - 0.3) * 1.5
        else:
            # High scores: expand them more
            normalized_score = 0.75 + (boosted_score - 0.6) * 0.625
        
        # Apply sigmoid with temperature for final smoothing
        temperature = 2.5
        final_score = 1 / (1 + np.exp(-temperature * (normalized_score - 0.5)))
        
        return max(0.0, min(1.0, final_score))
    
    def _compute_similarity_hybrid(self, resume_text: str, job_text: str) -> float:
        """Compute hybrid similarity using transformer + keyword matching + TF-IDF with optimized weights."""
        # 1. Transformer embedding similarity
        # If using sentence-transformers, give it more weight (they're fine-tuned for similarity)
        # If using raw DistilBERT, give it less weight (not fine-tuned)
        if self.use_sentence_transformer:
            trans_weight = 0.4  # Sentence-transformers are reliable
        else:
            trans_weight = 0.2  # Raw DistilBERT is less reliable
        
        trans_score = self._compute_similarity(resume_text, job_text, return_attention=False)
        
        # 2. Keyword overlap score (most reliable signal)
        resume_keywords = set(extract_keywords(resume_text, max_keywords=30))
        job_keywords = set(extract_keywords(job_text, max_keywords=30))
        
        if job_keywords:
            keyword_overlap = len(resume_keywords & job_keywords) / len(job_keywords)
        else:
            keyword_overlap = 0.0
        
        # 3. TF-IDF similarity (reliable for exact matches)
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=2000)
            tfidf_matrix = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_text)])
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_score = 0.0
        
        # Adjust weights: more weight on reliable signals (keywords + TF-IDF)
        keyword_weight = 0.5 - (trans_weight * 0.5)  # Adjust based on transformer weight
        tfidf_weight = 1.0 - trans_weight - keyword_weight
        
        # Weighted combination with optimized weights
        final_score = trans_weight * trans_score + keyword_weight * keyword_overlap + tfidf_weight * tfidf_score
        
        # Apply additional normalization to increase discrimination
        # Stretch the score range to make differences more visible
        if final_score < 0.4:
            # Low scores: compress them
            normalized = final_score * 0.5
        elif final_score < 0.7:
            # Medium scores: expand them
            normalized = 0.2 + (final_score - 0.4) * 1.33
        else:
            # High scores: expand them more
            normalized = 0.6 + (final_score - 0.7) * 1.33
        
        return max(0.0, min(1.0, normalized))
    
    def _extract_attention_weights(self, resume_text: str, job_text: str) -> Dict:
        """Extract attention weights for interpretability."""
        if not self.enable_interpretability:
            return {}
        
        # Attention weights are only available when using raw transformer models
        # Sentence-transformers don't expose attention weights in the same way
        if self.use_sentence_transformer or self.tokenizer is None or self.model is None:
            # Return basic keyword-based interpretability instead
            resume_keywords = extract_keywords(resume_text, max_keywords=10)
            job_keywords = extract_keywords(job_text, max_keywords=10)
            matching_keywords = list(set(resume_keywords) & set(job_keywords))
            
            return {
                'resume_keywords': resume_keywords[:10],
                'job_keywords': job_keywords[:10],
                'matching_keywords': matching_keywords,
                'note': 'Using keyword-based interpretability (sentence-transformers mode)'
            }
        
        try:
            inputs = self._prepare_sequence_pair(resume_text, job_text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                
                if not hasattr(outputs, 'attentions') or not outputs.attentions:
                    return {}
                
                # Get attention from last layer (most relevant)
                attention = outputs.attentions[-1]  # Shape: (batch, heads, seq_len, seq_len)
                attention = attention[0].mean(dim=0).cpu().numpy()  # Average over heads
                
                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
                
                # Extract attention for resume and job tokens separately
                # Find [SEP] token position
                sep_idx = tokens.index('[SEP]') if '[SEP]' in tokens else len(tokens) // 2
                
                resume_tokens = tokens[1:sep_idx]  # Skip [CLS]
                job_tokens = tokens[sep_idx+1:-1]  # Skip [SEP] and last [SEP]
                
                # Get attention scores for resume-job interactions
                resume_job_attention = attention[1:sep_idx, sep_idx+1:-1]
                
                return {
                    'resume_tokens': resume_tokens,
                    'job_tokens': job_tokens,
                    'attention_matrix': resume_job_attention.tolist(),
                    'max_attention_indices': np.unravel_index(
                        np.argmax(resume_job_attention), 
                        resume_job_attention.shape
                    )
                }
        except Exception as e:
            # Fallback to keyword-based interpretability if attention extraction fails
            resume_keywords = extract_keywords(resume_text, max_keywords=10)
            job_keywords = extract_keywords(job_text, max_keywords=10)
            matching_keywords = list(set(resume_keywords) & set(job_keywords))
            
            return {
                'resume_keywords': resume_keywords[:10],
                'job_keywords': job_keywords[:10],
                'matching_keywords': matching_keywords,
                'note': f'Using keyword-based interpretability (error: {str(e)})'
            }
    
    def fit(self, job_postings: List[Dict], resume_texts: Optional[List[str]] = None):
        """
        Fit the transformer matcher.
        For transformer models, this mainly stores job postings.
        Optionally fine-tune on provided resume-job pairs.
        """
        self.job_meta = job_postings
        self.job_texts = [
            prepare_text_for_embeddings(j.get("text", "") + " " + j.get("title", ""))
            for j in job_postings
        ]
        self.is_fitted = True
        
        # Optional: Fine-tuning could be added here if resume_texts are provided
        # For now, we use the pre-trained model as-is
        
        return self
    
    def match(self, resume_text: str, k=5, return_interpretability: bool = False) -> List[Dict]:
        """
        Match resume against jobs using transformer model.
        
        Args:
            resume_text: Resume text to match
            k: Number of top matches to return
            return_interpretability: Whether to include attention weights in results
        
        Returns:
            List of match dictionaries with similarity scores and optional attention info
        """
        if not self.is_fitted:
            raise ValueError("Matcher not fitted. Call fit(job_postings) first.")
        
        resume_processed = prepare_text_for_embeddings(resume_text)
        scores = []
        interpretability_data = []
        
        # Process in batches for performance
        for i in range(0, len(self.job_texts), self.batch_size):
            batch_jobs = self.job_texts[i:i+self.batch_size]
            batch_scores = []
            
            for job_text in batch_jobs:
                # Use hybrid scoring for better discrimination
                score = self._compute_similarity_hybrid(resume_processed, job_text)
                batch_scores.append(score)
                
                # Extract attention if requested
                if return_interpretability and self.enable_interpretability:
                    attn_data = self._extract_attention_weights(resume_processed, job_text)
                    interpretability_data.append(attn_data)
                else:
                    interpretability_data.append({})
            
            scores.extend(batch_scores)
        
        # Apply relative ranking and scaling to make the best match stand out
        scores_array = np.array(scores)
        
        # Get top k indices before scaling
        idx = np.argsort(-scores_array)[:k]
        
        # Apply relative scaling: best match gets high score, others scale down
        if len(scores_array) > 0 and len(idx) > 0:
            top_scores = scores_array[idx]
            max_score = top_scores[0]  # Best score
            min_score = top_scores[-1] if len(top_scores) > 1 else max_score  # Worst in top k
            score_range = max_score - min_score if max_score > min_score else max_score
            
            # Dynamic scaling based on actual score distribution
            scaled_scores = []
            for rank, i in enumerate(idx):
                raw_score = scores_array[i]
                
                # Calculate how much better this score is relative to the worst in top k
                if score_range > 0.001:  # Avoid division by very small numbers
                    relative_quality = (raw_score - min_score) / score_range
                else:
                    relative_quality = 1.0 if rank == 0 else 0.5
                
                # Apply exponential scaling: top match gets 88-95%, others scale down
                # Use exponential decay so differences are more pronounced
                if rank == 0:
                    # Top match: 88-95% (best match should be clearly highest)
                    scaled = 0.88 + (relative_quality * 0.07)
                else:
                    # Other matches: scale down exponentially based on rank and relative quality
                    # Base score decreases with rank
                    base_score = 0.75 - (rank * 0.12)  # 75%, 63%, 51%, 39%, 27%...
                    
                    # Adjust based on relative quality within the range
                    quality_adjustment = relative_quality * 0.10
                    
                    # Apply penalty for being lower than top match
                    score_diff = max_score - raw_score
                    if score_diff > 0:
                        # More aggressive penalty for larger differences
                        penalty = min(0.20, score_diff * 3.0)
                        base_score -= penalty
                    
                    scaled = max(0.15, base_score + quality_adjustment)
                
                scaled_scores.append(float(scaled))
        else:
            scaled_scores = [float(scores_array[i]) for i in idx] if len(idx) > 0 else []
        
        results = []
        for rank, (i, scaled_score) in enumerate(zip(idx, scaled_scores)):
            job_text = self.job_meta[i].get("text", "") + " " + self.job_meta[i].get("title", "")
            matched_context = extract_matched_context(resume_text, job_text)
            
            result = {
                "job_id": self.job_meta[i].get("id"),
                "job_title": self.job_meta[i].get("title"),
                "similarity": scaled_score,  # Use scaled score instead of raw
                "job_meta": self.job_meta[i],
                "matched_context": matched_context,
                "rank": rank + 1
            }
            
            # Add interpretability data if available
            if return_interpretability and interpretability_data[i]:
                result["attention_data"] = interpretability_data[i]
            
            results.append(result)
        
        return results
    
    def get_feature_importance(self, resume_text: str, job_text: str) -> Dict:
        """
        Get feature importance using attention weights or keyword matching.
        Returns top matching tokens/keywords between resume and job.
        """
        attn_data = self._extract_attention_weights(resume_text, job_text)
        
        if not attn_data:
            return {"resume_keywords": [], "job_keywords": [], "matches": []}
        
        # If using keyword-based interpretability (sentence-transformers)
        if 'matching_keywords' in attn_data:
            resume_keywords = attn_data.get('resume_keywords', [])
            job_keywords = attn_data.get('job_keywords', [])
            matching_keywords = attn_data.get('matching_keywords', [])
            
            # Create matches from matching keywords
            matches = []
            for kw in matching_keywords[:10]:
                matches.append({
                    "resume_token": kw,
                    "job_token": kw,
                    "attention_score": 1.0  # Perfect match
                })
            
            return {
                "resume_keywords": resume_keywords[:20],
                "job_keywords": job_keywords[:20],
                "matches": matches,
                "note": attn_data.get('note', 'Keyword-based matching')
            }
        
        # If using attention-based interpretability (raw transformers)
        if 'attention_matrix' in attn_data:
            attention_matrix = np.array(attn_data['attention_matrix'])
            resume_tokens = attn_data['resume_tokens']
            job_tokens = attn_data['job_tokens']
            
            # Get top attention scores
            top_n = min(10, attention_matrix.size)
            flat_indices = np.argsort(attention_matrix.flatten())[-top_n:][::-1]
            
            matches = []
            for idx in flat_indices:
                resume_idx, job_idx = np.unravel_index(idx, attention_matrix.shape)
                if resume_idx < len(resume_tokens) and job_idx < len(job_tokens):
                    matches.append({
                        "resume_token": resume_tokens[resume_idx],
                        "job_token": job_tokens[job_idx],
                        "attention_score": float(attention_matrix[resume_idx, job_idx])
                    })
            
            return {
                "resume_keywords": resume_tokens[:20],  # Top resume tokens
                "job_keywords": job_tokens[:20],  # Top job tokens
                "matches": matches[:10]  # Top 10 token matches
            }
        
        # Fallback: return keyword-based matching
        resume_keywords = extract_keywords(resume_text, max_keywords=20)
        job_keywords = extract_keywords(job_text, max_keywords=20)
        matching_keywords = list(set(resume_keywords) & set(job_keywords))
        
        matches = [{"resume_token": kw, "job_token": kw, "attention_score": 1.0} 
                  for kw in matching_keywords[:10]]
        
        return {
            "resume_keywords": resume_keywords[:20],
            "job_keywords": job_keywords[:20],
            "matches": matches
        }

# Sample data and demo
SAMPLE_JOBS = [
    {"id":"J1", "title":"Data Scientist", "text":"Looking for a Data Scientist with Python, pandas, scikit-learn, statistics, machine learning, feature engineering, SQL, model deployment."},
    {"id":"J2", "title":"Frontend Engineer", "text":"Frontend developer with HTML, CSS, JavaScript, React, Next.js, responsive design, UI/UX collaboration."},
    {"id":"J3", "title":"DevOps Engineer", "text":"DevOps engineer experienced in Docker, Kubernetes, AWS, CI/CD pipelines, Terraform, monitoring, Linux."},
    {"id":"J4", "title":"Data Engineer", "text":"Data Engineer with ETL, SQL, Spark, Airflow, data pipelines, database design, Python."},
    {"id":"J5", "title":"Backend Engineer", "text":"Backend developer, Node.js, Express, REST APIs, databases, authentication, scaling."},
    {"id":"J6", "title":"ML Researcher", "text":"ML Researcher with deep learning, PyTorch, TensorFlow, publications, experimental design."}
]

SAMPLE_RESUME = "I build machine learning models in Python, use pandas and scikit-learn, and deploy models to production."

def demo():
    print("=== TF-IDF Matcher Demo ===")
    tfm = TFIDFMatcher().fit(SAMPLE_JOBS)
    tf_matches = tfm.match(SAMPLE_RESUME, k=3)
    for m in tf_matches:
        print(m['job_id'], m['job_title'], f"{m['similarity']:.4f}")

    if SentenceTransformer is not None:
        print("\n=== Embedding Matcher Demo (sentence-transformers) ===")
        em = EmbeddingMatcher(model_name="all-MiniLM-L6-v2", use_faiss=False)
        em.fit(SAMPLE_JOBS)
        emb_matches = em.match(SAMPLE_RESUME, k=3)
        for m in emb_matches:
            print(m['job_id'], m['job_title'], f"{m['similarity']:.4f}")
    else:
        print("\nInstall sentence-transformers to try embedding-based matching.")
    
    if TRANSFORMERS_AVAILABLE:
        print("\n=== Transformer Matcher Demo (BERT-based) ===")
        try:
            tm = TransformerMatcher(
                model_name="distilbert-base-uncased",
                model_type="classification",
                batch_size=4,
                enable_interpretability=True
            )
            tm.fit(SAMPLE_JOBS)
            trans_matches = tm.match(SAMPLE_RESUME, k=3, return_interpretability=True)
            for m in trans_matches:
                print(m['job_id'], m['job_title'], f"{m['similarity']:.4f}")
                if 'attention_data' in m and m['attention_data']:
                    print(f"  -> Interpretability: {len(m['attention_data'].get('resume_tokens', []))} resume tokens matched")
        except Exception as e:
            print(f"Transformer matcher not available: {e}")
    else:
        print("\nInstall transformers and torch to try transformer-based matching: pip install transformers torch")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("resume_matcher.py - module loaded. Run 'python resume_matcher.py demo' to see a demo.")
    