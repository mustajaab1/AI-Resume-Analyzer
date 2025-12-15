"""
resume_matcher.py

Provides:
- TF-IDF based search (keyword-based)
- Deep Analyzer search (BERT classification + SBERT semantic matching)

Refactored to only focus on these two methods as per user request.
"""

from typing import List, Dict, Optional, Tuple, Any
import re
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports for Deep Analyzer
try:
    import torch
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from sentence_transformers import SentenceTransformer, util
    DEEP_ANALYZER_AVAILABLE = True
except ImportError:
    DEEP_ANALYZER_AVAILABLE = False
    print("Warning: transformers, torch, or sentence-transformers not installed. Deep Analyzer will not work.")

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
    Uses TF-IDF mainly here to avoid circular dependency or heavy loading.
    """
    if not resume_text or not job_text:
        return resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
    
    # Better text chunking: split by sentences, bullet points, and newlines
    chunks = re.split(r'[.!?\n•\-\*]+', resume_text)
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
    
    # Fallbacks for chunking
    if len(chunks) < 3:
        line_chunks = resume_text.split('\n')
        line_chunks = [chunk.strip() for chunk in line_chunks if len(chunk.strip()) > 10]
        if len(line_chunks) > len(chunks):
            chunks = line_chunks
            
    if len(chunks) < 3:
        comma_chunks = re.split(r'[,;]+', resume_text)
        comma_chunks = [chunk.strip() for chunk in comma_chunks if len(chunk.strip()) > 20]
        if len(comma_chunks) > len(chunks):
            chunks = comma_chunks

    if not chunks or len(chunks) < 2:
        chunk_size = 100
        chunks = [resume_text[i:i+chunk_size].strip() for i in range(0, len(resume_text), chunk_size)]
        chunks = [chunk for chunk in chunks if len(chunk) > 10]
    
    if not chunks:
        return resume_text[:300] + "..." if len(resume_text) > 300 else resume_text
    
    try:
        # Use TF-IDF for context extraction efficiency
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=5000)
        all_texts = [clean_text(job_text)] + [clean_text(chunk) for chunk in chunks]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        job_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        scores = cosine_similarity(job_vector, chunk_vectors)[0]
        
        top_indices = np.argsort(-scores)[:max_chunks]
        
        matched_chunks = []
        seen_chunks = set()
        
        # Threshold
        min_threshold = 0.1 if any(scores > 0.15) else 0.0
        
        for idx in top_indices:
            chunk = chunks[idx]
            chunk_lower = chunk.lower()[:50]
            if chunk_lower not in seen_chunks:
                if scores[idx] >= min_threshold or min_threshold == 0.0:
                    matched_chunks.append(chunk)
                    seen_chunks.add(chunk_lower)
        
        if matched_chunks:
            result = " ... ".join(matched_chunks)
            if len(result) > 500:
                result = result[:500] + "..."
            return result
        else:
            return " ... ".join(chunks[:max_chunks]) if chunks else resume_text[:300] + "..."

    except Exception:
        return " ... ".join(chunks[:max_chunks]) if chunks else resume_text[:300] + "..."

def is_likely_name(word: str) -> bool:
    """Check if a word is likely a person's name."""
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
    
    if word.lower() in common_names:
        return True
    
    if word[0].isupper() and len(word) > 2:
        common_caps = {
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'usa', 'uk', 'nyc', 'la', 'sf', 'ny', 'ca', 'tx', 'fl', 'il',
            'python', 'java', 'sql', 'html', 'css', 'react', 'node', 'aws' # Added tech explicitly to avoid false positives
        }
        if word.lower() not in common_caps:
            return True
    
    return False

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text, filtering out names, punctuation, and irrelevant tokens."""
    try:
        cleaned = clean_text(text)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned)
        if not words:
            return []
        
        from collections import Counter
        word_freq = Counter(words)
        
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
            'through', 'throughout', 'toward', 'under', 'underneath', 'until', 'within', 'without',
            'resume', 'curriculum', 'vitae', 'email', 'phone', 'address', 'summary', 'objective'
        }
        
        filtered_keywords = []
        for word, count in word_freq.most_common(max_keywords * 3):
            word_lower = word.lower()
            if word_lower in stop_words:
                continue
            if is_likely_name(word):
                continue
            if len(word) < 3:
                continue
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
        # Sort desc
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

class DeepAnalyzerMatcher:
    """
    Combines BERT Classification (for category detection) + SBERT (for semantic similarity).
    Requires 'models/bert_classifier' and 'models/sbert_matcher' to be present.
    """
    def __init__(self, models_dir: str = "models"):
        if not DEEP_ANALYZER_AVAILABLE:
            raise RuntimeError("Deep Analyzer dependencies missing.")
        
        self.models_dir = models_dir
        self.bert_path = os.path.join(models_dir, "bert_classifier")
        self.sbert_path = os.path.join(models_dir, "sbert_matcher")
        
        self.bert_available = os.path.exists(self.bert_path)
        self.sbert_available = os.path.exists(self.sbert_path)
        
        if not self.bert_available or not self.sbert_available:
            raise ValueError(f"Models not found in {models_dir}. BERT: {self.bert_available}, SBERT: {self.sbert_available}")

        # Load models
        print("Loading BERT Classifier...")
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)
        self.bert_model = BertForSequenceClassification.from_pretrained(self.bert_path)
        
        print("Loading SBERT Matcher...")
        self.sbert_model = SentenceTransformer(self.sbert_path)
        
        self.job_meta = []
        self.job_embs = None
        
        # Load label map for BERT
        self.label_map = {}
        map_path = os.path.join(self.bert_path, "label_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                self.label_map = json.load(f)

    def fit(self, job_postings: List[Dict]):
        self.job_meta = job_postings
        # Prepare job texts for SBERT
        job_texts = [j.get("text", "") + " " + j.get("title", "") for j in job_postings]
        self.job_embs = self.sbert_model.encode(job_texts, convert_to_tensor=True)
        return self

    def match(self, resume_text: str, k=5) -> List[Dict]:
        if self.job_embs is None:
            raise ValueError("Matcher not fitted. Call fit(job_postings) first.")

        # 1. BERT Classification
        inputs = self.bert_tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = self.bert_model(**inputs).logits
        pred_idx = logits.argmax().item()
        category = self.label_map.get(str(pred_idx), str(pred_idx))
        
        # 2. SBERT Semantic Matching
        resume_emb = self.sbert_model.encode(resume_text, convert_to_tensor=True)
        scores = util.cos_sim(resume_emb, self.job_embs)[0]
        
        # Top k
        top_k_indices = scores.argsort(descending=True)[:k]
        
        results = []
        for idx in top_k_indices:
            idx = idx.item()
            score = scores[idx].item()
            job = self.job_meta[idx]
            
            # Boost score if category matches job
            # Heuristic: 5% boost
            if category.lower() in job.get('title', '').lower() or category.lower() in job.get('text', '').lower():
                score += 0.05
            
            matched_context = extract_matched_context(resume_text, job.get('text', ''))
            
            results.append({
                "job_id": job.get("id"),
                "job_title": job.get("title"),
                "similarity": score,
                "job_meta": job,
                "matched_context": f"detected Category: {category} | " + matched_context
            })
            
        return results

class SkillGapAnalyzer:
    """
    Analyzes missing skills by comparing resume keywords to job description keywords.
    """
    def __init__(self):
        # Common technical skills to look for (expanded list)
        self.tech_skills = {
            'python', 'java', 'c++', 'javascript', 'html', 'css', 'react', 'angular', 'vue', 
            'node', 'express', 'django', 'flask', 'sql', 'nosql', 'mongodb', 'postgresql', 
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd', 'linux',
            'machine learning', 'deep learning', 'nlp', 'pytorch', 'tensorflow', 'scikit-learn',
            'pandas', 'numpy', 'statistics', 'data visualization', 'tableau', 'power bi',
            'agile', 'scrum', 'jira', 'communication', 'leadership', 'teamwork'
        }

    def analyze(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        resume_keywords = set(extract_keywords(resume_text, max_keywords=50))
        job_keywords = set(extract_keywords(job_text, max_keywords=50))
        
        # Filter for known tech skills to reduce noise
        resume_skills = {k for k in resume_keywords if k in self.tech_skills or k in job_keywords}
        job_skills = {k for k in job_keywords if k in self.tech_skills}
        
        # If job_skills is empty (extraction failed), fallback to raw intersection
        if not job_skills:
            job_skills = job_keywords
            
        missing_skills = list(job_skills - resume_skills)
        matching_skills = list(job_skills & resume_skills)
        
        return {
            "missing_skills": missing_skills[:10], # Top 10
            "matching_skills": matching_skills,
            "match_percentage": len(matching_skills) / len(job_skills) if job_skills else 0.0
        }

class CareerAdvisor:
    """
    Provides learning resources for missing skills.
    """
    def __init__(self):
        self.resources = {
            "python": "Course: 'Python for Everybody' on Coursera / Docs: python.org",
            "machine learning": "Course: Andrew Ng's Machine Learning on Coursera",
            "sql": "Practice: LeetCode SQL 50 / Course: Khan Academy SQL",
            "react": "Docs: react.dev / Tutorial: Fullstack Open",
            "aws": "Cert: AWS Certified Cloud Practitioner / Guide: AWS Ramp-Up Guide",
            "docker": "Guide: Docker Get Started / Course: Docker Mastery on Udemy",
            "kubernetes": "Docs: kubernetes.io / Course: K8s for Beginners",
            "statistics": "Book: 'Naked Statistics' / Course: Khan Academy Statistics",
            "nlp": "Course: Hugging Face NLP Course / Stanford CS224n",
            "git": "Guide: Dangit, Git! / Pro Git Book",
            "default": "Search on Coursera, Udemy, or freeCodeCamp for this skill."
        }

    def suggest_resources(self, missing_skills: List[str]) -> Dict[str, str]:
        suggestions = {}
        for skill in missing_skills:
            # Simple fuzzy lookup or direct map
            res = self.resources.get(skill.lower())
            if not res:
                # Try simple partial match
                for k, v in self.resources.items():
                    if k in skill.lower():
                        res = v
                        break
            
            suggestions[skill] = res if res else self.resources["default"]
        return suggestions

class ResumeFeedback:
    """
    Provides constructive feedback on resume quality.
    """
    def generate_feedback(self, text: str) -> List[str]:
        feedback = []
        text_lower = text.lower()
        
        # 1. Structure / Sections Check
        common_sections = ['experience', 'education', 'skills', 'projects', 'summary', 'objective', 'certifications']
        found_sections = [s for s in common_sections if s in text_lower]
        if 'experience' not in found_sections and 'history' not in text_lower:
            feedback.append("⚠️ **Missing Experience Section**: Recruiters look for 'Experience' or 'Work History' first.")
        if 'education' not in found_sections and 'academic' not in text_lower:
            feedback.append("⚠️ **Missing Education Section**: It's important to list your academic background clearly.")

        # 2. Action Verbs Check
        strong_verbs = {
            'managed', 'created', 'developed', 'led', 'designed', 'implemented', 'improved', 'increased', 'reduced',
            'launched', 'engineered', 'orchestrated', 'spearheaded', 'formulated', 'initiated'
        }
        found_verbs = {v for v in strong_verbs if v in text_lower} # Use set for unique matches
        if len(found_verbs) < 3:
            feedback.append("⚠️ **Weak Action Verbs**: Your resume uses few strong action words. try adding verbs like 'Spearheaded', 'Orchestrated', or 'Implemented'.")
        elif len(found_verbs) < 5:
            feedback.append("💡 **Variety**: You have some good action verbs, but could use more variety to describe your contributions.")
            
        # 3. Quantification Check (Metrics)
        # Look for %, $, "increased by", "reduced by", or numbers followed by "people", "users", "revenue"
        metric_patterns = [
            r'\d+%', 
            r'\$\d+', 
            r'increased.*by', 
            r'reduced.*by',
            r'improved.*by', 
            r'\d+\s+(people|users|clients|customers|revenue|sales)'
        ]
        has_metrics = any(re.search(p, text_lower) for p in metric_patterns)
        
        if not has_metrics:
            feedback.append("⚠️ **Lack of Metrics**: quantifiability is key. Try adding details like 'Increased sales by 15%' or 'Managed a team of 10 people'.")
            
        # 4. Soft Skills
        soft_skills = {
            'communication', 'leadership', 'teamwork', 'problem solving', 'adaptability', 
            'collaboration', 'time management', 'critical thinking', 'creativity'
        }
        found_soft = [s for s in soft_skills if s in text_lower]
        if not found_soft:
            feedback.append("💡 **Soft Skills**: Don't forget to mention soft skills like 'Collaboration' or 'Leadership' if relevant.")
            
        # 5. Length/Detail Check
        word_count = len(text.split())
        if word_count < 150:
            feedback.append("⚠️ **Too Short**: Your resume seems very brief (< 150 words). Consider adding more details about your projects and roles.")
        elif word_count > 1000:
            feedback.append("💡 **Length**: Your resume is quite long (> 1000 words). Ensure it's concise and relevant.")

        if not feedback:
            feedback.append("✅ **Great Job**: Your resume is well-structured, uses strong verbs, and includes metrics!")
            
        return feedback

def validate_resume_content(text: str) -> Dict[str, Any]:
    """
    Validate if the text content appears to be a resume.
    Returns a dict with 'is_valid', 'confidence', and 'issues'.
    """
    if not text or not text.strip():
        return {"is_valid": False, "confidence": 0.0, "issues": ["Empty text"]}
    
    text = text.lower()
    issues = []
    score = 0.0
    
    # 1. Length Check
    word_count = len(text.split())
    if word_count < 50:
        issues.append(f"Text too short ({word_count} words). Resumes usually have >100 words.")
        score -= 0.2
    elif word_count > 4000:
        issues.append(f"Text too long ({word_count} words). Might be a paper or book.")
        score -= 0.1
    else:
        score += 0.2  # Reasonable length
    
    # 2. Section Headers Check (Critical for structure)
    common_headers = [
        'experience', 'education', 'skills', 'projects', 'summary', 
        'profile', 'objective', 'work history', 'qualifications',
        'certifications', 'achievements', 'languages', 'technical skills',
        'employment history', 'academic background', 'professional experience'
    ]
    
    # Check for direct presence of headers (sometimes they are standalone lines, but simple containment is a start)
    header_count = sum(1 for h in common_headers if h in text)
    
    if header_count < 2:
        issues.append("Missing standard resume sections (e.g. Experience, Education, Skills)")
        # Penalize heavily if no headers found
        score -= 0.3
    else:
        # Boost confidence significantly if multiple headers are found
        score += min(0.5, header_count * 0.15)
        
    # 3. Contact Info Check
    has_email = '@' in text and '.' in text
    has_phone = bool(re.search(r'\d{3}[-\s.]?\d{3}[-\s.]?\d{4}', text))
    
    if has_email:
        score += 0.2
    if has_phone:
        score += 0.1
        
    if not (has_email or has_phone):
        issues.append("No contact info (email/phone) detected")
        
    # 4. Keyword Density Check
    keywords = extract_keywords(text, max_keywords=20)
    if len(keywords) > 5:
        score += 0.1
    else:
        issues.append("Few professional keywords found")

    # Final decision
    # Normalize score 0-1
    confidence = max(0.0, min(1.0, score))
    
    # Stricter Rule: MUST have at least 2 headers OR (1 header + contact info) to be valid
    # Just generic text with length shouldn't pass
    structure_check = (header_count >= 2) or (header_count >= 1 and (has_email or has_phone))
    
    is_valid = confidence >= 0.5 and structure_check
    
    if not structure_check and is_valid:
        # Downgrade if structure is missing despite high score (unlikely but safe)
        is_valid = False
        issues.append("Structure unclear: contains keywords but lacks resume sections")
    
    return {
        "is_valid": is_valid,
        "confidence": confidence,
        "issues": issues
    }

# Sample data for demo
# Load jobs from CSV if available, otherwise use samples
def load_jobs_from_csv(file_path: str = "jobs_fallback.csv") -> List[Dict]:
    jobs = []
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Remove duplicates
            df = df.drop_duplicates(subset=["Job Title", "Job Description"])
            
            for i, row in df.iterrows():
                jobs.append({
                    "id": f"JOB_{i}",
                    "title": str(row.get("Job Title", "Unknown Role")),
                    "text": str(row.get("Job Description", ""))
                })
            print(f"Loaded {len(jobs)} jobs from {file_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
    return jobs

# Initial Sample Jobs (Fallback)
_FALLBACK_JOBS = [
    {"id":"J1", "title":"Data Scientist", "text":"Looking for a Data Scientist with Python, pandas, scikit-learn, statistics, machine learning..."},
    {"id":"J2", "title":"Frontend Engineer", "text":"Frontend developer with HTML, CSS, JavaScript, React, Next.js..."},
]

# Try to load from CSV, fallback if empty
SAMPLE_JOBS = load_jobs_from_csv("jobs_fallback.csv")
if not SAMPLE_JOBS:
    print("Using fallback sample jobs.")
    SAMPLE_JOBS = _FALLBACK_JOBS
SAMPLE_RESUME = "I am a Data Scientist experienced in Python and Machine Learning."

def demo():
    print("=== TF-IDF Matcher Demo ===")
    tfm = TFIDFMatcher().fit(SAMPLE_JOBS)
    matches = tfm.match(SAMPLE_RESUME, k=2)
    for m in matches:
        print(f"{m['job_title']}: {m['similarity']:.3f}")

    if DEEP_ANALYZER_AVAILABLE:
        print("\n=== Deep Analyzer Demo ===")
        # Check if local models exist before trying
        if os.path.exists("models/bert_classifier") and os.path.exists("models/sbert_matcher"):
            try:
                dam = DeepAnalyzerMatcher()
                dam.fit(SAMPLE_JOBS)
                matches = dam.match(SAMPLE_RESUME, k=2)
                for m in matches:
                    print(f"{m['job_title']}: {m['similarity']:.3f} (Ctx: {m['matched_context'][:30]}...)")
            except Exception as e:
                print(f"Deep Analyzer Error: {e}")
        else:
            print("Local models not found in 'models/' directory. Please train/download them.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("resume_matcher.py loaded.")