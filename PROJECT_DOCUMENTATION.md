# Intelligent Resume & Career Advisor - Project Documentation

**Project:** Intelligent Resume & Career Advisor
**Domain:** Artificial Intelligence / NLP
**Version:** 2.0 (Deep Advisor)

---

## 1. Project Overview
The **Intelligent Resume & Career Advisor** is an AI-powered application designed to bridge the gap between job seekers and their target roles. Unlike simple keyword matchers, this system uses a hybrid approach combining **Deep Learning (BERT/SBERT)** and **Traditional NLP (TF-IDF)** to provide holistic career guidance.

### Key Features
1.  **Resume Parsing**: Extractor for PDF, JSON, and Text formats.
2.  **Intelligent Validation**: Automatically identifies and rejects invalid documents (e.g., invoices, images).
3.  **Hybrid Matching Engine**:
    *   **Deep Analyzer**: Uses BERT for category classification and SBERT for semantic similarity.
    *   **TF-IDF Matcher**: Uses statistical keyword analysis for fast filtering.
4.  **Skill Gap Analysis**: Identifies exactly which skills a candidate is missing for a specific job.
5.  **Career Advisor**: Recommends personalized learning resources (courses, guides) for missing skills.
6.  **Resume Quality Feedback**: improved Logic to analyze structure, action verbs, and quantification metrics.

---

## 2. Technical Architecture

### Tech Stack
*   **Language**: Python 3.10+
*   **Frontend**: Streamlit
*   **NLP & ML**:
    *   `sentence-transformers` (SBERT)
    *   `transformers` (HuggingFace BERT)
    *   `scikit-learn` (TF-IDF, Cosine Similarity)
    *   `torch` (PyTorch Backend)
*   **Data Processing**: `pandas`, `numpy`, `PyPDF2`

### Dataset Used
*   **Job Corpus**: Custom trained dataset (`jobs_fallback.csv`).
*   **Content**: Contains specialized job descriptions for roles like Data Scientist, Java Developer, Web Designer, and HR Manager.
*   **Preprocessing**: The system automatically de-duplicates and indexes this dataset on startup.

---

## 3. Installation & Startup Guide

### Prerequisites
*   Python 3.8 or higher installed.
*   A folder containing the project files.
*   (Optional) GPU for faster inference.

### Setup Instructions
1.  **Clone/Copy the Project**: Ensure all files are in a local directory.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Required libraries: `streamlit`, `pandas`, `sentence-transformers`, `transformers`, `torch`, `PyPDF2`*

3.  **Model Setup**:
    *   Ensure the `models/` directory contains your trained `bert_classifier` and `sbert_matcher` folders.
    *   *Note: If models are missing, the system gracefully falls back to TF-IDF only.*

4.  **Run the Application**:
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 4. User Guide (How to Use)
1.  **Upload Resume**: Drag & drop your Resume PDF or pasting the text.
2.  **Validation**: The system first checks if the file is a valid resume. If invalid, processing stops to save resources.
3.  **Select Method**: Choose "Deep Analyzer (BERT + SBERT)" for best results or "TF-IDF" for speed.
4.  **View Analysis**:
    *   **Job Matches**: See top matching jobs from the dataset.
    *   **Skill Gaps**: Expand a job to see green (matched) vs red (missing) skills.
    *   **Learning Paths**: Click the suggested resources to learn missing skills.
    *   **Resume Feedback**: Read the automated critique to improve your document's impact.

---

## 5. Directory Structure
```
/
├── streamlit_app.py      # Main Frontend Application
├── resume_matcher.py     # Core Backend Logic (Matchers, Advisors)
├── jobs_fallback.csv     # Custom Job Dataset
├── requirements.txt      # Python Dependencies
├── verify_advisor.py     # Verification Scripts
└── models/               # Directory for Trained Models
    ├── bert_classifier/  # BERT Model Artifacts
    └── sbert_matcher/    # SBERT Model Artifacts
```

---
**Submission Date**: December 2025
