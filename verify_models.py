
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelCheck")

def check_models():
    models_dir = "models"
    bert_path = os.path.join(models_dir, "bert_classifier")
    sbert_path = os.path.join(models_dir, "sbert_matcher")
    
    print(f"Checking models in {os.path.abspath(models_dir)}...")
    
    if not os.path.exists(bert_path):
        print(f"❌ BERT path missing: {bert_path}")
        return False
        
    if not os.path.exists(sbert_path):
        print(f"❌ SBERT path missing: {sbert_path}")
        return False
        
    # Check for large files (safetensors or bin) to verify LFS download
    # BERT model.safetensors should be > 100MB
    bert_model_file = os.path.join(bert_path, "model.safetensors")
    if os.path.exists(bert_model_file):
        size_mb = os.path.getsize(bert_model_file) / (1024 * 1024)
        print(f"BERT model size: {size_mb:.2f} MB")
        if size_mb < 1.0: # Arbitrary small threshold, usually pointers are 1KB
            print("❌ BERT model file appears to be an LFS pointer (download pending).")
            return False
    else:
        print("❌ BERT model.safetensors missing.")
        return False

    # SBERT model.safetensors
    sbert_model_file = os.path.join(sbert_path, "model.safetensors") 
    if os.path.exists(sbert_model_file):
        size_mb = os.path.getsize(sbert_model_file) / (1024 * 1024)
        print(f"SBERT model size: {size_mb:.2f} MB")
        if size_mb < 1.0:
            print("❌ SBERT model file appears to be an LFS pointer (download pending).")
            return False
    else:
        print("❌ SBERT model.safetensors missing.")
        return False
        
    print("✅ System paths look correct for models.")
    
    # Try loading
    try:
        from resume_matcher import DeepAnalyzerMatcher, SAMPLE_JOBS
        print("Attempting to load DeepAnalyzerMatcher...")
        matcher = DeepAnalyzerMatcher(models_dir=models_dir)
        matcher.fit(SAMPLE_JOBS)
        print("✅ DeepAnalyzerMatcher successfully loaded and fitted!")
        return True
    except Exception as e:
        print(f"❌ Failed to load DeepAnalyzerMatcher: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_models()
    if not success:
        sys.exit(1)
