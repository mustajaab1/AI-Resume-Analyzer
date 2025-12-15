from resume_matcher import SAMPLE_JOBS

print(f"Loaded {len(SAMPLE_JOBS)} jobs.")
if len(SAMPLE_JOBS) > 2:
    print("✅ Successfully loaded jobs from CSV.")
    # Check first job
    print(f"First Job: {SAMPLE_JOBS[0]}")
else:
    print("❌ Still using fallback jobs.")
