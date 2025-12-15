from resume_matcher import SkillGapAnalyzer, CareerAdvisor, ResumeFeedback

print("=== Testing Advisor Features ===")

# 1. Skill Gap Analysis
print("\n[1] Skill Gap Analyzer")
analyzer = SkillGapAnalyzer()
resume_text = "Experienced Developer with Python and Machine Learning knowledge."
job_text = "Looking for Python Developer with SQL, AWS, and Docker experience."

gap = analyzer.analyze(resume_text, job_text)
print(f"Resume: {resume_text}")
print(f"Job Needed: {job_text}")
print(f"Detected Missing Skills: {gap['missing_skills']}")
print(f"Detected Matching Skills: {gap['matching_skills']}")

if 'sql' in gap['missing_skills'] and 'python' in gap['matching_skills']:
    print("✅ Gap Analysis passed (Found SQL missing, Python matched)")
else:
    print("❌ Gap Analysis failed")

# 2. Career Advisor
print("\n[2] Career Advisor")
advisor = CareerAdvisor()
resources = advisor.suggest_resources(gap['missing_skills'])
print("Suggested Resources:")
for skill, res in resources.items():
    print(f"- {skill}: {res}")
    
if 'sql' in resources:
    print("✅ Resource suggestion passed")
else:
    print("❌ Resource suggestion failed")

# 3. Resume Feedback
print("\n[3] Resume Feedback")
feedback = ResumeFeedback()

weak_resume = "I worked at a company and did coding."
print(f"Testing Weak Resume: '{weak_resume}'")
tips = feedback.generate_feedback(weak_resume)
for t in tips:
    print(t)
    
if any("Weak Action Verbs" in t for t in tips):
    print("✅ Weak verbs detected")
else:
    print("❌ Weak verbs detection failed")
    
strong_resume = "Managed a team of 5, increased efficiency by 20%, and developed Python APIs."
print(f"\nTesting Strong Resume: '{strong_resume}'")
tips_strong = feedback.generate_feedback(strong_resume)
for t in tips_strong:
    print(t)

if any("Great Job" in t for t in tips_strong):
    print("✅ Good quality detected")
else:
    print("❌ Good quality detection failed") # Note: might fail if other checks trigger, but looking for positive signal
