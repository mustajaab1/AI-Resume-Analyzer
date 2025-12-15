from resume_matcher import ResumeFeedback

feedback = ResumeFeedback()

print("=== Testing Improved Feedback Logic ===")

# 1. Weak Resume
weak = "I worked at a company. I did programming."
print(f"\n[Weak Resume] '{weak}'")
tips = feedback.generate_feedback(weak)
for t in tips:
    print(t)

# 2. Strong Resume
strong = """
Experience
Software Engineer at Google.
- Spearheaded a new API initiative.
- Reduced latency by 50%.
- Orchestrated the deployment of 5 microservices.

Education
BS Computer Science, Stanford.

Skills
Python, Java, Leadership, Communication.
"""
print(f"\n[Strong Resume] (Contains headers, strong verbs, metrics)")
tips = feedback.generate_feedback(strong)
for t in tips:
    print(t)

# Check specifically for new checks
if any("Too Short" in t for t in tips): # Weak resume is short
    print("\n✅ Caught 'Too Short' issue")
else:
    print("\n❌ Failed 'Too Short' check")

# Check Metrics check
if not any("Lack of Metrics" in t for t in feedback.generate_feedback(strong)):
    print("✅ Correctly identified metrics in strong resume")
else:
    print("❌ False negative on metrics")
