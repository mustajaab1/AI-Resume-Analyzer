from resume_matcher import validate_resume_content

print("=== Testing Resume Validation ===")

# 1. Valid Resume
valid_resume = """
John Doe
Software Engineer
Email: john.doe@example.com | Phone: 123-456-7890

Summary
Experienced software engineer with 5 years in Python and Machine Learning.

Experience
Software Developer at Tech Co
- Built ML models using Scikit-Learn.
- Deployed APIs with Flask.

Education
BS Computer Science, University of Technology

Skills
Python, SQL, Docker, AWS
"""
res1 = validate_resume_content(valid_resume)
print(f"Valid Resume: {res1['is_valid']} (Confidence: {res1['confidence']:.2f})")
if not res1['is_valid']:
    print(f"Issues: {res1['issues']}")

# 2. Invalid Text (Random snippet)
invalid_text = """
This is just a random paragraph about nothing effectively. 
It does not contain any resume sections or contact info. 
Just checking if the validator catches this.
"""
res2 = validate_resume_content(invalid_text)
print(f"Invalid Text: {res2['is_valid']} (Confidence: {res2['confidence']:.2f})")
if not res2['is_valid']:
    print(f"Issues: {res2['issues']}")

# 3. Short Text
short_text = "My name is Bob."
res3 = validate_resume_content(short_text)
print(f"Short Text: {res3['is_valid']} (Confidence: {res3['confidence']:.2f})")
