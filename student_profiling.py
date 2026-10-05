import pandas as pd
import re
import ast
import os

print("=" * 60)
print("SKILLBRIDGE AI - STUDENT PROFILING & RESUME ANALYSIS")
print("=" * 60)

# ==========================================================
# 1. Build Skill Knowledge Base
# ==========================================================
print("\nBuilding Skill Knowledge Base from Job Postings Dataset...")

job_path = "SkillBridge AI/dataset/job_postings_dataset.csv"
if not os.path.exists(job_path):
    job_path = "dataset/job_postings_dataset.csv"

jobs_df = pd.read_csv(job_path)

# Extract skills from job_skill_set
all_skills = set()
for skill_str in jobs_df['job_skill_set'].dropna():
    try:
        # Convert string representation of list to actual list
        skills_list = ast.literal_eval(skill_str)
        if isinstance(skills_list, list):
            for skill in skills_list:
                all_skills.add(skill.lower().strip())
    except (ValueError, SyntaxError):
        # In case the string isn't a proper list format
        pass

# Convert to a sorted list for consistent usage
skill_kb = sorted(list(all_skills))
print(f"Successfully extracted {len(skill_kb)} unique skills for the knowledge base.")

# ==========================================================
# 2. Resume Analysis & Skill Extraction Function
# ==========================================================
def extract_skills_from_text(text, knowledge_base):
    """
    Scans the provided text (like a resume) and matches it against the knowledge base.
    """
    if not text:
        return []
    
    # Simple text cleaning
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    extracted = []
    # Basic keyword matching (can be upgraded to NLP like SpaCy later)
    for skill in knowledge_base:
        # Avoid matching very short generic words, enforce boundaries
        if len(skill) > 2:
            # Using regex to find whole words only
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                extracted.append(skill)
                
    return extracted

# ==========================================================
# 3. Student Profiling Function
# ==========================================================
def create_student_profile(name, career_goal, resume_text, current_level="Beginner"):
    """
    Creates a standardized student profile.
    """
    print(f"\nAnalyzing resume for {name}...")
    extracted_skills = extract_skills_from_text(resume_text, skill_kb)
    
    profile = {
        "Name": name,
        "Career Goal": career_goal,
        "Current Level": current_level,
        "Extracted Skills": extracted_skills,
        "Skill Count": len(extracted_skills)
    }
    return profile

# ==========================================================
# 4. Test with a Simulated Resume
# ==========================================================
sample_resume = """
John Doe
Email: john.doe@email.com
Objective: Aspiring Data Scientist looking for opportunities.
Experience:
- Worked on python projects involving data analysis.
- Used sql for database management.
- Built predictive models using machine learning and pandas.
- Familiar with deep learning and basic statistics.
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING RESUME ANALYSIS & PROFILING")
    print("=" * 60)

    student_profile = create_student_profile(
        name="John Doe",
        career_goal="Data Scientist",
        resume_text=sample_resume,
        current_level="Beginner"
    )

    # Display Profile
    for key, value in student_profile.items():
        if key == "Extracted Skills":
            print(f"{key}: {', '.join(value[:10])}" + ("..." if len(value) > 10 else ""))
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
