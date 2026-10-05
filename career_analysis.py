import pandas as pd
import ast
import os
from collections import Counter

print("=" * 60)
print("SKILLBRIDGE AI - CAREER ANALYSIS & SKILL GAP")
print("=" * 60)

# ==========================================================
# 1. Load Job Postings Dataset
# ==========================================================
job_path = "SkillBridge AI/dataset/job_postings_dataset.csv"
if not os.path.exists(job_path):
    job_path = "dataset/job_postings_dataset.csv"

jobs_df = pd.read_csv(job_path)

# ==========================================================
# 2. Career Matching Function
# ==========================================================
def get_required_skills(target_career, top_n=10):
    """
    Finds the most commonly required skills for a given career goal
    by analyzing the job postings dataset.
    """
    target_career = target_career.lower()
    
    # Filter jobs that match the target career in their job title
    matched_jobs = jobs_df[jobs_df['job_title'].str.lower().str.contains(target_career, na=False)]
    
    if matched_jobs.empty:
        # Graceful fallback: match on individual words in job titles
        words = [w for w in target_career.split() if len(w) > 3]
        for w in words:
            word_matches = jobs_df[jobs_df['job_title'].str.lower().str.contains(w, na=False)]
            if not word_matches.empty:
                matched_jobs = word_matches
                break
                
    if matched_jobs.empty:
        # Graceful fallback: match to relevant domain category
        if any(t in target_career for t in ['data', 'software', 'engineer', 'developer', 'python', 'cloud', 'ai', 'tech', 'system', 'web']):
            matched_jobs = jobs_df[jobs_df['category'] == 'INFORMATION-TECHNOLOGY']
        elif any(t in target_career for t in ['finance', 'account', 'audit']):
            matched_jobs = jobs_df[jobs_df['category'] == 'FINANCE']
        elif any(t in target_career for t in ['hr', 'human', 'talent', 'people']):
            matched_jobs = jobs_df[jobs_df['category'] == 'HR']
        elif any(t in target_career for t in ['sale', 'sales']):
            matched_jobs = jobs_df[jobs_df['category'] == 'SALES']
        elif any(t in target_career for t in ['manager', 'business', 'project', 'product', 'marketing']):
            matched_jobs = jobs_df[jobs_df['category'] == 'BUSINESS-DEVELOPMENT']
        else:
            matched_jobs = jobs_df
        
    all_matched_skills = []
    
    for skill_str in matched_jobs['job_skill_set'].dropna():
        try:
            skills_list = ast.literal_eval(skill_str)
            if isinstance(skills_list, list):
                all_matched_skills.extend([s.lower().strip() for s in skills_list])
        except:
            pass
            
    # Count the most frequent skills
    skill_counts = Counter(all_matched_skills)
    
    # Return the top N skills
    required_skills = [skill for skill, count in skill_counts.most_common(top_n)]
    return required_skills

# ==========================================================
# 3. Skill Gap Analysis Function
# ==========================================================
def analyze_skill_gap(user_skills, required_skills):
    """
    Compares user skills against required skills to find the gap.
    """
    user_skills_set = set(user_skills)
    required_skills_set = set(required_skills)
    
    # Skills the user already has
    possessed_skills = required_skills_set.intersection(user_skills_set)
    
    # Skills the user is missing
    missing_skills = required_skills_set.difference(user_skills_set)
    
    return list(possessed_skills), list(missing_skills)

# ==========================================================
# 4. Career Readiness Score Function
# ==========================================================
def calculate_readiness_score(possessed_skills, required_skills):
    """
    Calculates a percentage score of how ready the user is for the career.
    """
    if not required_skills:
        return 0
    
    score = (len(possessed_skills) / len(required_skills)) * 100
    return round(score, 2)

# ==========================================================
# 5. Skill Priority & Job Market Evidence Helper
# ==========================================================
def get_skill_evidence(target_career, missing_skills):
    """
    Analyzes job postings for the target career and calculates
    real evidence (frequency and percentage) for each missing skill.
    Categorizes skills into Learn First, Learn Next, Explore Later.
    """
    target_career_str = str(target_career).lower()
    matched_jobs = jobs_df[jobs_df['job_title'].str.lower().str.contains(target_career_str, na=False)]
    
    # If specific title matches fewer than 5 postings, broaden to category domain for robust statistical evidence
    if len(matched_jobs) < 5:
        if any(t in target_career_str for t in ['data', 'software', 'engineer', 'developer', 'python', 'cloud', 'ai', 'tech', 'system', 'web']):
            matched_jobs = jobs_df[jobs_df['category'] == 'INFORMATION-TECHNOLOGY']
        elif any(t in target_career_str for t in ['finance', 'account', 'audit']):
            matched_jobs = jobs_df[jobs_df['category'] == 'FINANCE']
        elif any(t in target_career_str for t in ['hr', 'human', 'talent', 'people']):
            matched_jobs = jobs_df[jobs_df['category'] == 'HR']
        elif any(t in target_career_str for t in ['sale', 'sales']):
            matched_jobs = jobs_df[jobs_df['category'] == 'SALES']
        elif any(t in target_career_str for t in ['manager', 'business', 'project', 'product', 'marketing']):
            matched_jobs = jobs_df[jobs_df['category'] == 'BUSINESS-DEVELOPMENT']
        else:
            matched_jobs = jobs_df
            
    total_jobs = len(matched_jobs)
    if total_jobs == 0:
        total_jobs = len(jobs_df)
        matched_jobs = jobs_df
        
    all_matched_skills = []
    for skill_str in matched_jobs['job_skill_set'].dropna():
        try:
            skills_list = ast.literal_eval(skill_str)
            if isinstance(skills_list, list):
                all_matched_skills.extend([s.lower().strip() for s in skills_list])
        except:
            pass
            
    skill_counts = Counter(all_matched_skills)
    
    skill_evidence = {}
    for skill in missing_skills:
        skill_lower = skill.lower().strip()
        count = skill_counts.get(skill_lower, 0)
        percentage = round((count / total_jobs) * 100, 1) if total_jobs > 0 else 0
        skill_evidence[skill] = {
            "skill": skill,
            "job_count": count,
            "total_jobs": total_jobs,
            "percentage": percentage
        }
        
    # Sort missing skills by frequency descending
    sorted_skills = sorted(missing_skills, key=lambda s: skill_evidence.get(s, {}).get("percentage", 0), reverse=True)
    
    total_count = len(sorted_skills)
    tier_size = max(1, total_count // 3) if total_count > 0 else 0
    
    learn_first = sorted_skills[:tier_size]
    learn_next = sorted_skills[tier_size:tier_size * 2]
    explore_later = sorted_skills[tier_size * 2:]
    
    return {
        "evidence": skill_evidence,
        "learn_first": learn_first,
        "learn_next": learn_next,
        "explore_later": explore_later,
        "total_matched_jobs": total_jobs
    }

# ==========================================================
# 6. Test the Career Analysis Pipeline
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING CAREER ANALYSIS (Phase 3)")
    print("=" * 60)

    # Simulating the student profile from Phase 2
    student_goal = "Data Scientist"
    student_skills = ['data analysis', 'database management', 'machine learning', 'management', 'python', 'sql', 'statistics']

    print(f"\nStudent Goal: {student_goal}")
    print(f"Student's Current Skills: {student_skills}")

    # Get Required Skills
    required_skills_for_goal = get_required_skills(student_goal, top_n=15)
    print(f"\nTop 15 Required Skills for '{student_goal}':\n  {required_skills_for_goal}")

    # Skill Gap Analysis
    possessed, missing = analyze_skill_gap(student_skills, required_skills_for_goal)
    print(f"\nPossessed Core Skills ({len(possessed)}):\n  {possessed}")
    print(f"Missing Core Skills (Skill Gap) ({len(missing)}):\n  {missing}")

    # Career Readiness Score
    readiness = calculate_readiness_score(possessed, required_skills_for_goal)
    print(f"\nCareer Readiness Score: {readiness}%")

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
