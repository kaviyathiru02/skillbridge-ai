import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import os

print("=" * 60)
print("SKILLBRIDGE AI - LEARNING ROADMAP & EXPLAINABILITY")
print("=" * 60)

# ==========================================================
# 1. Load Data and Models (from Phase 1)
# ==========================================================
data_path = "SkillBridge AI/dataset/engineered_course_dataset.csv"
vectorizer_path = "SkillBridge AI/models/tfidf_vectorizer.pkl"
matrix_path = "SkillBridge AI/models/tfidf_matrix.npz"

if not os.path.exists(data_path):
    data_path = "dataset/engineered_course_dataset.csv"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    matrix_path = "models/tfidf_matrix.npz"

print("Loading dataset and recommendation models...")
df = pd.read_csv(data_path)

with open(vectorizer_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
tfidf_matrix = load_npz(matrix_path)

# ==========================================================
# 2. Roadmap Generation Function
# ==========================================================
def generate_learning_roadmap(missing_skills, target_career):
    """
    Generates a step-by-step learning roadmap to fill the skill gap.
    Provides explainable recommendations for each step.
    """
    roadmap = []
    
    # To avoid recommending the same course multiple times
    recommended_indices = set()
    
    step = 1
    for skill in missing_skills:
        # Create a query specifically for this missing skill
        # We add the target career context to ensure relevant course domain
        query = f"{skill} {target_career}"
        query_vec = tfidf_vectorizer.transform([query])
        
        # Calculate similarity
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Sort indices by highest score
        top_indices = sim_scores.argsort()[::-1]
        
        # Find the best course we haven't recommended yet
        for idx in top_indices:
            if idx not in recommended_indices and sim_scores[idx] > 0:
                course_data = df.iloc[idx]
                
                # Filter out courses with missing names
                c_name = str(course_data['course_name']).strip().lower()
                if pd.isna(course_data['course_name']) or c_name == 'nan' or c_name == '':
                    continue
                
                recommended_indices.add(idx)
                
                # Create the Explainable Recommendation
                explanation = f"Recommended because you are missing '{skill}' for your goal of '{target_career.title()}'."
                explanation_ta = f"'{skill.title()}' திறன் இடைவெளியை சரிசெய்து உங்கள் '{target_career.title()}' இலக்கை அடைய இந்த பாடநெறி பரிந்துரைக்கப்படுகிறது."
                
                roadmap_step = {
                    "Step": step,
                    "Target Skill": skill.title(),
                    "Course Name": course_data['course_name'].title(),
                    "Provider": course_data['provider'].title(),
                    "Level": course_data['level'].title() if pd.notnull(course_data['level']) else "All Levels",
                    "Explanation (AI)": explanation,
                    "Explanation_ta": explanation_ta,
                    "URL": str(course_data['url']) if 'url' in course_data and pd.notnull(course_data['url']) else "",
                    "Rating": str(course_data['rating']) if 'rating' in course_data and pd.notnull(course_data['rating']) else "Not available",
                    "Duration": str(course_data['duration']) if 'duration' in course_data and pd.notnull(course_data['duration']) else "Not available"
                }
                
                roadmap.append(roadmap_step)
                step += 1
                break # Move to the next missing skill
                
        # Limit to top 5-7 steps to not overwhelm the user
        if step > 5:
            break
            
    return roadmap

# ==========================================================
# 3. Test the Roadmap & Explainability
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING PERSONALIZED LEARNING ROADMAP (Phase 4)")
    print("=" * 60)

    # Simulating the gap found in Phase 3
    target_goal = "Data Scientist"
    missing_skills_from_phase3 = ['data visualization', 'predictive modeling', 'project management', 'communication', 'time series analysis']

    print(f"\nTarget Career: {target_goal}")
    print(f"Addressing Skill Gap: {missing_skills_from_phase3}\n")

    roadmap_result = generate_learning_roadmap(missing_skills_from_phase3, target_goal)

    # Display the Roadmap
    print("--- YOUR PERSONALIZED LEARNING ROADMAP ---\n")
    for step in roadmap_result:
        print(f"Step {step['Step']}: Learn {step['Target Skill']}")
        print(f"  Course: {step['Course Name']} ({step['Provider']} - {step['Level']})")
        print(f"  Why: {step['Explanation (AI)']}\n")

    print("=" * 60)
    print("PHASE 4 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
