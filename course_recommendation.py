import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
import os

if __name__ == "__main__":
    print("=" * 60)
    print("SKILLBRIDGE AI - TF-IDF & COURSE RECOMMENDATION")
    print("=" * 60)

    # ==========================================================
    # 1. Load Engineered Dataset
    # ==========================================================
    data_path = "SkillBridge AI/dataset/engineered_course_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "dataset/engineered_course_dataset.csv"

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # ==========================================================
    # 2. TF-IDF Vectorization
    # ==========================================================
    print("\nPerforming TF-IDF Vectorization on 'course_features'...")

    # Initialize TF-IDF Vectorizer
    # Using stop_words='english' and max_features to limit memory usage and remove noise
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)

    # Fit and transform the course features
    tfidf_matrix = tfidf.fit_transform(df['course_features'].fillna(''))
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

    # Save the Vectorizer and TF-IDF Matrix for future use (e.g., Streamlit App)
    models_dir = "SkillBridge AI/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(models_dir, "tfidf_matrix.npz")

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)
    save_npz(matrix_path, tfidf_matrix)

    print(f"TF-IDF Vectorizer saved to {vectorizer_path}")
    print(f"TF-IDF Matrix saved to {matrix_path}")

    print("\n" + "=" * 60)
    print("TESTING PERSONALIZED RECOMMENDATION")
    print("=" * 60)

    # Simulating a user's extracted skills and career goal
    sample_user_profile = "data scientist machine learning python statistics deep learning ai predictive modeling"

    print(f"\nUser Profile/Query: '{sample_user_profile}'\n")
    print("Top 5 Recommended Courses:")

    recommended_df = recommend_courses(
        user_query=sample_user_profile,
        df=df,
        tfidf_model=tfidf,
        tfidf_mat=tfidf_matrix,
        top_n=5
    )

    print(recommended_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
def recommend_courses(user_query, df, tfidf_model, tfidf_mat, top_n=5):
    """
    Recommend courses based on a user's profile/query using Cosine Similarity.
    """
    # Vectorize the user query
    query_vec = tfidf_model.transform([user_query])
    
    # Calculate cosine similarity between the user query and all courses
    sim_scores = cosine_similarity(query_vec, tfidf_mat).flatten()
    
    # Sort indices by highest score
    sorted_indices = sim_scores.argsort()[::-1]
    
    valid_indices = []
    for idx in sorted_indices:
        course_name = str(df.iloc[idx]['course_name']).strip().lower()
        if pd.notna(df.iloc[idx]['course_name']) and course_name != 'nan' and course_name != '':
            valid_indices.append(idx)
        if len(valid_indices) >= top_n:
            break
            
    # Prepare the recommended courses dataframe
    recommendations = df.iloc[valid_indices].copy()
    recommendations['similarity_score'] = sim_scores[valid_indices]
    
    # Return core columns plus any existing course metadata for rich cards
    output_cols = ['course_name', 'provider', 'level', 'similarity_score']
    for extra_col in ['rating', 'duration', 'url', 'skills', 'description']:
        if extra_col in recommendations.columns and extra_col not in output_cols:
            output_cols.append(extra_col)
            
    return recommendations[output_cols]

# ==========================================================
# 4. Test the Recommendation System
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING PERSONALIZED RECOMMENDATION")
    print("=" * 60)

    # Simulating a user's extracted skills and career goal
    sample_user_profile = "data scientist machine learning python statistics deep learning ai predictive modeling"

    print(f"\nUser Profile/Query: '{sample_user_profile}'\n")
    print("Top 5 Recommended Courses:")

    recommended_df = recommend_courses(
        user_query=sample_user_profile,
        df=df,
        tfidf_model=tfidf,
        tfidf_mat=tfidf_matrix,
        top_n=5
    )

    print(recommended_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
