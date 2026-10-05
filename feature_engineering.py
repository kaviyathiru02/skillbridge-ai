import pandas as pd
import numpy as np
import re
import os

print("=" * 60)
print("SKILLBRIDGE AI - FEATURE ENGINEERING")
print("=" * 60)

# ==========================================================
# Load Final Course Dataset
# ==========================================================
input_path = "SkillBridge AI/dataset/final_course_dataset.csv"
if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    # fallback if running inside the inner folder
    input_path = "dataset/final_course_dataset.csv"

df = pd.read_csv(input_path)

print(f"\nDataset Loaded Successfully!")
print(f"Shape: {df.shape}")

# ==========================================================
# Final Preprocessing (Handle Missing Values in Text Columns)
# ==========================================================

# Text columns to combine for feature engineering
text_columns = ['course_name', 'description', 'skills', 'level', 'provider', 'subject']

print("\nFilling missing values in text columns with empty strings...")
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna('')

# Optional: basic text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text columns...")
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

# ==========================================================
# Feature Engineering (Create course_features)
# ==========================================================

print("\nCreating 'course_features' column by combining textual metadata...")

# We combine course_name, description, skills, level, provider, and subject.
# Weighting can be implicitly done by repeating important fields like course_name or skills.
df['course_features'] = (
    df['course_name'] + " " + 
    df['course_name'] + " " + # Double weighting course name
    df['skills'] + " " +
    df['skills'] + " " +      # Double weighting skills
    df['description'] + " " +
    df['level'] + " " +
    df['provider'] + " " +
    df['subject']
)

# Apply final clean to the combined features
df['course_features'] = df['course_features'].apply(clean_text)

print("\nSample of 'course_features':")
print(df['course_features'].head())

# ==========================================================
# Save Engineered Dataset
# ==========================================================
output_path = "SkillBridge AI/dataset/engineered_course_dataset.csv"
if not os.path.exists("SkillBridge AI/dataset"):
    output_path = "dataset/engineered_course_dataset.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"Saved Location: {output_path}")
print(f"Total Courses: {df.shape[0]}")
print(f"Total Features (including new 'course_features'): {df.shape[1]}")
