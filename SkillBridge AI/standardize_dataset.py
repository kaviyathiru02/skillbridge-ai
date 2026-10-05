import pandas as pd

print("=" * 60)
print("SKILLBRIDGE AI - DATASET STANDARDIZATION")
print("=" * 60)

# ==========================================================
# Load Cleaned Datasets
# ==========================================================

udemy = pd.read_csv("dataset/cleaned_udemy_dataset.csv")
combined = pd.read_csv("dataset/cleaned_combined_dataset.csv")

print("\nDatasets Loaded Successfully!\n")

print("Udemy Shape:", udemy.shape)
print("Combined Shape:", combined.shape)

# ==========================================================
# Standardize Udemy Dataset
# ==========================================================

udemy.rename(columns={
    "course_title": "course_name",
    "content_duration": "duration",
    "num_reviews": "reviews",
    "subject": "skills"
}, inplace=True)

# Add missing columns
udemy["description"] = "Description not available"
udemy["provider"] = "Udemy"
udemy["rating"] = None
udemy["organization"] = None
udemy["instructor"] = None
udemy["subject"] = udemy["skills"]

# ==========================================================
# Standardize Combined Dataset
# ==========================================================

# Rename only Duration
combined.rename(columns={
    "Duration": "duration"
}, inplace=True)

# If reviews column is missing, use nu_reviews
if "reviews" not in combined.columns and "nu_reviews" in combined.columns:
    combined.rename(columns={"nu_reviews": "reviews"}, inplace=True)

# Remove nu_reviews if both exist
if "reviews" in combined.columns and "nu_reviews" in combined.columns:
    combined.drop(columns=["nu_reviews"], inplace=True)

# Remove duplicate column names (important)
combined = combined.loc[:, ~combined.columns.duplicated()]

# Add missing columns
combined["price"] = None
combined["num_subscribers"] = None

# ==========================================================
# Clean Level Column
# ==========================================================

def clean_level(level):

    level = str(level).strip()

    if "Beginner" in level and "Intermediate" in level:
        return "Mixed"

    elif "Intermediate" in level and "Advanced" in level:
        return "Mixed"

    elif "Beginner" in level and "Advanced" in level:
        return "Mixed"

    elif "Beginner" in level:
        return "Beginner"

    elif "Intermediate" in level:
        return "Intermediate"

    elif "Advanced" in level:
        return "Advanced"

    elif level.lower() == "mixed":
        return "Mixed"

    else:
        return "Unknown"

combined["level"] = combined["level"].apply(clean_level)

# ==========================================================
# Select Common Columns
# ==========================================================

columns = [
    "course_name",
    "description",
    "skills",
    "level",
    "duration",
    "reviews",
    "rating",
    "provider",
    "url",
    "organization",
    "instructor",
    "subject",
    "price",
    "num_subscribers"
]

udemy = udemy[columns]
combined = combined[columns]

# ==========================================================
# Display Columns
# ==========================================================

print("\nUdemy Columns")
print(udemy.columns.tolist())

print("\nCombined Columns")
print(combined.columns.tolist())

# ==========================================================
# Save Standardized Datasets
# ==========================================================

udemy.to_csv(
    "dataset/standardized_udemy.csv",
    index=False
)

combined.to_csv(
    "dataset/standardized_combined.csv",
    index=False
)

print("\nStandardized datasets saved successfully!")