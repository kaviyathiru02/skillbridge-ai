import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/standardized_combined.csv")

print("Dataset Loaded Successfully!")

# -------------------------------
# Graph 1 : Provider Distribution
# -------------------------------

provider_counts = df["provider"].value_counts()

plt.figure(figsize=(6,5))
provider_counts.plot(kind="bar")
plt.title("Course Provider Distribution")
plt.xlabel("Provider")
plt.ylabel("Number of Courses")
plt.tight_layout()
plt.savefig("provider_distribution.png")
plt.show()

# -------------------------------
# Graph 2 : Course Level Distribution
# -------------------------------

level_counts = df["level"].value_counts()

plt.figure(figsize=(7,5))
level_counts.plot(kind="bar")
plt.title("Course Level Distribution")
plt.xlabel("Level")
plt.ylabel("Number of Courses")
plt.tight_layout()
plt.savefig("course_level_distribution.png")
plt.show()

# -------------------------------
# Graph 3 : Top 10 Skills
# -------------------------------

top_skills = (
    df["skills"]
    .dropna()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,6))
top_skills.plot(kind="barh")
plt.title("Top 10 Skills")
plt.xlabel("Number of Courses")
plt.tight_layout()
plt.savefig("top_skills.png")
plt.show()

# -------------------------------
# Graph 4 : Course Duration
# -------------------------------

plt.figure(figsize=(8,5))
df["duration"].dropna().plot(kind="hist", bins=20)
plt.title("Course Duration Distribution")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("course_duration.png")
plt.show()

# -------------------------------
# Graph 5 : Rating Distribution
# -------------------------------

top_rating = df["rating"].value_counts().head(10)

plt.figure(figsize=(8,5))
top_rating.plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Courses")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

# -------------------------------
# Graph 6 : Missing Values
# -------------------------------

missing = df.isnull().sum()

plt.figure(figsize=(10,6))
missing.plot(kind="bar")
plt.title("Missing Values in Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("missing_values.png")
plt.show()

print("\nEDA Graphs Generated Successfully!")