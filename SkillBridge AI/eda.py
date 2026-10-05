import pandas as pd

combined = pd.read_csv("dataset/standardized_combined.csv")
udemy = pd.read_csv("dataset/standardized_udemy.csv")

print("Combined Shape:", combined.shape)
print("Udemy Shape:", udemy.shape)
print(combined.info())
print(combined.describe(include="all"))
print(combined.isnull().sum())
print(combined["provider"].value_counts())
print(combined["level"].value_counts())
skills = combined["skills"].dropna().astype(str)

all_skills = ",".join(skills).split(",")

import pandas as pd

skill_count = pd.Series(all_skills).str.strip().value_counts()

print(skill_count.head(20))
print(combined["rating"].describe())
print(combined["duration"].value_counts().head(20))