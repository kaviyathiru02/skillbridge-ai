import pandas as pd
import json

print("=" * 60)
print("SKILLBRIDGE AI - DATA CLEANING")
print("=" * 60)

# Load Udemy Dataset
udemy = pd.read_csv("dataset/udemy_courses.csv")

# Load Combined Dataset
with open("dataset/combined_dataset.json", "r", encoding="utf-8") as file:
    combined = pd.DataFrame(json.load(file))

print("\nDatasets Loaded Successfully!")

print("\nOriginal Shapes")
print("Udemy:", udemy.shape)
print("Combined:", combined.shape)
print("\nRemoving Duplicate Records...")

udemy = udemy.drop_duplicates()

combined = combined.drop_duplicates(subset=["course_name"])

print("Duplicates Removed!")

print("\nNew Shapes")
print("Udemy:", udemy.shape)
print("Combined:", combined.shape)
print("\nMissing Values")

print("\nUdemy")
print(udemy.isnull().sum())

print("\nCombined")
print(combined.isnull().sum())
combined["description"] = combined["description"].fillna("No Description")

combined["rating"] = combined["rating"].fillna("No Rating")

combined["level"] = combined["level"].fillna("Not Specified")
def convert_list_to_text(column):

    return column.apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

combined["skills"] = convert_list_to_text(combined["skills"])

combined["organization"] = convert_list_to_text(combined["organization"])
combined.to_csv("dataset/cleaned_combined_dataset.csv", index=False)

udemy.to_csv("dataset/cleaned_udemy_dataset.csv", index=False)

print("\nCleaned datasets saved successfully!")
