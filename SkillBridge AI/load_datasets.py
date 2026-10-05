import pandas as pd
import json

print("=" * 60)
print("SKILLBRIDGE AI - DATASET LOADING")
print("=" * 60)

# Load Udemy Dataset
udemy = pd.read_csv("dataset/udemy_courses.csv")

# Load Coursera Dataset
with open("dataset/processed_coursera_data.json", "r", encoding="utf-8") as file:
    coursera = pd.DataFrame(json.load(file))

# Load Combined Dataset
with open("dataset/combined_dataset.json", "r", encoding="utf-8") as file:
    combined = pd.DataFrame(json.load(file))

print("\nDatasets Loaded Successfully!\n")

print("Udemy Shape:", udemy.shape)
print("Coursera Shape:", coursera.shape)
print("Combined Shape:", combined.shape)
print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print("\nUdemy Columns:")
print(udemy.columns.tolist())

print("\nCoursera Columns:")
print(coursera.columns.tolist())

print("\nCombined Columns:")
print(combined.columns.tolist())
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nUdemy Missing Values:")
print(udemy.isnull().sum())

print("\nCoursera Missing Values:")
print(coursera.isnull().sum())

print("\nCombined Missing Values:")
print(combined.isnull().sum())
print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

print("Udemy:", udemy.duplicated().sum())
print("Coursera:", coursera.duplicated(subset=["course_name"]).sum())
print("Combined:", combined.duplicated(subset=["course_name"]).sum())