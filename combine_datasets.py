import pandas as pd

print("=" * 60)
print("SKILLBRIDGE AI - COMBINING DATASETS")
print("=" * 60)

# ==========================================================
# Load Standardized Datasets
# ==========================================================

combined = pd.read_csv(
    "SkillBridge AI/dataset/standardized_combined.csv"
)

udemy = pd.read_csv(
    "SkillBridge AI/dataset/standardized_udemy.csv"
)

print("\n✅ Datasets Loaded Successfully!")

print("\nCombined Dataset Shape:", combined.shape)
print("Udemy Dataset Shape:", udemy.shape)

# ==========================================================
# Check Dataset Columns
# ==========================================================

print("\nCombined Dataset Columns:")
print(combined.columns.tolist())

print("\nUdemy Dataset Columns:")
print(udemy.columns.tolist())

# ==========================================================
# Combine Both Datasets
# ==========================================================

final_dataset = pd.concat(
    [combined, udemy],
    ignore_index=True
)

print("\n📊 Shape After Combining:", final_dataset.shape)

# ==========================================================
# Remove Duplicate Courses
# ==========================================================

before_duplicates = final_dataset.shape[0]

# Remove duplicates based on course name
final_dataset.drop_duplicates(
    subset=["course_name"],
    keep="first",
    inplace=True
)

after_duplicates = final_dataset.shape[0]

print("\nDuplicate Courses Removed:",
      before_duplicates - after_duplicates)

# ==========================================================
# Reset Index
# ==========================================================

final_dataset.reset_index(
    drop=True,
    inplace=True
)

# ==========================================================
# Display Final Dataset Information
# ==========================================================

print("\n" + "=" * 60)
print("FINAL DATASET INFORMATION")
print("=" * 60)

print("\nFinal Dataset Shape:", final_dataset.shape)

print("\nFinal Dataset Columns:")
print(final_dataset.columns.tolist())

print("\nMissing Values:")
print(final_dataset.isnull().sum())

print("\nProvider Distribution:")
print(final_dataset["provider"].value_counts())

# ==========================================================
# Save Final Dataset
# ==========================================================

output_path = (
    "SkillBridge AI/dataset/final_course_dataset.csv"
)

final_dataset.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("✅ FINAL DATASET CREATED SUCCESSFULLY!")
print("=" * 60)

print("\n📂 Saved Location:")
print(output_path)

print("\n📊 Total Courses:", final_dataset.shape[0])
print("📊 Total Features:", final_dataset.shape[1])