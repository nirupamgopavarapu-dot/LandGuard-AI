
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "land_records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 1. Dataset shape
print("\nDataset Shape:")
print(df.shape)

# 2. First five records
print("\nFirst Five Records:")
print(df.head())

# 3. Dataset information
print("\nDataset Information:")
df.info()

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# 7. Dispute distribution
print("\nDispute Distribution:")
print(df["dispute_status"].value_counts())

# 8. Dispute distribution graph
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="dispute_status")
plt.title("Land Record Dispute Distribution")
plt.xlabel("Dispute Status (0 = No Dispute, 1 = Dispute)")
plt.ylabel("Number of Records")
plt.savefig(OUTPUT_DIR / "dispute_distribution.png")
plt.show()

# 9. Correlation heatmap
plt.figure(figsize=(10, 7))
correlation = df.select_dtypes(include="number").corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Land Record Feature Correlation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.show()

# 10. Feature distributions
features = [
    "ownership_chain_length",
    "record_age_gap_years",
    "mutation_count",
    "land_area_acres"
]

for feature in features:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feature}_distribution.png")
    plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in the outputs folder.")
