
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent

# Locate dataset
data_paths = [
    BASE_DIR / "data" / "land_records.csv",
    BASE_DIR / "data" / "data" / "land_records.csv"
]

DATA_PATH = next(
    (p for p in data_paths if p.exists()), None
)

if DATA_PATH is None:
    raise FileNotFoundError("Dataset not found!")

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
DBSCAN_PATH = BASE_DIR / "outputs" / "dbscan_results.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load files
df = pd.read_csv(DATA_PATH)
dbscan_df = pd.read_csv(DBSCAN_PATH)
model = joblib.load(MODEL_PATH)

print("\n========== COMBINED ANALYSIS ==========")

# Select model features
features = [
    "ownership_chain_length",
    "boundary_overlap_flag",
    "record_age_gap_years",
    "mutation_count",
    "land_area_acres",
    "encumbrance_flag"
]

# Apply Random Forest to all records
df["predicted_dispute"] = model.predict(df[features])

df["dispute_risk_probability"] = (
    model.predict_proba(df[features])[:, 1]
)

df["dispute_risk_percentage"] = (
    df["dispute_risk_probability"] * 100
).round(2)

# Add DBSCAN anomaly labels
df["anomaly_label"] = dbscan_df["anomaly_label"].values

df["is_anomaly"] = (
    df["anomaly_label"] == -1
)

df["is_dispute_risk"] = (
    df["predicted_dispute"] == 1
)

# Create combined categories
def classify_record(row):

    if row["is_anomaly"] and row["is_dispute_risk"]:
        return "Both Flags"

    elif row["is_anomaly"]:
        return "Structural Anomaly Only"

    elif row["is_dispute_risk"]:
        return "Dispute Risk Only"

    else:
        return "Neither Flag"

df["combined_result"] = df.apply(
    classify_record, axis=1
)

# Display summary
print("\nCombined Results Distribution:")

print(
    df["combined_result"].value_counts()
)

print("\nFirst 10 Records:")

print(
    df[[
        "land_id",
        "anomaly_label",
        "predicted_dispute",
        "dispute_risk_percentage",
        "combined_result"
    ]].head(10)
)

# Save combined results
output_file = (
    OUTPUT_DIR / "combined_land_record_analysis.csv"
)

df.to_csv(output_file, index=False)

print("\nCombined analysis saved successfully!")
print("File:", output_file)

print("\n========== COMPLETED ==========")
