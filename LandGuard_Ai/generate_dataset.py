
import numpy as np
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Generate 1000 demonstration records
np.random.seed(42)
N = 1000

ownership_chain_length = np.random.poisson(3, N) + 1
boundary_overlap_flag = np.random.binomial(1, 0.12, N)
record_age_gap_years = np.random.exponential(3, N).round(1)
mutation_count = np.random.poisson(2, N)
land_area_acres = np.random.uniform(0.2, 20, N).round(2)
encumbrance_flag = np.random.binomial(1, 0.15, N)

# Add unusual records for anomaly-detection demonstration
anomaly_indices = np.random.choice(N, 60, replace=False)

ownership_chain_length[anomaly_indices] = np.random.randint(
    12, 25, len(anomaly_indices)
)
boundary_overlap_flag[anomaly_indices] = 1
record_age_gap_years[anomaly_indices] = np.random.uniform(
    15, 40, len(anomaly_indices)
).round(1)
mutation_count[anomaly_indices] = np.random.randint(
    12, 25, len(anomaly_indices)
)

# Create demonstration dispute labels
risk_signal = (
    0.9 * (ownership_chain_length >= 7)
    + 1.1 * boundary_overlap_flag
    + 0.7 * (record_age_gap_years >= 8)
    + 0.5 * (mutation_count >= 7)
    + 0.6 * encumbrance_flag
)

probability = 1 / (1 + np.exp(-(risk_signal - 1.8)))
dispute_status = np.random.binomial(1, probability)

# Create dataset
df = pd.DataFrame({
    "land_id": [f"LAND{i:04d}" for i in range(1, N + 1)],
    "ownership_chain_length": ownership_chain_length,
    "boundary_overlap_flag": boundary_overlap_flag,
    "record_age_gap_years": record_age_gap_years,
    "mutation_count": mutation_count,
    "land_area_acres": land_area_acres,
    "encumbrance_flag": encumbrance_flag,
    "dispute_status": dispute_status
})

# Save dataset
output_file = DATA_DIR / "land_records.csv"
df.to_csv(output_file, index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst five records:")
print(df.head())
print("\nDispute label distribution:")
print(df["dispute_status"].value_counts())
print("\nSaved to:", output_file)
print("\nNote: This is synthetic demonstration data.")
