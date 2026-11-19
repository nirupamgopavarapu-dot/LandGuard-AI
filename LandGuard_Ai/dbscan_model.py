
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "land_records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("========== DBSCAN ANOMALY DETECTION ==========")

# Select features for anomaly detection
features = [
    "ownership_chain_length",
    "boundary_overlap_flag",
    "record_age_gap_years",
    "mutation_count",
    "land_area_acres",
    "encumbrance_flag"
]

X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed!")

# Apply DBSCAN
dbscan = DBSCAN(
    eps=1.5,
    min_samples=10
)

df["anomaly_label"] = dbscan.fit_predict(X_scaled)

# Identify anomalies
anomalies = df[df["anomaly_label"] == -1]
normal_records = df[df["anomaly_label"] != -1]

# Display results
print("\nTotal Records:", len(df))
print("Normal Records:", len(normal_records))
print("Anomalous Records:", len(anomalies))

print("\nCluster Distribution:")
print(df["anomaly_label"].value_counts().sort_index())

print("\nFirst 10 Anomalous Records:")
print(anomalies.head(10))

# Save results
output_file = OUTPUT_DIR / "dbscan_results.csv"
df.to_csv(output_file, index=False)

print("\nResults saved to:", output_file)

# Visualize anomalies
plt.figure(figsize=(9, 6))

plt.scatter(
    normal_records["ownership_chain_length"],
    normal_records["record_age_gap_years"],
    label="Normal Records",
    alpha=0.6
)

plt.scatter(
    anomalies["ownership_chain_length"],
    anomalies["record_age_gap_years"],
    label="Anomalies",
    marker="x"
)

plt.xlabel("Ownership Chain Length")
plt.ylabel("Record Age Gap (Years)")
plt.title("DBSCAN Land Record Anomaly Detection")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "dbscan_anomalies.png")
plt.show()

print("\nDBSCAN COMPLETED SUCCESSFULLY!")
