
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

# Support both possible dataset locations
possible_paths = [
    BASE_DIR / "data" / "land_records.csv",
    BASE_DIR / "data" / "data" / "land_records.csv"
]

DATA_PATH = next(
    (path for path in possible_paths if path.exists()),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "land_records.csv not found. Check your data folder."
    )

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\n========== RANDOM FOREST CLASSIFICATION ==========")

print("\nDataset loaded successfully!")
print("Total records:", len(df))

# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "ownership_chain_length",
    "boundary_overlap_flag",
    "record_age_gap_years",
    "mutation_count",
    "land_area_acres",
    "encumbrance_flag"
]

X = df[features]
y = df["dispute_status"]

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 5. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 6. CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1"
    ],
    n_jobs=-1
)

print("\n========== CROSS VALIDATION RESULTS ==========")

for metric in ["accuracy", "precision", "recall", "f1"]:
    scores = cv_results["test_" + metric]

    print(
        f"{metric.capitalize()}: "
        f"{scores.mean():.4f} "
        f"(+/- {scores.std():.4f})"
    )

# ==========================================
# 7. TRAIN FINAL MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")

# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# ==========================================
# 9. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_probability)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Dispute", "Dispute"],
    yticklabels=["No Dispute", "Dispute"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "feature_importance.png")
plt.show()

# ==========================================
# 12. SAVE PREDICTIONS
# ==========================================

results = df.loc[X_test.index, [
    "land_id",
    "dispute_status"
]].copy()

results["predicted_dispute"] = y_pred
results["dispute_risk_probability"] = y_probability
results["dispute_risk_percentage"] = (
    y_probability * 100
).round(2)

results.to_csv(
    OUTPUT_DIR / "random_forest_predictions.csv",
    index=False
)

# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_DIR / "random_forest_model.pkl"
)

# ==========================================
# 14. SAVE METRICS
# ==========================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "random_forest_metrics.csv",
    index=False
)

print("\n====================================")
print("RANDOM FOREST COMPLETED SUCCESSFULLY!")
print("Model saved in models folder.")
print("Results saved in outputs folder.")
print("====================================")
