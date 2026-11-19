# LandGuard AI – Land Record Anomaly Detection and Title-Dispute Risk Scoring System

## Project Overview

LandGuard AI is a student project designed to identify unusual patterns in digitized land records and estimate potential title-dispute risk using machine learning.

The system combines DBSCAN-based anomaly detection with a Random Forest classification model and provides an interactive dashboard using Streamlit.

## Problem Statement

Digitized land records may contain inconsistencies that require manual verification. Reviewing every record manually can be time-consuming.

LandGuard AI demonstrates how machine learning can help prioritize records for further human review.

## Key Features

- Land record dataset analysis and visualization.
- DBSCAN-based structural anomaly detection.
- Random Forest-based dispute-risk classification.
- Combined anomaly and risk analysis.
- Interactive Streamlit dashboard.
- Land record search and analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Dataset

This project uses a **synthetic dataset containing 1,000 land records and 8 columns**.

The dataset is artificially generated for academic demonstration and does not represent actual government land records or verified historical dispute cases.

## Model Performance

Random Forest test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 74.50% |
| Precision | 43.59% |
| Recall | 36.96% |
| F1-score | 40.00% |
| ROC-AUC | 0.6214 |

DBSCAN identified 33 anomaly records out of 1,000 records in the implemented experiment.

These results are based on synthetic data and should not be interpreted as real-world predictive performance.

## Dashboard Modules

1. Dashboard
2. Land Record Search
3. Anomaly Records
4. Dispute Risk Analysis

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nirupamgopavarapu-dot/LandGuard-AI.git
```

### 2. Open the project folder

```bash
cd LandGuard-AI
```

### 3. Install dependencies

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

Navigate to the folder containing `app.py` if necessary, then run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## Limitations

- The dataset is synthetic, not real government data.
- Dispute labels are artificially generated, not verified historical outcomes.
- Model performance has not been validated on real land records.
- The application is an academic prototype, not a legal decision-making system.
- All predictions require human verification.

## Team

- G. Nirupam Reddy
- Laxman

**Institution:** Lovely Professional University  
**Department:** Computer Science and Engineering  
**Academic Year:** 2026–2027

## Disclaimer

LandGuard AI is developed for educational and demonstration purposes only. It does not establish land ownership, detect legally proven fraud, or replace official land-record verification.
