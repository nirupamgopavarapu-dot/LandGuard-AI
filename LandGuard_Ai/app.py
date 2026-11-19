
import streamlit as st
import pandas as pd
from pathlib import Path

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="LandGuard AI",
    page_icon="🏡",
    layout="wide"
)

# =====================================
# 2. LOAD DATA
# =====================================

BASE_DIR = Path(__file__).resolve().parent

possible_paths = [
    BASE_DIR / "outputs" / "combined_land_record_analysis.csv"
]

DATA_PATH = next(
    (p for p in possible_paths if p.exists()),
    None
)

st.title("🏡 LandGuard AI")
st.subheader(
    "Land Record Anomaly Detection "
    "and Title-Dispute Risk Scoring System"
)

st.markdown("---")

if DATA_PATH is None:
    st.error(
        "Combined dataset not found. "
        "Run combine_results.py first."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

# =====================================
# 3. SIDEBAR
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Land Record Search",
        "Anomaly Records",
        "Dispute Risk Analysis"
    ]
)

# =====================================
# 4. DASHBOARD
# =====================================

if page == "Dashboard":

    st.header("System Overview")

    total = len(df)

    anomalies = int(df["is_anomaly"].sum())

    dispute_risks = int(
        df["is_dispute_risk"].sum()
    )

    both_flags = int(
        (
            df["is_anomaly"]
            & df["is_dispute_risk"]
        ).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total)
    col2.metric("Anomalies", anomalies)
    col3.metric("Dispute Risks", dispute_risks)
    col4.metric("Both Flags", both_flags)

    st.markdown("---")

    st.subheader("Combined Classification")

    category_counts = (
        df["combined_result"]
        .value_counts()
    )

    st.bar_chart(category_counts)

    st.subheader("Dispute Risk Distribution")

    st.bar_chart(
        df["dispute_risk_percentage"]
        .value_counts(bins=10)
        .sort_index()
    )

    st.subheader("Recent Land Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.info(
        "This dashboard supports manual review. "
        "Model flags are not proof of legal disputes."
    )

# =====================================
# 5. LAND RECORD SEARCH
# =====================================

elif page == "Land Record Search":

    st.header("Search Land Records")

    land_id = st.text_input(
        "Enter Land ID",
        placeholder="Example: LAND0001"
    )

    if land_id:

        result = df[
            df["land_id"].str.upper()
            == land_id.strip().upper()
        ]

        if not result.empty:

            st.success("Land record found!")

            st.dataframe(
                result,
                use_container_width=True
            )

            record = result.iloc[0]

            st.metric(
                "Predicted Dispute Risk",
                f"{record['dispute_risk_percentage']}%"
            )

            st.write(
                "Combined Result:",
                record["combined_result"]
            )

        else:
            st.warning("No matching land record found.")

# =====================================
# 6. ANOMALY RECORDS
# =====================================

elif page == "Anomaly Records":

    st.header("Structurally Anomalous Records")

    anomalies_df = df[
        df["is_anomaly"] == True
    ]

    st.write(
        "Total anomalous records:",
        len(anomalies_df)
    )

    st.dataframe(
        anomalies_df,
        use_container_width=True
    )

    st.download_button(
        "Download Anomaly Records",
        data=anomalies_df.to_csv(index=False),
        file_name="land_anomalies.csv",
        mime="text/csv"
    )

# =====================================
# 7. DISPUTE RISK ANALYSIS
# =====================================

elif page == "Dispute Risk Analysis":

    st.header("Dispute Risk Analysis")

    min_risk = st.slider(
        "Minimum predicted risk percentage",
        min_value=0,
        max_value=100,
        value=50
    )

    risk_df = df[
        df["dispute_risk_percentage"]
        >= min_risk
    ]

    st.write(
        "Records matching selected risk:",
        len(risk_df)
    )

    st.dataframe(
        risk_df,
        width="stretch"
    )

    st.download_button(
        "Download Risk Records",
        data=risk_df.to_csv(index=False),
        file_name="land_dispute_risk_records.csv",
        mime="text/csv"
    )

# =====================================
# 8. FOOTER
# =====================================

st.markdown("---")

st.caption(
    "LandGuard AI | SIH26017 | "
    "Educational prototype using synthetic data"
)
