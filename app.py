import os
import pandas as pd
import streamlit as st
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import datetime
import gdown

# ── Streamlit config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="🔍 Fraud Detection System", layout="wide")

# ── Data loading from Google Drive ─────────────────────────────────────────────
CSV_FILE = "creditcard.csv"  # Saved directly in project root
DRIVE_ID = "13E4KHR2-eq3P-rj08nORBZ6Bhc1W5E3n"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"

if not os.path.exists(CSV_FILE):
    st.info("📥 Downloading dataset from Google Drive...")
    gdown.download(DRIVE_URL, CSV_FILE, quiet=False)

# Load the full dataset (used later for sample or fallback)
df = pd.read_csv(CSV_FILE)

# ── Load model & explainer ─────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_explainer():
    try:
        model     = joblib.load("model_xgb.pkl")
        explainer = joblib.load("shap_explainer.pkl")
        return model, explainer
    except Exception as e:
        st.error(f"Failed to load model/explainer: {e}")
        st.stop()

model, explainer = load_model_and_explainer()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📁 Upload/Test Data", "🤖 Predict Fraud",
     "📊 Explainability", "📈 Business Insights"]
)

# ── Helper to convert DataFrame to CSV ─────────────────────────────────────────
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ── Page 1: Home ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🔐 Enterprise-Grade Credit Card Fraud Detection System")
    st.markdown("""
        This application uses an XGBoost model trained on PCA-transformed data.
        - Real-time predictions
        - SHAP & LIME explainability
        - Business impact & ROI metrics
    """)

# ── Page 2: Upload / Sample Data ─────────────────────────────────────────────────
elif page == "📁 Upload/Test Data":
    st.title("📁 Upload Your CSV or Use Demo Sample")
    uploaded = st.file_uploader(
        "Upload a CSV with V1–V28, Amount, Time columns", type=["csv"]
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded!")
        st.dataframe(df.head())
        st.session_state["df"] = df
    else:
        if st.button("Use Demo Sample"):
            df_sample = df.sample(200, random_state=42)
            st.session_state["df"] = df_sample
            st.success("✅ Loaded demo sample from full dataset (200 rows)")
            st.dataframe(df_sample.head())

# ── Page 3: Predict Fraud ───────────────────────────────────────────────────────
elif page == "🤖 Predict Fraud":
    st.title("🤖 Predict Fraudulent Transactions")
    if "df" in st.session_state:
        df = st.session_state["df"]
        features = [c for c in df.columns if c.startswith("V")] + ["Amount", "Time"]
        X = df[features]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        df_out = df.copy()
        df_out["Fraud_Probability"] = probs
        df_out["Fraud_Prediction"]  = preds

        st.success("✅ Predictions complete!")
        st.dataframe(df_out.head())
        st.download_button(
            "📥 Download Results",
            convert_df(df_out),
            file_name="fraud_predictions.csv"
        )
        st.session_state["df_results"] = df_out
        st.session_state["X"] = X
    else:
        st.warning("⚠️ Please upload or load data first.")

# ── Page 4: Explainability ──────────────────────────────────────────────────────
elif page == "📊 Explainability":
    st.title("📊 Model Explainability: SHAP & LIME")
    if "df_results" in st.session_state and "X" in st.session_state:
        X = st.session_state["X"]
        shap_vals = explainer.shap_values(X)

        st.subheader("🔍 SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, X, show=False)
        st.pyplot(fig)

        st.subheader("🔎 SHAP Force Plot for One Prediction")
        idx = st.slider("Select index", 0, len(X) - 1, 0)
        shap.initjs()
        force_plot_html = shap.force_plot(
            explainer.expected_value, shap_vals[idx], X.iloc[idx], matplotlib=False
        )
        st.components.v1.html(force_plot_html, height=300)

        st.subheader("🌐 LIME Explanation")
        st.markdown(
            "📎 [View LIME explanation (Transaction #15)]"
            "(lime_explanation_transaction_15.html)",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please run predictions first.")

# ── Page 5: Business Insights ──────────────────────────────────────────────────
elif page == "📈 Business Insights":
    st.title("📈 Business Impact Analysis")
    if "df_results" in st.session_state:
        df_out = st.session_state["df_results"]
        fraud_ct = int(df_out["Fraud_Prediction"].sum())
        total   = len(df_out)
        rate     = round(fraud_ct / total * 100, 2)
        savings  = fraud_ct * 500  # $500 per fraud

        st.metric("🚨 Fraudulent Transactions", fraud_ct)
        st.metric("📊 Detection Rate", f"{rate}%")
        st.metric("💰 Estimated Savings", f"${savings:,}")

        st.markdown("### 📌 Strategic Recommendations")
        st.markdown("""
            - Investigate high-risk transactions (>90% probability).
            - Integrate model into real-time payment flow.
            - Retrain quarterly with fresh data.
            - Combine SHAP/LIME with manual review for compliance.
        """)
    else:
        st.warning("⚠️ Please predict fraud first.")

# ── Footer & License ──────────────────────────────────────────────────────────
st.markdown("""
---
© 2025 Sweety Seelam · Licensed under the MIT License  
Fraud Detection System – Streamlit App  
GitHub: https://github.com/SweetySeelam2/Fraud_Detection_ML
""")
st.sidebar.caption(f"🕒 Last updated: {datetime.datetime.now():%Y-%m-%d}")