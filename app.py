import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Machine Learning Assignment 2",
    page_icon="🤖",
    layout="wide"
)

# Main Heading
st.title("🤖 Machine Learning Assignment 2")
st.markdown(
    "<h3 style='margin-left:40px;'>Implementation, Evaluation and Deployment of Classification Models</h3>",
    unsafe_allow_html=True
)
st.caption("Implementation, Evaluation and Deployment of Classification Models.")
# Subheading
st.subheader("Implementation, Evaluation and Deployment of Classification Models")

# Normal Text Instruction
# st.write("Upload and test data for Streamlit app.")
# st.title("🤖 Machine Learning Assignment 2")
# st.markdown('<h1 class="main-header">Implementation, Evaluation and Deployment of Classification Models</h1>', unsafe_allow_html=True)
# st.title("🤖 Machine Learning Assignment 2 – Implementation, Evaluation and Deployment of Classification Models ")
# st.caption("Upload test data, select a trained model, and evaluate performance in real-time.")

# st.markdown("---")

# -----------------------------
# Download Sample Test Data
# -----------------------------
st.subheader("📥 Download Sample Test Data")

TEST_DATA_PATH = "data/heart_test.csv"  # must match your saved test CSV

if os.path.exists(TEST_DATA_PATH):
    test_df = pd.read_csv(TEST_DATA_PATH)
    csv_bytes = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Sample Test CSV",
        data=csv_bytes,
        file_name="sample_test_data.csv",
        mime="text/csv"
    )
else:
    st.warning("⚠️ Sample test CSV not found. Please ensure `data/sample_test_data.csv` exists.")

st.markdown("---")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Model Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])
st.sidebar.info("ℹ️ Upload test data, select a trained model, and evaluate performance in real-time.")

# -----------------------------
# Main Evaluation Logic
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # st.subheader("🗂 Uploaded Test Data Preview")
    # st.dataframe(df.head(), use_container_width=True)

    if "target" not in df.columns:
        st.error("❌ Uploaded CSV must contain a `target` column for evaluation.")
    else:
        X = df.drop("target", axis=1)
        y_true = df["target"]

        model_path = f"model/{model_name}.pkl"

        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
        else:
            model = joblib.load(model_path)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            # -----------------------------
            # Metrics
            # -----------------------------
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            st.subheader("📊 Model Evaluation Metrics")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("AUC", f"{auc:.3f}")
            col3.metric("Precision", f"{prec:.3f}")
            col4.metric("Recall", f"{rec:.3f}")
            col5.metric("F1-score", f"{f1:.3f}")
            col6.metric("MCC", f"{mcc:.3f}")

            st.markdown("---")

            # -----------------------------
            # Classification Report
            # -----------------------------
            st.subheader("📋 Classification Report")

            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

            # -----------------------------
            # Confusion Matrix
            # -----------------------------
            st.subheader("📉 Confusion Matrix")

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title(f"Confusion Matrix - {model_name}")

            st.pyplot(fig)