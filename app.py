import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

st.title("ML Classification Model Demo")

# -----------------------------
# Download Sample Test Data
# -----------------------------
st.subheader("Download Sample Test Data")

TEST_DATA_PATH = "data/test/heart_test.csv"   # adjust path if needed

st.title("ML Classification Model Demo")
if os.path.exists(TEST_DATA_PATH):
    test_df = pd.read_csv(TEST_DATA_PATH)
    csv_bytes = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Sample Test CSV",
        data=csv_bytes,
        file_name="heart_test.csv",
        mime="text/csv"
    )
else:
    st.warning("Test CSV not found in repository. Please check data path.")

st.markdown("---")

# -----------------------------
# Model Selection
# -----------------------------
model_name = st.selectbox("Choose Model", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = joblib.load(f"model/{model_name}.pkl")
    y_pred = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
