import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.title("ML Classification Model Demo")

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
