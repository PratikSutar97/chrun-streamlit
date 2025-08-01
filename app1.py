import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📉 Telco Customer Churn Predictor App")

# 📂 Upload CSV file
uploaded_file = st.file_uploader("Upload Telco Customer Churn Dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Data Preview")
    st.write(df.head())

    # 🧹 Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(['customerID'], axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 🔁 One-hot encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 📊 Show churn distribution
    st.subheader("📊 Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)

    # 🔍 Split data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 📈 Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 🧪 Model Evaluation
    y_pred = model.predict(X_test)
    st.subheader("✅ Model Evaluation")
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # 💾 Save model for future use (optional)
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "features.pkl")

    # 🔍 User input for prediction
    st.subheader("🔎 Predict Churn for New Customer")

    input_data = {}
    for col in X.columns:
        if col.endswith("Yes") or col.endswith("No"):
            input_data[col] = st.selectbox(col, [0, 1])
        else:
            input_data[col] = st.number_input(col, value=0.0)

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        result = "🚨 Churn" if prediction == 1 else "✅ No Churn"
        st.success(f"Prediction: {result}")
else:
    st.warning("📂 Please upload your dataset to continue.")
