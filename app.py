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

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("📉 Telco Customer Churn – Interactive Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload Telco Customer Churn CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(['customerID'], axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Tabs: EDA and Prediction
    tab1, tab2 = st.tabs(["📊 EDA Dashboard", "🔮 Churn Prediction"])

    # ---------------------------- EDA Dashboard ----------------------------
    with tab1:
        st.subheader("📌 Dataset Overview")
        st.dataframe(df.head(10))

        st.subheader("📈 Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Churn Rate", f"{df['Churn'].mean() * 100:.2f}%")
        with col3:
            st.metric("Avg. Tenure (months)", f"{df['tenure'].mean():.1f}")

        st.subheader("🧮 Churn Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax1)
        st.pyplot(fig1)

        st.subheader("📊 Contract Type vs Churn")
        if 'Contract_Two year' in df.columns:
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Churn', y='Contract_Two year', data=df, ax=ax2)
            st.pyplot(fig2)

    # ---------------------------- Prediction Tab ----------------------------
    with tab2:
        st.subheader("🔍 Predict Churn for a New Customer")

        input_data = {}
        for col in X.columns:
            if "Yes" in col or "No" in col:
                input_data[col] = st.selectbox(col, [0, 1])
            else:
                input_data[col] = st.number_input(col, value=0.0)

        if st.button("Predict Now"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            if prediction == 1:
                st.error(f"🚨 This customer is likely to churn! (Probability: {prob:.2f})")
            else:
                st.success(f"✅ This customer is likely to stay. (Probability: {prob:.2f})")

else:
    st.warning("Please upload a CSV file to begin.")
