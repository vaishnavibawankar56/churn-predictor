import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to estimate the chance they will leave.")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
    senior = st.selectbox("Senior citizen", [0, 1])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
with col2:
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    security = st.selectbox("Online security", ["Yes", "No", "No internet service"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly, "SeniorCitizen": senior,
        "Contract": contract, "InternetService": internet,
        "PaymentMethod": payment, "TechSupport": tech,
        "OnlineSecurity": security, "PaperlessBilling": paperless,
    }])
    prob = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{prob:.1%}")
    if prob > 0.5:
        st.error("⚠️ High risk: consider a retention offer.")
    else:
        st.success("✅ Low risk of churning.")