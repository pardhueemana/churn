import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load files
model = load_model("customer_churn_model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

# Basic Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
dependents = st.selectbox("Dependents", [0, 1])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", [0, 1])

paperless = st.selectbox("Paperless Billing", [0, 1])

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer",
        "Credit card",
        "Electronic check",
        "Mailed check"
    ]
)

if st.button("Predict"):

    data = dict.fromkeys(columns, 0)

    # Numerical
    data["SeniorCitizen"] = senior
    data["Partner"] = partner
    data["Dependents"] = dependents
    data["tenure"] = tenure
    data["PhoneService"] = phone_service
    data["PaperlessBilling"] = paperless
    data["MonthlyCharges"] = monthly_charges
    data["TotalCharges"] = total_charges

    # Gender
    if gender == "Male":
        data["gender_Male"] = 1

    # Internet Service
    if internet == "Fiber optic":
        data["InternetService_Fiber optic"] = 1
    elif internet == "No":
        data["InternetService_No"] = 1

    # Contract
    if contract == "One year":
        data["Contract_One year"] = 1
    elif contract == "Two year":
        data["Contract_Two year"] = 1

    # Payment Method
    if payment == "Credit card":
        data["PaymentMethod_Credit card (automatic)"] = 1

    elif payment == "Electronic check":
        data["PaymentMethod_Electronic check"] = 1

    elif payment == "Mailed check":
        data["PaymentMethod_Mailed check"] = 1

    input_df = pd.DataFrame([data])

    # Scale only training-scaled columns
    input_df[['tenure',
              'MonthlyCharges',
              'TotalCharges']] = scaler.transform(
        input_df[['tenure',
                  'MonthlyCharges',
                  'TotalCharges']]
    )

    prediction = model.predict(input_df)

    probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    if probability > 0.5:
        st.error(
            f"⚠️ Customer likely to churn ({probability:.2%})"
        )
    else:
        st.success(
            f"✅ Customer likely to stay ({1-probability:.2%})"
        )