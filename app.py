import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ================= LOAD MODEL =================
model = load_model("churn_model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ================= CUSTOM UI =================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1 {
        text-align: center;
        color: #00d4ff;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("📊 AI-Powered Customer Churn Prediction Platform")
st.markdown("### Predict whether a customer will leave next month using ANN")

st.markdown("---")

# ================= INPUT FORM =================
st.subheader("📥 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col3:
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

st.markdown("---")

# ================= PREPROCESS =================
def preprocess():
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'InternetService': internet,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    df = pd.DataFrame([input_dict])

    # One-hot encoding (match training style)
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled

# ================= PREDICTION =================
if st.button("🔍 Predict Churn Risk"):

    input_data = preprocess()
    prob = model.predict(input_data)[0][0]

    st.markdown("## 📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        if prob > 0.7:
            st.error("🔴 HIGH RISK CUSTOMER")
        elif prob > 0.4:
            st.warning("🟠 MEDIUM RISK CUSTOMER")
        else:
            st.success("🟢 LOW RISK CUSTOMER")

    st.markdown("---")

    # ================= BUSINESS INSIGHT =================
    st.subheader("📌 Business Recommendation")

    if prob > 0.7:
        st.write("""
        ⚠️ Immediate Action Required:
        - Offer discount or retention plan  
        - Assign account manager  
        - Provide long-term contract benefits  
        """)
    elif prob > 0.4:
        st.write("""
        🟡 Moderate Risk:
        - Send promotional offers  
        - Improve engagement via email/SMS  
        """)
    else:
        st.write("""
        🟢 Safe Customer:
        - Maintain service quality  
        - Upsell premium plans  
        """)