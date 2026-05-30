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

# ================= LOAD MODEL SAFELY =================
@st.cache_resource
def load_assets():
    model = load_model("churn_model.h5")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_assets()

# ================= CUSTOM UI =================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    color: #00d4ff;
}
.block {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}
.stButton>button {
    background-color: #00d4ff;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("📊 AI-Powered Customer Churn Intelligence Platform")
st.markdown("### Business-grade churn prediction using Artificial Neural Networks")

st.markdown("---")

# ================= INPUT SECTION =================
st.subheader("📥 Customer Profile Input")

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
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

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

    # one-hot encoding
    df = pd.get_dummies(df)

    # align columns safely
    df = df.reindex(columns=columns, fill_value=0)

    # scaling
    return scaler.transform(df)

# ================= PREDICTION =================
if st.button("🔍 Predict Churn Risk"):

    input_data = preprocess()
    prob = float(model.predict(input_data)[0][0])

    st.markdown("## 📊 Prediction Result")

    # ================= RISK SCORE BAR =================
    st.progress(min(prob, 1.0))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        if prob >= 0.7:
            st.error("🔴 HIGH RISK CUSTOMER")
            risk_level = "High Risk"
        elif prob >= 0.4:
            st.warning("🟠 MEDIUM RISK CUSTOMER")
            risk_level = "Medium Risk"
        else:
            st.success("🟢 LOW RISK CUSTOMER")
            risk_level = "Low Risk"

    st.markdown("---")

    # ================= BUSINESS INSIGHT =================
    st.subheader("📌 Business Recommendation Engine")

    if prob >= 0.7:
        st.markdown("""
        <div class="block">
        ⚠️ <b>Immediate Action Required</b><br><br>
        • Offer personalized discount or retention plan<br>
        • Assign dedicated account manager<br>
        • Provide long-term contract upgrade incentives<br>
        • Priority support activation
        </div>
        """, unsafe_allow_html=True)

    elif prob >= 0.4:
        st.markdown("""
        <div class="block">
        🟡 <b>Moderate Risk Detected</b><br><br>
        • Send promotional offers<br>
        • Improve engagement via email/SMS campaigns<br>
        • Recommend bundle upgrades
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="block">
        🟢 <b>Low Risk Customer</b><br><br>
        • Maintain service quality<br>
        • Upsell premium plans<br>
        • Encourage referrals
        </div>
        """, unsafe_allow_html=True)

    # ================= EXTRA BUSINESS VALUE =================
    st.markdown("---")
    st.subheader("📊 Business Insight Summary")

    st.write(f"""
    - Risk Category: **{risk_level}**
    - Model Confidence: **{prob:.2%} churn probability**
    - Recommendation: Automated retention strategy suggested based on risk level
    """)
