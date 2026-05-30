import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("👥 AI-Powered Telecom Customer Churn Prediction")

st.markdown("""
### About this Project
This application uses an Artificial Neural Network (ANN) to predict whether a telecom customer is likely to leave the company (Churn) or stay.

#### Business Benefits
- Identify high-risk customers
- Reduce customer loss
- Improve retention strategies
- Increase revenue and customer satisfaction
""")

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=24)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

# Encoding maps
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

gender_map = {
    "Female": 0,
    "Male": 1
}

senior_map = {
    "No": 0,
    "Yes": 1
}

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Churn"):

    features = np.array([[
        gender_map[gender],
        senior_map[senior],
        tenure,
        monthly_charges,
        total_charges,
        contract_map[contract],
        internet_map[internet],
        payment_map[payment]
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled, verbose=0)

    probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error(
            f"⚠️ Customer is likely to CHURN\n\n"
            f"Confidence: {probability*100:.2f}%"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY\n\n"
            f"Confidence: {(1-probability)*100:.2f}%"
        )

    st.divider()

    st.subheader("Customer Summary")

    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Monthly Charges:** ₹{monthly_charges}")
    st.write(f"**Total Charges:** ₹{total_charges}")
    st.write(f"**Contract Type:** {contract}")

    st.divider()

    st.subheader("Model Performance")

    # Replace with your actual results
    accuracy = 0.81
    precision = 0.77
    recall = 0.71

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Precision", f"{precision*100:.2f}%")
    col3.metric("Recall", f"{recall*100:.2f}%")

    st.info(
        "Accuracy measures overall correctness, "
        "Precision measures how many predicted churn customers actually churned, "
        "and Recall measures how many actual churn customers were successfully identified."
    )