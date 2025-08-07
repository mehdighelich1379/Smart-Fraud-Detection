import streamlit as st
import joblib
import numpy as np
import random

# Load trained model (trained on 7 million real transactions)
pipeline = joblib.load("src/models/fraud_catboost_pipeline.pkl")

# Set Streamlit page config
st.set_page_config(page_title="💳 Smart Fraud Detector", layout="centered")

# App Title
st.title("💳 Smart Fraud Detector")

# Description block for clarity (very important for interview/demo)
with st.expander("ℹ️ About this App"):
    st.write("""
        This application demonstrates a real-time fraud detection model trained on over **7 million real-world financial transactions**.
        The machine learning pipeline includes data preprocessing, feature engineering, and a **CatBoost Classifier** to detect suspicious money transfers.
        
        To help users test the app quickly, a **synthetic data generator** is included. This creates random transactions that *mimic* either legitimate or fraudulent behavior for testing purposes only.

        ⚠️ **Note:** The model used for prediction is real. The randomly generated inputs are only meant to simulate transactions, and are **not hardcoded or rule-based predictions**.
    """)

# Function to generate random test input
def generate_sample_transaction():
    amount = round(random.uniform(100.0, 100000.0), 2)
    sender_before = round(random.uniform(amount, amount + 1e5), 2)
    
    # Fraudulent pattern: sudden drop
    if random.random() < 0.5:
        sender_after = round(sender_before - amount - random.uniform(1, 2000), 2)
        receiver_before = round(random.uniform(0, 1000), 2)
        receiver_after = receiver_before  # no change
    else:
        sender_after = round(sender_before - amount, 2)
        receiver_before = round(random.uniform(100, 5000), 2)
        receiver_after = round(receiver_before + amount, 2)
        
    return {
        "type": "TRANSFER",
        "step": random.choice(["Midnight", "Morning", "Afternoon", "Evening"]),
        "amount": amount,
        "oldbalanceOrg": sender_before,
        "newbalanceOrig": sender_after,
        "oldbalanceDest": receiver_before,
        "newbalanceDest": receiver_after
    }

# --- Sidebar or main form ---
st.subheader("📥 Enter Transaction Details Below")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox("Transaction Type", ["TRANSFER"])
        transaction_time = st.selectbox("Transaction Time of Day", ["Midnight", "Morning", "Afternoon", "Evening"])
        amount = st.number_input("Amount ($)", min_value=0.0, format="%.2f")
        
    with col2:
        sender_before = st.number_input("Sender Balance Before ($)", min_value=0.0, format="%.2f")
        sender_after = st.number_input("Sender Balance After ($)", min_value=0.0, format="%.2f")
        receiver_before = st.number_input("Receiver Balance Before ($)", min_value=0.0, format="%.2f")
        receiver_after = st.number_input("Receiver Balance After ($)", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("🔍 Predict Fraud")

# Button to auto-generate random test input
if st.button("🎲 Generate Random Test Transaction"):
    test = generate_sample_transaction()
    st.success("✅ Sample data generated!")
    
    # Refill the form with sample values (user has to copy them manually)
    st.write("Please copy the following values into the form manually to test:")
    st.code(f"""
Transaction Type: {test['type']}
Transaction Time of Day: {test['step']}
Amount: {test['amount']}
Sender Balance Before: {test['oldbalanceOrg']}
Sender Balance After: {test['newbalanceOrig']}
Receiver Balance Before: {test['oldbalanceDest']}
Receiver Balance After: {test['newbalanceDest']}
    """)

# If form submitted, predict
if submitted:
    # Encoding time of day manually
    time_mapping = {"Midnight": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
    time_encoded = time_mapping.get(transaction_time, 0)

    # Features in the order model expects
    features = np.array([
        amount,
        sender_before,
        sender_after,
        receiver_before,
        receiver_after,
        time_encoded
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Probability of Fraud: {proba:.2%})")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and CatBoost")

