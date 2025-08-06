import base64
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------- Background Image Setup --------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)),
                    url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: white !important;
        font-weight: 600 !important;
    }}

    .stSelectbox div[data-baseweb="select"],
    .stTextInput input,
    .stNumberInput input {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border-radius: 8px;
        padding: 6px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------- Load Model --------------
pipeline = joblib.load("src/models/fraud_catboost_pipeline.pkl")

# -------------- UI Config --------------
st.set_page_config(page_title="Fraud Detection App", layout="centered")
set_background("images/fraud_image2.jpg")

st.markdown("<h1 style='text-align: center;'>💸 Real-Time Fraud Detection App</h1>", unsafe_allow_html=True)
st.markdown("### Please enter transaction details below:")

# -------------- Inputs --------------
transaction_type = st.selectbox("Transaction Type", options=["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
time_period = st.selectbox("Transaction Time of Day", options=["Midnight", "Morning", "Afternoon", "Night"],
                           help="Approximate time of transaction")

amount = st.number_input("Amount", min_value=0.0, value=0.0)
oldbalanceOrg = st.number_input("Sender Balance Before", min_value=0.0, value=0.0)
newbalanceOrig = st.number_input("Sender Balance After", min_value=0.0, value=0.0)
oldbalanceDest = st.number_input("Receiver Balance Before", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Receiver Balance After", min_value=0.0, value=0.0)

# -------------- Prepare Input --------------
input_data = pd.DataFrame({
    "type": [transaction_type],
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest],
    "time_period": [time_period]
})

# -------------- Predict --------------
if st.button("🧠 Predict Fraud"):
    prob = pipeline.predict_proba(input_data)[0][1]
    prediction = pipeline.predict(input_data)[0]

    st.markdown("## 🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted to be **FRAUDULENT**.\n\nFraud Probability: `{prob:.2f}`")
    else:
        st.success(f"✅ This transaction is predicted to be **LEGITIMATE**.\n\nFraud Probability: `{prob:.2f}`")

    # Risk Levels
    st.markdown("### 📊 Risk Assessment")
    if prob >= 0.85:
        st.warning("🔴 **Very High Risk:** Immediate investigation recommended.")
    elif prob >= 0.6:
        st.warning("🟠 **Moderate to High Risk:** Please review carefully.")
    elif prob >= 0.3:
        st.info("🟡 **Low to Moderate Risk:** Some unusual behavior detected.")
    else:
        st.success("🟢 **Low Risk:** No signs of fraud detected.")

    # Explanation
    st.markdown("### 🧠 Why this prediction?")
    try:
        fe_data = pipeline.named_steps['feature_engineer'].transform(input_data)
        processed = pipeline.named_steps['preprocessing'].transform(fe_data)
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        clean_names = [name.replace("scaling__", "").replace("ohe__", "") for name in feature_names]

        importances = pipeline.named_steps['model'].get_feature_importance(type='FeatureImportance')
        contributions = importances * processed[0]

        top_k = 5
        sorted_idx = np.argsort(np.abs(contributions))[::-1][:top_k]

        for idx in sorted_idx:
            st.markdown(f"- **{clean_names[idx]}** ➔ impact: `{contributions[idx]:.4f}`")

        st.info("These features contributed the most to the prediction.")
    except Exception as e:
        st.warning(f"⚠️ Could not explain prediction: {e}")

# -------------- Footer --------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and CatBoost")




