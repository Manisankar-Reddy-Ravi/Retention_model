import streamlit as st
import pandas as pd
import joblib
from utils.summary_generator import generate_customer_summary
from pages.insights_module import insights_page  # You should move the insights logic into insights_module.py

# --- Page Config ---
st.set_page_config(page_title="Auto Insurance Retention Model", layout="wide")

# --- Load model and encoders ---
model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

# --- Load dataset ---
df = pd.read_csv("data/Churn_full.csv")

# --- Header with logo and title ---
logo_col, title_col = st.columns([1, 4])
with logo_col:
    try:
        st.image("images/logo.png", width=120)
    except:
        st.warning("⚠️ Logo not found in 'images/' directory.")
with title_col:
    st.markdown("<h1 style='margin-top: 20px;'>Auto Insurance Churn Predictor</h1>", unsafe_allow_html=True)

# --- Tab Layout ---
tab1, tab2 = st.tabs(["Predict Churn", "Insights & Actionable Items"])

with tab1:
    # --- Form Starts ---
    with st.form("churn_form"):
        # 📄 Section 1: Customer & Policy Information
        st.markdown("### 📄 Customer & Policy Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            address_change_flag = st.selectbox("Address Change Flag", [0, 1])
        with col2:
            vin_validated = st.selectbox("VIN Validated", [0, 1])
        with col3:
            policy_tenure = st.number_input("Policy Tenure (Months)", min_value=0)

        col4, col5, col6 = st.columns(3)
        with col4:
            coverage_type = st.selectbox("Coverage Type", ["Liability", "Comprehensive"])
        with col5:
            deductibles = st.number_input("Deductibles", min_value=0)
        with col6:
            multi_policy = st.selectbox("Has Multi Policy", [0, 1])

        # 💳 Section 2: Billing & Payment Details
        st.markdown("### 💳 Billing & Payment Details")
        col7, col8, col9 = st.columns(3)
        with col7:
            loyalty_program = st.selectbox("Loyalty Program Enrollment", [0, 1])
        with col8:
            billing_method = st.selectbox("Billing Method", ["Auto-Pay", "Manual"])
        with col9:
            payment_method = st.selectbox("Payment Method", ["Card", "Bank", "UPI"])

        col10, col11, col12 = st.columns(3)
        with col10:
            discount_count = st.number_input("Discount Count", min_value=0)
        with col11:
            premium_change = st.number_input("Premium % Change", value=0.0)
        with col12:
            late_payments = st.number_input("Late Payment Count", min_value=0)

        # 🛠️ Section 3: Claims & Risk Profile
        st.markdown("### 🛠️ Claims & Risk Profile")
        col13, col14, col15 = st.columns(3)
        with col13:
            auto_renew = st.selectbox("Auto Renew Enabled", [0, 1])
        with col14:
            claims_lifetime = st.number_input("Claims Lifetime Count", min_value=0)
        with col15:
            fault_accidents = st.number_input("At-Fault Accident Count", min_value=0)

        # 🧠 Section 4: Engagement & Sentiment Metrics
        st.markdown("### 🧠 Engagement & Sentiment Metrics")
        col16, col17, col18 = st.columns(3)
        with col16:
            satisfaction_score = st.slider("Claim Satisfaction Score", 0, 100, 50)
        with col17:
            interaction_score = st.slider("Interaction Score", 0.0, 1.0, 0.5)
        with col18:
            nps = st.number_input("Net Promoter Score (NPS)", min_value=-100, max_value=100)

        col19, col20 = st.columns(2)
        with col19:
            complaint_count = st.number_input("Complaint Count", min_value=0)
        with col20:
            sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

        # --- Submit Button ---
        submitted = st.form_submit_button("🔍 Predict Churn")

    # --- Prediction Logic ---
    if submitted:
        input_dict = {
            "Address_Change_Flag": address_change_flag,
            "VIN_Validated": vin_validated,
            "Policy_Tenure_Months": policy_tenure,
            "Coverage_Type": coverage_type,
            "Deductibles": deductibles,
            "Has_Multi_Policy": multi_policy,
            "Loyalty_Program_Enrollment": loyalty_program,
            "Billing_Method": billing_method,
            "Payment_Method": payment_method,
            "Discount_Count": discount_count,
            "Premium_Change_Percent_Last_Renewal": premium_change,
            "Late_Payment_Count": late_payments,
            "Auto_Renew_Enabled": auto_renew,
            "Claims_Count_Lifetime": claims_lifetime,
            "At_Fault_Accident_Count": fault_accidents,
            "Claim_Satisfaction_Score": satisfaction_score,
            "Interaction_Score": interaction_score,
            "NPS": nps,
            "Complaint_Count": complaint_count,
            "Sentiment_Score": sentiment_score
        }

        input_df = pd.DataFrame([input_dict])

        # Encode categorical features
        for col in ['Coverage_Type', 'Billing_Method', 'Payment_Method']:
            encoder = encoders.get(col)
            if encoder:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except Exception as e:
                    st.error(f"Encoding error in column '{col}': {e}")

        # Predict churn
        churn_score = model.predict_proba(input_df)[0][1]
        churn_percent = round(churn_score * 100, 2)

        # Generate AI summary
        summary = generate_customer_summary(input_dict)

        st.markdown(f"<h4 style='text-align: center;'>Predicted Churn Probability: <span style='color:#f63366'>{churn_percent}%</span></h4>", unsafe_allow_html=True)
        st.progress(int(churn_percent))

        st.markdown("### 📜 Customer Summary")
        st.info(summary)

with tab2:
    insights_page()
