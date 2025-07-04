import streamlit as st
import pandas as pd
import joblib
from utils.summary_generator import generate_customer_summary
from utils.insights_module import insights_page

# Page Configuration
st.set_page_config(page_title="Auto Insurance Churn Predictor", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .stButton>button {
        background-color: #004080 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem;
        font-weight: 600 !important;
        font-size: 16px !important;
        margin-top: 10px;
    }

    .form-section {
        margin-bottom: 2rem;
        padding-top: 1rem;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

# Load data
df = pd.read_csv("data/Churn_full_2.csv")

# --- Logo and Title Alignment
logo_col, title_col = st.columns([1, 6])
with logo_col:
    try:
        st.image("images/logo.png", width=100)
    except:
        st.warning("⚠️ Logo not found.")
with title_col:
    st.markdown(
        "<h1 style='padding-top: 1.2rem; color: #004080;'>Auto Insurance Churn Predictor</h1>",
        unsafe_allow_html=True
    )

# --- Tabs
tab1, tab2 = st.tabs(["🔍 Predict Churn", "📊 Insights & Actions"])

# -------------------- TAB 1 -------------------- #
with tab1:
    st.markdown("<h3>📄 Customer & Policy Information</h3>", unsafe_allow_html=True)
    st.markdown('<div class="form-section">', unsafe_allow_html=True)

    with st.form("churn_form"):
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

        st.markdown('</div><h3>💳 Billing & Payment Details</h3><div class="form-section">', unsafe_allow_html=True)

        col7, col8, col9 = st.columns(3)
        with col7:
            loyalty_program = st.selectbox("Loyalty Program Enrollment", [0, 1])
        with col8:
            billing_method = st.selectbox("Billing Method", ["Auto-Pay", "Manual"])
        with col9:
            payment_display = st.selectbox("Payment Method", ["Card", "Bank"])
            payment_method = "ACH" if payment_display == "Bank" else "Card"


        col10, col11, col12 = st.columns(3)
        with col10:
            discount_count = st.number_input("Discount Count", min_value=0)
        with col11:
            premium_change = st.number_input("Premium % Change", value=0.0)
        with col12:
            late_payments = st.number_input("Late Payment Count", min_value=0)

        st.markdown('</div><h3>🛠️ Claims & Risk Profile</h3><div class="form-section">', unsafe_allow_html=True)

        col13, col14, col15 = st.columns(3)
        with col13:
            auto_renew = st.selectbox("Auto Renew Enabled", [0, 1])
        with col14:
            claims_lifetime = st.number_input("Claims Lifetime Count", min_value=0)
        with col15:
            fault_accidents = st.number_input("At-Fault Accident Count", min_value=0)

        st.markdown('</div><h3>🧠 Engagement & Sentiment Metrics</h3><div class="form-section">', unsafe_allow_html=True)

        col16, col17, col18 = st.columns(3)
        with col16:
            satisfaction_score = st.slider("Claim Satisfaction Score", 0, 100, 50)
        with col17:
            interaction_score = st.slider("Interaction Score", 0.0, 1.0, 0.5)
        with col18:
            sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

        col19, col20 = st.columns(2)
        with col19:
            complaint_count = st.number_input("Complaint Count", min_value=0)
        with col20:
            nps = st.number_input("Net Promoter Score (NPS)", min_value=0, max_value=10)

        # Submit Button in Center
        st.markdown("</div>", unsafe_allow_html=True)
        center_col = st.columns([4, 1, 4])[1]
        with center_col:
            submitted = st.form_submit_button("Submit")

    # On Form Submission
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
        for col in ['Coverage_Type', 'Billing_Method', 'Payment_Method']:
            encoder = encoders.get(col)
            if encoder:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except Exception as e:
                    st.error(f"Encoding error in column '{col}': {e}")

        churn_score = model.predict_proba(input_df)[0][1]
        churn_percent = churn_score * 100
        churn_progress = int(churn_percent)

        st.markdown("### 📊 Model Results")

        # Progress Bar with Risk Levels
        import streamlit.components.v1 as components

        churn_percentage = float(round(churn_score * 100, 2))

        # Determine risk level
        if churn_percentage >= 80:
            risk_level = "High"
            color = "#e63946"   # Red
        elif churn_percentage >= 40:
            risk_level = "Medium"
            color = "#f4a261"   # Orange
        else:
            risk_level = "Low"
            color = "#2a9d8f"   # Green

        # Render colored progress bar using HTML + CSS
        st.markdown(f"""
        <div style="margin-top: 10px;">
        <div style="font-weight: 600;">Churn Probability: {churn_percentage:.2f}%</div>
        <div style="height: 24px; width: 100%; background-color: #e0e0e0; border-radius: 8px; overflow: hidden;">
            <div style="height: 100%; width: {churn_percentage}%; background-color: {color}; text-align: center; line-height: 24px; color: white; font-weight: bold;">
            {risk_level}
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(churn_progress)


        if churn_percent >= 80:
            st.error("🚨 High Risk of Churn")
        elif churn_percent >= 40:
            st.warning("⚠️ Moderate Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")

        # Summary
        summary = generate_customer_summary(input_dict)
        st.markdown("<h4 style='text-align:left;'>📜 Customer Summary</h4>", unsafe_allow_html=True)
        st.write(summary)

# -------------------- TAB 2 -------------------- #
with tab2:
    insights_page()
