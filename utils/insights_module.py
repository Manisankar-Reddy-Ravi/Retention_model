import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib
from utils.summary_generator import generate_customer_summary_tab2

# Load the data for insights
df_insights = pd.read_csv("data/Churn_full.csv")
df_insights["Policy_Expiry_Date"] = pd.to_datetime(df_insights["Policy_Expiry_Date"], errors='coerce')

# Load model and encoders
model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

# Apply encoding if necessary
def encode_df(input_df):
    for col in ['Coverage_Type', 'Billing_Method', 'Payment_Method']:
        encoder = encoders.get(col)
        if encoder and col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except Exception as e:
                st.warning(f"Encoding error for {col}: {e}")
    return input_df

def insights_page():
    st.markdown("### 📊 Insights & Actionable Items")
    st.write("Explore key scenarios and visualize important patterns contributing to customer churn.")

    col1, col2, col3, col4 = st.columns(4)

    scenario_data = None
    scenario_title = ""
    scenario_feature = ""
    today = pd.Timestamp.today()

    # ---- BUTTON 1: Policy Expiry in Next 60 Days ----
    with col1:
        if st.button("📅 Imminent Policy Expirations", use_container_width=True):
            scenario_title = "Customers with Policies Expiring in Next 60 Days"
            scenario_feature = "Policy_Tenure_Months"
            scenario_data = df_insights[
                (df_insights["Policy_Expiry_Date"] > today) &
                (df_insights["Policy_Expiry_Date"] <= today + pd.Timedelta(days=60))
            ]

    # ---- BUTTON 2: Late Payments ----
    with col2:
        if st.button("💳 High Payment Delinquency", use_container_width=True):
            scenario_title = "Customers with Late Payments"
            scenario_feature = "Late_Payment_Count"
            scenario_data = df_insights[df_insights["Late_Payment_Count"] > 0]

    # ---- BUTTON 3: Premium Increased ----
    with col3:
        if st.button("📈 Significant Premium Increase", use_container_width=True):
            scenario_title = "Customers with Premium Increase > 5%"
            scenario_feature = "Premium_Change_Percent_Last_Renewal"
            scenario_data = df_insights[df_insights["Premium_Change_Percent_Last_Renewal"] > 5]

    # ---- BUTTON 4: Claims Filed Recently ----
    with col4:
        if st.button("📂 Recent Claims Activity", use_container_width=True):
            scenario_title = "Customers Who Filed Claims More Than Once"
            scenario_feature = "Claims_Count_Lifetime"
            scenario_data = df_insights[df_insights["Claims_Count_Lifetime"] > 1]

    # ---- Show Scenario Data, Plots, and Downloads ----
    if scenario_data is not None and not scenario_data.empty:
        st.markdown(f"#### {scenario_title}")
        st.write(f"Records found: {len(scenario_data)}")

        # EDA Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        if pd.api.types.is_numeric_dtype(scenario_data[scenario_feature]):
            sns.histplot(scenario_data[scenario_feature], bins=20, kde=True, ax=ax)
            ax.set_title(f"Distribution of {scenario_feature}")
        else:
            sns.countplot(y=scenario_data[scenario_feature], ax=ax)
            ax.set_title(f"Counts of {scenario_feature}")
        st.pyplot(fig)

        # Prediction for scenario data
        st.markdown("#### 🔍 Top 5 Customers with High Retention Probability")
        try:
            prediction_df = scenario_data.copy()
            prediction_input = prediction_df[[
                "Address_Change_Flag", "VIN_Validated", "Policy_Tenure_Months", "Coverage_Type",
                "Deductibles", "Has_Multi_Policy", "Loyalty_Program_Enrollment", "Billing_Method",
                "Payment_Method", "Discount_Count", "Premium_Change_Percent_Last_Renewal",
                "Late_Payment_Count", "Auto_Renew_Enabled", "Claims_Count_Lifetime",
                "At_Fault_Accident_Count", "Claim_Satisfaction_Score", "Interaction_Score", "NPS",
                "Complaint_Count", "Sentiment_Score"
            ]].copy()
            prediction_input = encode_df(prediction_input)
            churn_probs = model.predict_proba(prediction_input)[:, 1]
            prediction_df['Retention_Probability (%)'] = (1 - churn_probs) * 100
            top5 = prediction_df.sort_values('Retention_Probability (%)', ascending=False).head(5)
            st.dataframe(top5[["Customer_ID", "Retention_Probability (%)", scenario_feature]])

            # Download Button for retained customers in selected scenario
            st.markdown("#### 📥 Download Retained Customers (Filtered by Scenario)")
            full_encoded = encode_df(scenario_data.copy())
            churn_probs_all = model.predict_proba(full_encoded[[
                "Address_Change_Flag", "VIN_Validated", "Policy_Tenure_Months", "Coverage_Type",
                "Deductibles", "Has_Multi_Policy", "Loyalty_Program_Enrollment", "Billing_Method",
                "Payment_Method", "Discount_Count", "Premium_Change_Percent_Last_Renewal",
                "Late_Payment_Count", "Auto_Renew_Enabled", "Claims_Count_Lifetime",
                "At_Fault_Accident_Count", "Claim_Satisfaction_Score", "Interaction_Score", "NPS",
                "Complaint_Count", "Sentiment_Score"
            ]])
            full_encoded['Retention_Probability (%)'] = (1 - churn_probs_all[:, 1]) * 100
            retained_final = full_encoded[full_encoded['Retention_Probability (%)'] > 50]
            csv_download = retained_final.to_csv(index=False).encode('utf-8')
            st.download_button("Download Retained Customers CSV", data=csv_download,
                               file_name="retained_customers_filtered.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

        # Summary using LLM
        st.markdown("#### 🧠 AI Summary")
        if st.button("Generate Summary for This Scenario"):
            try:
                first_row = scenario_data.head(1).to_dict(orient="records")[0]
                summary = generate_customer_summary_tab2(first_row)
                st.info(summary)
            except Exception as e:
                st.warning(f"Summary generation failed: {e}")
    elif scenario_data is not None:
        st.warning("No data available for selected scenario.")
