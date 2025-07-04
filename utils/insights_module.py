import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import joblib
from utils.summary_generator import generate_customer_summary_tab2

# Load data and models
df_insights = pd.read_csv("data/Churn_full_2.csv")
df_insights["Policy_Expiry_Date"] = pd.to_datetime(df_insights["Policy_Expiry_Date"], errors='coerce')
df_insights["Claim_Closed_Date"] = pd.to_datetime(df_insights.get("Claim_Closed_Date"), errors='coerce')
model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

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
    st.markdown("<h3 style='text-align:center;'>📊 Business Insights & Customer Scenarios</h3>", unsafe_allow_html=True)
    st.write("Explore data-driven churn risk patterns and AI-generated customer insights.")

    # Uniform button CSS
    st.markdown("""
        <style>
            div[data-testid="column"] > div > button {
                height: 60px !important;
                width: 100% !important;
                font-size: 15px !important;
                font-weight: 600 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    scenario_data, scenario_title, scenario_feature, eda_plot_type, filename_prefix = None, "", "", "", ""
    today = pd.Timestamp.today()

    if "scenario_selected" not in st.session_state:
        st.session_state.scenario_selected = False
    if "show_top5" not in st.session_state:
        st.session_state.show_top5 = False

    with col1:
        if st.button("📅 Policy Expiring in 90 Days"):
            st.session_state.scenario_selected = "policy"
            st.session_state.show_top5 = False

    with col2:
        if st.button("💳 Late Payment Insights"):
            st.session_state.scenario_selected = "late"
            st.session_state.show_top5 = False

    with col3:
        if st.button("📈 Premium Increase by five percent"):
            st.session_state.scenario_selected = "premium"
            st.session_state.show_top5 = False

    with col4:
        if st.button("📂 Claims Closed in 90 Days"):
            st.session_state.scenario_selected = "claims"
            st.session_state.show_top5 = False

    if st.session_state.scenario_selected:
        selected = st.session_state.scenario_selected
        if selected == "policy":
            scenario_title = "Customers with Policies Expiring in Next 90 Days"
            scenario_feature = "Coverage_Type"
            eda_plot_type = "bar"
            filename_prefix = "policy_expiring"
            scenario_data = df_insights[
                (df_insights["Policy_Expiry_Date"] > today) &
                (df_insights["Policy_Expiry_Date"] <= today + pd.Timedelta(days=90))
            ]

        elif selected == "late":
            scenario_title = "Customers with Late Payments"
            scenario_feature = "Late_Payment_Count"
            eda_plot_type = "late_region"
            filename_prefix = "late_payments"
            scenario_data = df_insights[df_insights["Late_Payment_Count"] > 0]

        elif selected == "premium":
            scenario_title = "Customers with Premium Increase > 5%"
            scenario_feature = "Premium_Change_Percent_Last_Renewal"
            eda_plot_type = "premium_line"
            filename_prefix = "premium_increase"
            scenario_data = df_insights[df_insights["Premium_Change_Percent_Last_Renewal"] > 5]

        elif selected == "claims":
            scenario_title = "Customers with Claims Closed in Last 90 Days"
            scenario_feature = "Claim_Outcome"
            eda_plot_type = "claim_outcome"
            filename_prefix = "claims_closed"
            scenario_data = df_insights[
                (df_insights["Claim_Closed_Date"] >= today - pd.Timedelta(days=90)) &
                (df_insights["Claim_Closed_Date"] <= today)
            ]

        if scenario_data is not None and not scenario_data.empty:
            st.markdown(f"<h4 style='color:#0E1117; font-size:22px;'>🧾 <strong>{scenario_title}</strong></h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px; font-weight:bold;'>Records found: <span style='color:#FA4B3E'>{len(scenario_data)}</span></p>", unsafe_allow_html=True)

            # ---- EDA Plot ----
            st.markdown("#### 📊 Exploratory Data Analysis")
            fig, ax = plt.subplots(figsize=(8, 4))

            if eda_plot_type == "bar":
                sns.countplot(data=scenario_data, x=scenario_feature, ax=ax)
                ax.set_title(f"{scenario_feature} Distribution")

            elif eda_plot_type == "late_region":
                if "State" in scenario_data.columns:
                    region_late = scenario_data.groupby("State")["Late_Payment_Count"].sum().reset_index()
                    sns.barplot(data=region_late, x="State", y="Late_Payment_Count", ax=ax)
                    ax.set_title("Total Late Payments by State")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            elif eda_plot_type == "premium_line":
                premium_counts = scenario_data["Premium_Change_Percent_Last_Renewal"].round().value_counts().sort_index()
                ax.plot(premium_counts.index, premium_counts.values, marker='o', linestyle='-')
                ax.set_title("Customers by Premium Increase (%)")
                ax.set_xlabel("Premium Increase (%)")
                ax.set_ylabel("Customer Count")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            elif eda_plot_type == "claim_outcome":
                sns.countplot(data=scenario_data, x="Claim_Outcome", order=["Approved", "Declined"], ax=ax)
                ax.set_title("Claim Outcome: Approved vs Declined")

            st.pyplot(fig)

            # ---- AI Summary ----
            st.markdown("#### 🧠 AI-Generated Summary")
            try:
                row = scenario_data.head(1).to_dict(orient="records")[0]
                total_count = len(df_insights)
                scenario_count = len(scenario_data)
                percentage = round((scenario_count / total_count) * 100, 2)

                prediction_input = encode_df(scenario_data[[ 
                    "Address_Change_Flag", "VIN_Validated", "Policy_Tenure_Months", "Coverage_Type",
                    "Deductibles", "Has_Multi_Policy", "Loyalty_Program_Enrollment", "Billing_Method",
                    "Payment_Method", "Discount_Count", "Premium_Change_Percent_Last_Renewal",
                    "Late_Payment_Count", "Auto_Renew_Enabled", "Claims_Count_Lifetime",
                    "At_Fault_Accident_Count", "Claim_Satisfaction_Score", "Interaction_Score", "NPS",
                    "Complaint_Count", "Sentiment_Score"
                ]])
                churn_probs = model.predict_proba(prediction_input)[:, 1]
                avg_retention = round((1 - churn_probs.mean()) * 100, 2)

                stats = {
                    "count": scenario_count,
                    "percentage": percentage,
                    "avg_retention": avg_retention
                }

                summary = generate_customer_summary_tab2(row, scenario_title, stats)
                st.markdown(f"<div style='font-size:15px; font-weight:400;'>{summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Summary generation failed: {e}")

            # ---- Show Top 5 Customers ----
            if st.button("🎯 Show Top 5 Lowest Retention Customers"):
                st.session_state.show_top5 = True

            if st.session_state.show_top5:
                st.markdown("#### 🧮 Predicted Retention Probabilities")
                prediction_df = scenario_data.copy()
                prediction_df['Retention_Probability (%)'] = (1 - churn_probs) * 100

                top5 = prediction_df.sort_values('Retention_Probability (%)').head(5)
                st.markdown("#### 🥇 Top 5 Customers (Lowest Retention Probability)")
                st.dataframe(top5[["Customer_ID", "Retention_Probability (%)", scenario_feature]])

                # --- Updated Download Button with required columns ---
                csv_columns = ["Customer_ID", scenario_feature, "Policy_Number", "Policy_Expiry_Date", "State", "Retention_Probability (%)"]
                csv_download = prediction_df[csv_columns].sort_values("Retention_Probability (%)").to_csv(index=False).encode('utf-8')
                st.markdown("#### 📥 Download All Scenario Records")
                st.download_button("⬇️ Download CSV", data=csv_download, file_name=f"{filename_prefix}_customers.csv", mime="text/csv")

        else:
            st.warning("⚠️ No data available for the selected scenario.")
