import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from utils.summary_generator import generate_customer_summary, generate_scenario_summary

# Load the data for insights
df_insights = pd.read_csv("data/Churn_full.csv")
df_insights["Policy_Expiry_Date"] = pd.to_datetime(df_insights["Policy_Expiry_Date"], errors='coerce')

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
        if st.button("🗓️ Policy Expiry < 60 days", use_container_width=True):
            scenario_title = "Customers with Policies Expiring in Next 60 Days"
            scenario_feature = "Policy_Tenure_Months"
            scenario_data = df_insights[
                (df_insights["Policy_Expiry_Date"] > today) &
                (df_insights["Policy_Expiry_Date"] <= today + pd.Timedelta(days=60))
            ]

    # ---- BUTTON 2: Late Payments ----
    with col2:
        if st.button("💸 Late Payments", use_container_width=True):
            scenario_title = "Customers with Late Payments"
            scenario_feature = "Late_Payment_Count"
            scenario_data = df_insights[df_insights["Late_Payment_Count"] > 0]

    # ---- BUTTON 3: Premium Increased ----
    with col3:
        if st.button("📈 Premium Increased", use_container_width=True):
            scenario_title = "Customers with Recent Premium Increases"
            scenario_feature = "Premium_Change_Percent_Last_Renewal"
            scenario_data = df_insights[df_insights["Premium_Change_Percent_Last_Renewal"] > 0]

    # ---- BUTTON 4: Claims Filed Recently ----
    with col4:
        if st.button("📄 Claims Filed Recently", use_container_width=True):
            scenario_title = "Customers Who Filed Claims"
            scenario_feature = "Claims_Count_Lifetime"
            scenario_data = df_insights[df_insights["Claims_Count_Lifetime"] > 0]

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

        # Show sample data
        st.markdown("##### Sample Data")
        st.dataframe(scenario_data.head(10))

        # Top 5 churn prediction mock (if model integrated, you can add real prediction)
        st.markdown("#### 🔍 Top 5 Customers with High Retention Probability")
        if "Churned" in scenario_data.columns:
            top5 = scenario_data.sort_values("Churned").head(5)
            top5 = top5.assign(**{"Retention Probability (%)": (1 - top5["Churned"]) * 100})
            st.dataframe(top5[["Customer_ID", "Retention Probability (%)", scenario_feature]])
        else:
            st.warning("'Churned' column not available for probability sorting.")

        # Download button
        st.markdown("#### 📥 Download Filtered Data")
        csv_data = scenario_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_data, file_name="filtered_insights.csv", mime="text/csv")

        # Summary using LLM
        st.markdown("#### 🧠 AI Summary")
        if st.button("Generate Summary for This Scenario"):
            top_retained = scenario_data.sort_values("Churned").head(5)
            summary = generate_scenario_summary(
                scenario_title=scenario_title,
                total_customers=len(scenario_data),
                high_retention_customers=len(top_retained),
                retention_percentage=(1 - top_retained["Churned"]).mean() * 100
            )
            st.info(summary)

    elif scenario_data is not None:
        st.warning("No data available for selected scenario.")
