from openai_resourse import get_openai_response

# Main Customer Summary
def generate_customer_summary(customer_data: dict) -> str:
    details = "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in customer_data.items()])
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert insurance analyst. Summarize the customer's insurance profile, "
                "highlighting any key risk factors or indicators of churn. Be concise and professional."
            )
        },
        {
            "role": "user",
            "content": (
                f"Given the following customer details, write a short 3–4 line summary:\n\n{details}"
            )
        }
    ]
    
    return get_openai_response(messages)


# Enhanced Scenario-based AI Summary
def generate_customer_summary_tab2(sample_data: dict, scenario_title: str, stats: dict) -> str:
    details = "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in sample_data.items()])
    count = stats.get("count")
    percent = stats.get("percentage")
    retention = stats.get("avg_retention")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior data analyst specializing in customer churn for insurance companies. "
                "Provide an insightful summary of the scenario below. Focus on trends, business impact, and potential strategies."
            )
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario_title}\n\n"
                f"There are {count} customers falling under this scenario, accounting for {percent}% of the overall data. "
                f"The average predicted retention probability is {retention}%.\n\n"
                f"A sample customer from this group is:\n{details}\n\n"
                f"Based on this, write a short summary paragraph highlighting the customer behavior, possible churn risks, "
                f"and what strategic insights the business can gain from this group."
            )
        }
    ]

    return get_openai_response(messages)
