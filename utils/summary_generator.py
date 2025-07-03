from openai_resourse import get_openai_response

def generate_customer_summary(customer_data: dict) -> str:
    details = "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in customer_data.items()])
    messages = [
        {"role": "system", "content": "You are an insurance expert. Summarize customer details briefly and clearly."},
        {"role": "user", "content": f"Given the following customer details, generate a short summary:\n\n{details}"}
    ]
    return get_openai_response(messages)


# ✅ NEW FUNCTION FOR INSIGHTS TAB
def generate_customer_summary_tab2(sample_data: dict) -> str:
    details = "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in sample_data.items()])
    messages = [
        {
            "role": "system",
            "content": "You are a data analyst with expertise in customer churn. Generate a brief scenario-based summary for business users."
        },
        {
            "role": "user",
            "content": f"Given this filtered dataset row, generate a scenario summary:\n\n{details}"
        }
    ]
    return get_openai_response(messages)
