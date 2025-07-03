from openai_resourse import get_openai_response

def generate_customer_summary(customer_data: dict) -> str:
    details = "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in customer_data.items()])
    messages = [
        {"role": "system", "content": "You are an insurance expert. Summarize customer details briefly and clearly."},
        {"role": "user", "content": f"Given the following customer details, generate a short summary:\n\n{details}"}
    ]
    return get_openai_response(messages)
def generate_scenario_summary(scenario_title: str, total_customers: int, high_retention_customers: int, retention_percentage: float) -> str:
    summary_prompt = (
        f"As an insurance domain expert, analyze the scenario titled '{scenario_title}' involving {total_customers} customers. "
        f"{high_retention_customers} of them ({retention_percentage:.2f}%) are identified with high retention probability. "
        f"Generate a brief strategic summary including potential actions or considerations for this customer group."
    )
    messages = [
        {"role": "system", "content": "You are a senior insurance analyst. Provide a concise and insightful analysis of the customer scenario."},
        {"role": "user", "content": summary_prompt}
    ]
    return get_openai_response(messages)
